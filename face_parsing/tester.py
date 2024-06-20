import datetime
import os
import time

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn

# from torcheval.metrics import MulticlassAccuracy
import torchmetrics
from data_loader import Data_Loader
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from unet import unet
from utils import *


def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(
            transforms.Resize((imsize, imsize), interpolation=PIL.Image.NEAREST)
        )
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)

    return transform


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    f = dir.split("/")[-1].split("_")[-1]
    print(
        dir,
        len(
            [
                name
                for name in os.listdir(dir)
                if os.path.isfile(os.path.join(dir, name))
            ]
        ),
    )
    for i in range(
        len(
            [
                name
                for name in os.listdir(dir)
                if os.path.isfile(os.path.join(dir, name))
            ]
        )
    ):
        img = str(i) + ".jpg"
        path = os.path.join(dir, img)
        images.append(path)

    return images


class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.n_classes = config.n_classes

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_results_path = config.test_results_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model(n_classes=self.n_classes)

        # build test loader
        self.test_loader = Data_Loader(
            config.test_image_path,
            config.test_label_path,
            config.imsize,
            config.batch_size,
            False,
        ).loader(shuffle=False, num_workers=0)

        # create metrics
        self.metric = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                torchmetrics.classification.MulticlassJaccardIndex(
                    num_classes=self.n_classes
                ),
            ]
        ).cuda()
        # self.acc_metric = MulticlassAccuracy()

    def test(self):
        make_folder(self.test_results_path, "")
        make_folder(self.test_color_label_path, "")
        self.G.load_state_dict(
            torch.load(os.path.join(self.model_save_path, self.model_name))
        )
        self.G.eval()

        for i, (imgs, labels) in enumerate(self.test_loader):
            with torch.no_grad():
                imgs = imgs.cuda()
                labels *= 255.0
                labels = labels.cuda()
                labels_predict = self.G(imgs)
                labels_predict_plain = generate_label_plain(
                    labels_predict, self.imsize, self.n_classes
                )
                labels_predict_color = generate_label(
                    labels_predict, self.imsize, self.n_classes
                )
                for k in range(self.batch_size):
                    cv2.imwrite(
                        os.path.join(
                            self.test_results_path,
                            str(i * self.batch_size + k) + ".png",
                        ),
                        labels_predict_plain[k],
                    )
                    save_image(
                        labels_predict_color[k],
                        os.path.join(
                            self.test_color_label_path,
                            str(i * self.batch_size + k) + ".png",
                        ),
                    )

                # get metrics
                predict = torch.from_numpy(labels_predict_plain.flatten()).cuda()
                target = labels.squeeze().flatten()

                self.metric.update(predict, target)

        print(self.metric.compute())

    def build_model(self, n_classes):
        self.G = unet(n_classes=n_classes).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # print networks
        print(self.G)
