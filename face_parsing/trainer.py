import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision.utils import save_image
from unet import unet
from utils import *


class Trainer(object):
    def __init__(self, data_loader, val_loader, config):

        # Data loader
        self.data_loader = data_loader
        self.val_loader = val_loader

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

        self.use_tensorboard = config.use_tensorboard
        self.img_path = config.img_path
        self.label_path = config.label_path
        self.val_img_path = config.val_img_path
        self.val_label_path = config.val_label_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.stop_step = config.stop_step

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.model_saver = SaveBestModel(
            self.model_save_path, maxlen=config.top_save_num
        )
        self.writer = SummaryWriter(
            os.path.join(config.log_path, "tensorboard/training/")
        )

        self.build_model(n_classes=self.n_classes)

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        self.step = 0

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):
            self.step = step

            self.G.train()
            try:
                imgs, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            size = labels.size()
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
            labels_real_plain = labels[:, 0, :, :].cuda()
            labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
            oneHot_size = (size[0], self.n_classes, size[2], size[3])
            # labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            labels_real = torch.zeros(torch.Size(oneHot_size), dtype=torch.float).cuda()
            labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)

            imgs = imgs.cuda()
            # ================== Train G =================== #
            labels_predict = self.G(imgs)

            # Calculate cross entropy loss
            c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
            self.reset_grad()
            c_loss.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(
                    "Elapsed [{}], G_step [{}/{}], Cross_entrophy_loss: {:.4f}".format(
                        elapsed, step + 1, self.total_step, c_loss.data
                    )
                )

            label_batch_predict = generate_label(
                labels_predict, self.imsize, n_classes=self.n_classes
            )
            label_batch_real = generate_label(
                labels_real, self.imsize, n_classes=self.n_classes
            )

            # scalr info on tensorboardX
            self.writer.add_scalar("Loss/Cross_entrophy_loss", c_loss.data, step)

            # image infor on tensorboardX
            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]

            for i in range(1, imgs.shape[0]):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat(
                    [predict_combine, label_batch_predict[i]], 2
                )

            # Sample images
            if (step + 1) % self.sample_step == 0:
                self.writer.add_image(
                    "imresult/img", (img_combine.data + 1) / 2.0, step
                )
                self.writer.add_image("imresult/real", real_combine, step)
                self.writer.add_image("imresult/predict", predict_combine, step)
                labels_sample = self.G(imgs)
                labels_sample = generate_label(
                    labels_sample, self.imsize, n_classes=self.n_classes
                )
                # labels_sample = torch.from_numpy(labels_sample)
                save_image(
                    denorm(labels_sample.data),
                    os.path.join(self.sample_path, "{}_predict.png".format(step + 1)),
                )

            # validate and save model
            if (step + 1) % model_save_step == 0:
                val_score = self.val()
                self.model_saver.put(
                    val_score, step=step, state_dict=self.G.state_dict()
                )
                self.model_saver.save()
                self.writer.add_scalar("Loss/Cross_entrophy_val_loss", val_score, step)
                print(
                    "Elapsed [{}], G_step [{}/{}], validation_loss: {:.4f}".format(
                        elapsed, step + 1, self.total_step, val_score
                    )
                )

                # Stop training if no improvement
                if self.model_saver.last_queue_update > self.stop_step:
                    break

    def val(self):
        self.G.eval()
        val_losses = []
        imgs = None
        for imgs, labels in iter(self.val_loader):
            with torch.no_grad():
                size = labels.size()
                labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                labels_real_plain = labels[:, 0, :, :].cuda()
                labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
                oneHot_size = (size[0], self.n_classes, size[2], size[3])
                labels_real = torch.zeros(
                    torch.Size(oneHot_size), dtype=torch.float
                ).cuda()
                labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)

                imgs = imgs.cuda()
                labels_predict = self.G(imgs)

                c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                val_losses.append(c_loss.item())

        # Log last Val images
        if imgs is not None:
            label_batch_predict = generate_label(
                labels_predict, self.imsize, n_classes=self.n_classes
            )
            label_batch_real = generate_label(
                labels_real, self.imsize, n_classes=self.n_classes
            )

            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]

            for i in range(1, imgs.shape[0]):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat(
                    [predict_combine, label_batch_predict[i]], 2
                )

            self.writer.add_image(
                "imresult/val_img", (img_combine.data + 1) / 2.0, self.step
            )
            self.writer.add_image("imresult/val_real", real_combine, self.step)
            self.writer.add_image("imresult/val_predict", predict_combine, self.step)

        return np.mean(val_losses)

    def build_model(self, n_classes):

        self.G = unet(n_classes=n_classes).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            self.g_lr,
            [self.beta1, self.beta2],
        )

        # print networks
        print(self.G)

    def build_tensorboard(self):
        from logger import Logger

        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(
            torch.load(
                os.path.join(
                    self.model_save_path, "{}_G.pth".format(self.pretrained_model)
                )
            )
        )
        print("loaded trained models (step: {})..!".format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, "real.png"))
