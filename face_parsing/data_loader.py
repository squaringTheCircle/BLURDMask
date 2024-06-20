import json
import os
from enum import Enum, auto

import numpy as np
import torch
import torchvision.datasets as dsets
from PIL import Image
from skimage import morphology as mp
from torch.utils.data import Dataset
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2


class DatasetTypes(Enum):
    CELEBAMASKHQ = auto()
    BLURD3D = auto()
    BLURDSD = auto()
    BLURDBOTH = auto()


class CelebAMaskHQ:
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()

        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):

        for i in range(
            len(
                [
                    name
                    for name in os.listdir(self.img_path)
                    if os.path.isfile(os.path.join(self.img_path, name))
                ]
            )
        ):
            img_path = os.path.join(self.img_path, str(i) + ".jpg")
            label_path = os.path.join(self.label_path, str(i) + ".png")
            print(img_path, label_path)
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):

        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Blurd(Dataset):

    DIR_MAP = {DatasetTypes.BLURD3D: "renders", DatasetTypes.BLURDSD: "sd"}

    def __init__(self, root, transform=None, dataset_type=DatasetTypes.BLURD3D):
        self.root = root
        self.dataset_type = dataset_type
        self.transform = transform
        with open(
            os.path.join(self.root, "rendered_dataset.json"), "r", encoding="utf-8"
        ) as file:
            self.json_data = json.load(file)

        self.dataset = list(self.json_data.keys())

    def __getitem__(self, idx):
        uuid = self.dataset[idx]
        factors = self.json_data[uuid]
        gender = factors["gender"]
        uuid_pth = os.path.join(self.root, gender, uuid)
        if self.dataset_type == DatasetTypes.BLURD3D:
            img_dir = os.path.join(uuid_pth, "renders")
            img_pth = os.path.join(img_dir, "render-bg_0000.png")
        elif self.dataset_type == DatasetTypes.BLURDSD:
            img_dir = os.path.join(uuid_pth, "sd", "images_w_depth_normal")
            img_pth = os.path.join(img_dir, os.listdir(img_dir)[0])
        elif self.dataset_type == DatasetTypes.BLURDBOTH:
            if idx <= len(self.dataset):
                img_dir = os.path.join(uuid_pth, "renders")
                img_pth = os.path.join(img_dir, "render-bg_0000.png")
            else:
                idx -= len(self.dataset)
                img_dir = os.path.join(uuid_pth, "sd", "images_w_depth_normal")
                img_pth = os.path.join(img_dir, os.listdir(img_dir)[0])
        else:
            raise ValueError(
                f"dataset_type must be one of {', '.join([t.name for t in DatasetTypes])}, not {self.dataset_type}"
            )

        img = Image.open(img_pth).convert("RGB")
        label_pth = os.path.join(uuid_pth, "label", "label.png")
        label = Image.open(label_pth)
        if img.size != label.size:
            img = img.resize(label.size)

        # img = v2.functional.pil_to_tensor(img).to(torch.float)
        label = tv_tensors.Mask(label)

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label

    def __len__(self):
        if self.dataset_type == DatasetTypes.BLURDBOTH:
            return len(self.dataset) * 2
        else:
            return len(self.dataset)


class Data_Loader:
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self, shuffle=True, num_workers=2):
        transform_img = self.transform_img(True, True, True, False)
        transform_label = self.transform_label(True, True, False, False)
        dataset = CelebAMaskHQ(
            self.img_path, self.label_path, transform_img, transform_label, self.mode
        )

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
        return loader


class Data_Loader_B(Data_Loader):

    def __init__(self, root, image_size, batch_size, dataset_type):
        self.root = root
        self.batch = batch_size
        self.imsize = image_size
        self.dataset_type = dataset_type

    def loader(self):
        to_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        transform = v2.Compose(
            [
                lambda img, label: (to_image(img), label),
                v2.ColorJitter(brightness=0.5, hue=0.2),
                v2.RandomResizedCrop(size=(self.imsize, self.imsize), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Resize(size=(self.imsize, self.imsize), antialias=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                lambda img, label: (img, label / 255.0),
            ]
        )

        dataset = Blurd(self.root, transform=transform, dataset_type=self.dataset_type)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=True,
            num_workers=2,
            drop_last=False,
        )
        return loader
