import torch
from torch.utils.data import Dataset
from . import dataset_utils
import numpy as np
import torchvision.datasets
import torchvision.transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import joblib
from urllib.request import urlretrieve


@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device, root=None):
        super().__init__(root=root, train=cfg.data.train, download=cfg.data.download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1, 3)
        self.data = self.data.transpose(2, 3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.use_augm
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "CIFAR10", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "CIFAR10", "processed")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


@dataset_utils.register_dataset
class DiscreteMNIST(torchvision.datasets.MNIST):
    def __init__(self, cfg, device, root=None):
        super().__init__(root=root, train=cfg.data.train, download=cfg.data.download)
        # self.data = torch.from_numpy(self.data) # (N, H, W, C)
        self.data = self.data.to(device).view(-1, 1, 28, 28)

        self.targets = torch.from_numpy(np.array(self.targets))

        self.random_flips = cfg.data.use_augm
        if self.random_flips:
            self.flip = torchvision.transforms.RandomRotation((-10, 10))
        image_size = cfg.data.image_size
        self.resize = transforms.Resize((image_size, image_size))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]
        # img = self.resize(img)

        if self.random_flips:
            img = self.flip(img)

        return img, target
