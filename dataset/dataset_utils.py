import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .sampling import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(dataset, data_root, iid, num_users, data_aug, noniid_beta):
    ds = dataset

    if ds == "cifar10":

        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            data_root, train=True, download=False, transform=transform_train
        )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(
            data_root, train=False, download=False, transform=transform_test
        )

    if ds == "cifar100":
        if data_aug:
            print("data_aug:", data_aug)
            normalize = transforms.Normalize(
                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            )
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),  # transforms.ColorJitter(brightness=0.25, contrast=0.8),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        train_set = torchvision.datasets.CIFAR100(
            data_root, train=True, download=True, transform=transform_train
        )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100(
            data_root, train=False, download=False, transform=transform_test
        )

    if iid:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)
    else:
        dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, dict_users, train_idxs, val_idxs


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label
