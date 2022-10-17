# Code modified based on  https://github.com/AI-secure/semantic-randomized-smoothing
from typing import *
import torch
import os, pickle
import cupy as np
from torch.utils.data import Dataset




# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist", "metaroom"]


def get_dataset(dataset: str, split: str, typ: str, vanilla=False, uniform = False, beta=False) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "fashionmnist":
        return _fashion_mnist(split)
    elif dataset == "metaroom":
        return _metaroom(split, typ, vanilla, uniform, beta)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10
    elif dataset == "metaroom":
        return 20


def get_dataset_shape(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return (3, 224, 224)
    elif dataset == "cifar10":
        return (3, 32, 32)
    elif dataset == "mnist":
        return (1, 28, 28)
    elif dataset == "metaroom":
        return (3, 90, 160)


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    elif dataset == "metaroom":
        return NormalizeLayer(_DEFAULT_MEAN, _DEFAULT_STDDEV)
    else:
        raise Exception("Unknown dataset")


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]

_DEFAULT_MEAN = [0.5, 0.5, 0.5]
_DEFAULT_STDDEV = [0.5, 0.5, 0.5]

def _load_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def _set_env(split: str, typ: str, vanilla=False):
    if split == "certify" or split == "proj_test":
        os.environ[METAROOM_LOC_ENV] = './metaroom/metaroom_' + typ[-2:] + '/'
        print('./metaroom/metaroom_' + typ[-2:] + '/')
    else:
        if vanilla:
            os.environ[
                METAROOM_LOC_ENV] = './metaroom/metaroom_' + \
                                    typ[0] + typ[-1] + "_vanilla" + '/'
        else:
            os.environ[METAROOM_LOC_ENV] = './metaroom/metaroom_'+ typ[0] + typ[-1] +'/'

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)

def _metaroom(split: str, typ: str, vanilla=False, uniform = False, beta=False) -> Dataset:

    _set_env(split, typ, vanilla)
    transform = None
    subdir = ""
    if not METAROOM_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[METAROOM_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "val":
        subdir = os.path.join(dir, "test")# val
        transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif split == "test":
        if uniform:
            subdir = os.path.join(dir, "test_uniform")
        elif beta:
            subdir = os.path.join(dir, "test_beta")
        else:
            subdir = os.path.join(dir, "test")
        transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif split == "proj_test":
        subdir = os.path.join(dir, "proj_test")
        transform = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif split == "certify":
        subdir = os.path.join(dir, "certify")
        return datasets.DatasetFolder(subdir, _load_pickle, extensions="pkl")
    return datasets.ImageFolder(subdir, transform)


def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _fashion_mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.FashionMNIST("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        return (input - means) / sds
