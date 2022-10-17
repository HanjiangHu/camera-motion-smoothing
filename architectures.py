# Code modified based on  https://github.com/AI-secure/semantic-randomized-smoothing
import torch
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
import torchvision.models as tv_models
import torch.backends.cudnn as cudnn
# from archs.cifar_resnet import resnet as resnet_cifar
# from archs.fashionmnist_conv import Conv2FC2full, Conv2FC2simple
# from archs.mnist_conv import Conv4FC3
from datasets import get_normalize_layer
from torch.nn.functional import interpolate
import torch.nn as nn


# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", 'resnet18',  'resnet101', 'alexnet']

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and (dataset == "imagenet" or dataset == "metaroom"):
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet18":
        model = torch.nn.DataParallel(resnet18(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet101":
        model = torch.nn.DataParallel(resnet101(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "alexnet":
        model = torch.nn.DataParallel(tv_models.alexnet(pretrained=False)).cuda()
        cudnn.benchmark = True
    else:
        print("not supported")
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
