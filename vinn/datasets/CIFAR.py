from .DataSets import DataSets
from .DataSet import DataSet
from torchvision.datasets import CIFAR10, CIFAR100
from .. import log


@DataSets("CIFAR10", ("cifar10",))
def cifar10(train=True, transform=None, download=False, root='./data'):
    dataset = CIFAR10(train=train, transform=transform, download=download, root=root)
    if train:
        log.info(f"load train dataset")
    else:
        log.info(f"load test dataset")
    log.info(f"dataset size: {len(dataset.targets)}")
    log.info(f"image size: {dataset.data[0].shape}")
    log.info(f"n_channels: 3")
    log.info(f"{dataset.classes}")
    return dataset, len(dataset.classes), 3


@DataSets("CIFAR100", ("cifar100",))
def cifar100(train=True, transform=None, download=False, root='./data'):
    dataset = CIFAR100(train=train, transform=transform, download=download, root=root)
    if train:
        log.info(f"load train dataset")
    else:
        log.info(f"load test dataset")
    log.info(f"dataset size: {len(dataset.targets)}")
    log.info(f"image size: {dataset.data[0].shape}")
    log.info(f"n_channels: 3")
    log.info(f"{dataset.classes}")
    return dataset, len(dataset.classes), 3