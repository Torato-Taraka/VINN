from .DataSets import DataSets
from torchvision.datasets import MNIST
from .. import log


@DataSets("MNIST", ("mnist",))
def mnist(train=True, transform=None, download=False, root='./data'):
    dataset = MNIST(train=train, transform=transform, download=download, root=root)
    if train:
        log.info(f"load train dataset")
    else:
        log.info(f"load test dataset")
    log.info(f"dataset size: {len(dataset.targets)}")
    log.info(f"image size: {dataset.data[0].shape}")
    log.info(f"n_channels: 1")
    log.info(f"{dataset.classes}")
    return dataset, len(dataset.classes), 1