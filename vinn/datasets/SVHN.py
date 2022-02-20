from .DataSets import DataSets
from torchvision.datasets import SVHN
from .. import log


@DataSets("SVHN", ("svhn",))
def mnist(train=True, transform=None, download=False, root='./data'):
    if train:
        split = 'train'
    else:
        split = 'test'
    dataset = SVHN(split=split, transform=transform, download=download, root=root)
    if train:
        log.info(f"load train dataset")
    else:
        log.info(f"load test dataset")
    log.info(f"dataset size: {len(dataset.labels)}")
    log.info(f"image size: {dataset.data[0].shape}")
    log.info(f"n_channels: 3")
    log.info(f"labels: {set(dataset.labels)}")
    return dataset, 10, 3