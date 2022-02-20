from .DataSets import DataSets
from . import CIFAR
from . import MNIST
from . import SVHN
from . import flowers


def get_dataset(name: str):
    return DataSets.get_dataset(name)


def list_datasets():
    return DataSets.list_datasets()