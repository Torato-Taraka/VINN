from .Models import Models
from . import ResNet


def get_model(name: str):
    return Models.get_model(name)
