import torch
from torch import nn
from torchvision import transforms
import numpy as np


class Algorithm:
    """
        general algorithm framework
    """
    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 transform: transforms = None,
                 gpu: bool = True):
        """
        :param model: pytorch model
        :param image: raw image data, only support one image once,
                      if it's 3 channel, you have better to change it to RGB mode
        :param transform: the same as your model's transform
        :param gpu: whether use cuda or not
        """
        self.model = model
        self.model.eval()
        self.image = image
        self.transform = transform
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        self.mask = None
        self.masked_image = None

    def forward(self):
        pass
