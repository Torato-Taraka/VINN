import torch
from torch import nn
from torchvision import transforms
import numpy as np

from .Algorithm import Algorithm


class SaliencyMap(Algorithm):
    """
        Saliency Map
        Taken from the paper https://arxiv.org/abs/1312.6034
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
        super().__init__(model, image, transform, gpu)

    def forward(self):
        img = self.transform(self.image).unsqueeze(0).to(self.device)
        img.requires_grad = True

        output = self.model(img)
        self.model.zero_grad()
        index = output.argmax(1).cpu().detach().numpy()[0]
        output[0][index].backward()

        self.mask, _ = torch.max(abs(img.grad[0].detach()), dim=0)
        self.mask = self.mask.cpu().numpy()

        return self.mask
