import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import trange

from .Algorithm import Algorithm
from ..logger import log


class SmoothGrad(Algorithm):
    """
        Saliency Map
        Taken from the paper https://arxiv.org/abs/1312.6034
    """

    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 stdev_spread: int = 0.15,
                 n_samples: int = 25,
                 magnitude: bool = False,
                 transform: transforms = None,
                 gpu: bool = True):
        """
        :param model: pytorch model
        :param image: raw image data, only support one image once,
                      if it's 3 channel, you have better to change it to RGB mode
        :param stdev_spread: to control the std of Gaussian distribution, generating noise data
        :param n_samples: the times of sampling noise, to reduce error
        :param magnitude: whether to scale the grad by pow or not
        :param transform: the same as your model's transform
        :param gpu: whether use cuda or not
        """
        super().__init__(model, image, transform, gpu)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude

    def forward(self):
        img = self.transform(self.image).unsqueeze(0).to(self.device)
        stdev = self.stdev_spread * (img.max() - img.min())
        total_gradients = np.zeros_like(img.cpu().numpy())
        for i in trange(self.n_samples):
            log.info(f"epoch [{i + 1}|{self.n_samples}]...")
            noise = torch.normal(mean=0, std=stdev, size=img.shape, dtype=torch.float32).to(self.device)
            img_plus_noise = img + noise
            img_plus_noise.requires_grad = True

            output = self.model(img_plus_noise)
            index = output.argmax(1).cpu().detach().numpy()[0]
            output[0, index].backward()

            grad = img_plus_noise.grad.cpu().detach().numpy()

            if self.magnitutde:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad

        log.info("finish perturbation.")
        self.mask = total_gradients[0, :, :, :] / self.n_samples
        self.mask = (self.mask - self.mask.min()) / (self.mask.max() - self.mask.min())
        self.mask = self.mask.transpose(1, 2, 0)

        return self.mask
