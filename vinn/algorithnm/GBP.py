from torch import nn, clamp
import numpy as np
from torchvision import transforms
import cv2

from .Algorithm import Algorithm


class GBP(Algorithm):
    """
        Guided Backpropagation
        Taken from the paper https://arxiv.org/abs/1412.6806
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

        self.image_reconstruction = []
        self.activation_maps = []

        self.register_hooks()

    def forward(self):
        img = self.transform(self.image).unsqueeze(0).to(self.device)
        img.requires_grad = True

        output = self.model(img)
        self.model.zero_grad()
        index = output.argmax(1).cpu().detach().numpy()[0]
        output[0][index].backward()

        result = img.grad[0].detach().cpu().numpy().transpose(1, 2, 0)
        result = (result - result.mean()) / result.std()
        result *= 0.1
        result += 0.5
        H, W, C = result.shape
        result = cv2.resize(result, (W, H))
        result = result.clip(0, 1)
        self.mask = result

        return self.mask

    def register_hooks(self):
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activation_maps.append(output)

    def backward_hook(self, module, grad_in, grad_out):
        return (clamp(grad_in[0], min=0.0),)
