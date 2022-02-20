import numpy as np
from torch import nn
from torchvision import transforms
import cv2

from .Algorithm import Algorithm
from .utils import gen_mask_image


class GradCAM(Algorithm):
    """
        Grad CAM
        Taken from the paper https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 last_conv: nn.Conv2d,
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

        self.last_conv = last_conv
        self.grad_block = []
        self.feature_map_block = []

        self.register_hooks()

    def forward(self):
        img = self.transform(self.image).unsqueeze(0).to(self.device)

        output = self.model(img)
        self.model.zero_grad()
        index = output.argmax(1).cpu().detach().numpy()[0]
        output[0, index].backward()

        grads = self.grad_block[0].cpu().data.numpy()[0]
        feature_maps = self.feature_map_block[0].cpu().data.numpy()[0]

        img = img[0].cpu().numpy().transpose(1, 2, 0)
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)

        H, W, C = img.shape
        grads = grads.reshape([grads.shape[0], -1])
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))
        self.mask = cam

        self.masked_image = gen_mask_image(self.mask, self.image)

        return self.mask

    def register_hooks(self):
        self.last_conv.register_forward_hook(self.forward_hook)
        self.last_conv.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_in[0].detach())

    def forward_hook(self, module, input, output):
        self.feature_map_block.append(output)
