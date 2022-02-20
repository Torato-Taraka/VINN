from torch import nn, clamp

from .GradCAM import GradCAM
from .utils import gen_mask_image


class GuidedGradCAM(GradCAM):
    """
        GradCAM + Guided BP
        a little change
    """
    def __init__(self, model, image, last_conv, transform=None, gpu=True):
        super().__init__(model, image, last_conv, transform, gpu)

    def plot(self):
        self.masked_image = gen_mask_image(self.mask, self.image)
        params = {
            "origin image": self.image,
            "GuidedGradCAM": self.mask,
            "masked image": self.masked_image
        }
        return params

    def register_hooks(self):
        self.last_conv.register_forward_hook(self.forward_hook)
        self.last_conv.register_backward_hook(self.backward_hook)

        for module in self.model.children():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_relu_hook)

    def backward_relu_hook(self, module, grad_in, grad_out):
        return (clamp(grad_in[0], min=0), )


