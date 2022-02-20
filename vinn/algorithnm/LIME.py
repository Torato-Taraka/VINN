from lime import lime_image
import torch
import numpy as np
from torch import nn
from torchvision import transforms

from .Algorithm import Algorithm
from .utils import gen_mask_image


class LIME(Algorithm):
    """
        Local Interpretable Model-agnostic Explanations
        Taken from the paper https://arxiv.org/abs/1602.04938v3
    """

    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 label: int = None,
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

        if label:
            self.label = label
        else:
            img = self.transform(self.image).unsqueeze(0).to(self.device)
            output = self.model(img)
            label = output.argmax(1).cpu().detach().numpy()[0]
            self.label = label

    def forward(self):
        explainer = lime_image.LimeImageExplainer()
        result = explainer.explain_instance(self.image, self.predict_fn, labels=self.label, num_samples=1000)
        _, mask = result.get_image_and_mask(result.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        self.mask = mask

        self.masked_image = gen_mask_image(self.mask, self.image)

        return self.mask

    def predict_fn(self, x):
        x = torch.tensor(np.array([self.transform(i).numpy() for i in x])).to(self.device)
        pred = self.model(x)
        return pred.cpu().detach().numpy()
