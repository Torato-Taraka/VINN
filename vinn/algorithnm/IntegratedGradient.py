import numpy as np
from torch import nn
from torchvision import transforms
from tqdm import trange

from .Algorithm import Algorithm


class IntegratedGradient(Algorithm):
    """
        Integrated Gradient
        Taken from the paper https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 n_samples: int = 10,
                 steps: int = 50,
                 target_lable: int = None,
                 transform: transforms = None,
                 gpu: bool = True):
        """
        :param model: pytorch model
        :param image: raw image data, only support one image once,
                      if it's 3 channel, you have better to change it to RGB mode
        :param n_samples: the times of sampling a baseline, to reduce error
        :param steps: step size in approximate integration of rectangular method
        :param target_lable: the lable corresponding to the image
        :param transform: the same as your model's transform
        :param gpu: whether use cuda or not
        """
        super().__init__(model, image, transform, gpu)

        if target_lable:
            self.target_lable = target_lable
        else:
            img = self.transform(self.image).unsqueeze(0).to(self.device)
            self.target_lable = self.model(img).argmax(1).cpu().detach().numpy()[0]

        self.n_samples = n_samples
        self.steps = steps

    def forward(self):
        all_integrated_grads = []
        for i in trange(self.n_samples):
            baseline = 255.0 * np.random.random(self.image.shape)
            scaled_images = [baseline + (float(i) / self.steps) * (self.image - baseline) for i in
                             range(0, self.steps + 1)]

            gradients = []
            for x in scaled_images:
                x = self.transform(x).float().unsqueeze(0).to(self.device).requires_grad_(True)

                output = self.model(x)
                self.model.zero_grad()
                output[0, self.target_lable].backward()

                gradient = x.grad.detach().cpu().numpy()[0]
                gradients.append(gradient)

            avg_grads = np.average(gradients[:-1], axis=0)
            delta_img = (self.transform(self.image) - self.transform(baseline)).numpy()
            int_grad = avg_grads * delta_img
            int_grad = (int_grad - int_grad.min()) / (int_grad.max() - int_grad.min())
            all_integrated_grads.append(int_grad)

        avg_integrated_grad = np.average(np.array(all_integrated_grads), axis=0)
        self.mask = avg_integrated_grad.transpose(1, 2, 0)

        return self.mask
