import torch
from torch import nn
from torchvision import transforms
import numpy as np
import cv2
from tqdm import trange

from .Algorithm import Algorithm
from .utils import gen_mask_image


class Sensitivity(Algorithm):
    """
    a typical kind of sensitivity analysis
    Taken from the paper https://arxiv.org/abs/1704.03296

    The goal is to solve for a mask that explains why did the network output a score for a certain category.
    We create a low resolution (28x28) mask, and use it to perturb the input image to a deep learning network.
    The perturbation combines a blurred version of the image, the regular image, and the up-sampled mask.
    Wherever the mask contains low values, the input image will become more blurry.

    We want to optimize for the next properties:
    (loss )1.When using the mask to blend the input image and it's blurred versions,
             the score of the target category should drop significantly. The evidence of the category should be removed!
    (loss1)2.The mask should be sparse. Ideally the mask should be the minimal possible mask to drop the category score.
             This translates to a L1(1 - mask) term in the cost function.
    (loss2)3.The mask should be smooth. This translates to a total variation regularization in the cost function.
    (loss3)4.The mask shouldn't over-fit the network. Since the network activations might contain a lot of noise,
             it can be easy for the mask to just learn random values that cause the score to drop without being visually coherent.
             In addition to the other terms, this translates to solving for a lower resolution 28x28 mask.
    """

    def __init__(self, model: nn.Module,
                 image: np.ndarray,
                 lr: int = 0.1,
                 max_iters: int = 500,
                 l1_coeff: int = 0.01,
                 l2_coeff: int = 0.2,
                 l2_beta: int = 3,
                 transform: transforms = None,
                 gpu: bool = True):
        """
        :param model: pytorch model
        :param image: raw image data, only support one image once,
                      if it's 3 channel, you have better to change it to RGB mode
        :param lr: learning rate of generator
        :param max_iters: step times
        :param l1_coeff: the coefficient of loss1
        :param l2_coeff: the coefficient of loss2
        :param l2_beta: the coefficient of regularization
        :param transform: the same as your model's transform
        :param gpu: whether use cuda or not
        """
        super().__init__(model, image, transform, gpu)
        self.lr = lr
        self.max_iters = max_iters
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.l2_beta = l2_beta

    def forward(self):
        img = self.transform(self.image).unsqueeze(0).to(self.device)

        blurred_image = np.float32(cv2.medianBlur(self.image, 11)) / 255
        blurred_img = self.transform(blurred_image).unsqueeze(0).to(self.device)

        mask = torch.ones([1, 1, 28, 28], dtype=torch.float32).to(self.device).requires_grad_(True)
        upsample = torch.nn.UpsamplingBilinear2d(size=self.image.shape[:2]).to(self.device)
        optimizer = torch.optim.Adam([mask], lr=self.lr)

        target = self.model(img).argmax(1).cpu().detach().numpy()[0]

        for i in trange(self.max_iters):
            upsampled_mask = upsample(mask)
            upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
            perturbated_input = img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)
            noise = torch.normal(mean=0, std=0.2, size=(3, 256, 256)).to(self.device).requires_grad_(True)
            perturbated_input = perturbated_input + noise

            output = torch.nn.Softmax(dim=1)(self.model(perturbated_input))
            loss1 = self.l1_coeff * torch.mean(torch.abs(1 - mask))
            loss2 = self.l2_coeff * self.l2_norm(mask)
            loss3 = output[0, target]
            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask.data.clamp_(0, 1)

        mask = upsample(mask)
        self.mask = mask.cpu().detach().numpy()[0][0]

        self.masked_image = gen_mask_image(self.mask, self.image)

        return self.mask

    def l2_norm(self, input):
        img = input[0, 0, :]
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(self.l2_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(self.l2_beta))
        return row_grad + col_grad
