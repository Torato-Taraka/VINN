import numpy as np
import cv2


def gen_mask_image(mask: np.ndarray, image: np.ndarray,
                   mask_trans=0.5, image_trans=0.5, colormap=cv2.COLORMAP_JET):
    if len(image.shape) != 3:
        raise RuntimeError("image维数错误")
    if len(mask.shape) == 2:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = heatmap[:, :, ::-1]
        masked_image = cv2.addWeighted(image, image_trans, heatmap, mask_trans, 0)
        return masked_image
    elif len(mask.shape) == 3:
        return cv2.addWeighted(image, image_trans, mask, mask_trans, 0)
    else:
        raise RuntimeError("mask维数错误")