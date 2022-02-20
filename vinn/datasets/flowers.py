from .DataSets import DataSets
from .. import log

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch


class Flowers(Dataset):
    def __init__(self, root, train, transform):
        img_dir = os.path.join(root, '102flowers/jpg')
        img_path = []
        if not os.path.exists(img_dir):
            raise ValueError(f"{img_dir}\n未找到 102flowers/jpg 文件夹，检查是否路径错误")
        else:
            for img in os.listdir(img_dir):
                img_path.append(os.path.join(img_dir, img))
        img_path.sort()
        img_path = np.array(img_path)
        assert(len(img_path) == 8189)

        labels = loadmat(os.path.join(root, 'imagelabels.mat'))
        labels = labels['labels'][0] - 1

        setid = loadmat(os.path.join(root, 'setid.mat'))
        testid = np.append(setid['trnid'][0] - 1, setid['valid'][0] - 1)
        trainid = setid['tstid'][0] - 1

        if train:
            img_path = img_path[trainid]
            self.labels = labels[trainid]
        else:
            img_path = img_path[testid]
            self.labels = labels[testid]

        images = []
        for img in img_path:
            image = Image.open(img)
            image = image.resize((256, 256), Image.ANTIALIAS)
            image = np.array(image)
            images.append(image)
        self.images = np.array(images)

        self.transform = transform

    def __getitem__(self, index):
        target = self.transform(self.images[index])
        label = torch.tensor(self.labels[index]).long()
        return target, label

    def __len__(self):
        return len(self.labels)


@DataSets("102flowers", ("flowers",))
def flowers(train=True, transform=None, download=False, root='./data'):
    dataset = Flowers(root=root, train=train, transform=transform)

    if train:
        log.info(f"load train dataset")
    else:
        log.info(f"load test dataset")
    log.info(f"dataset size: {len(dataset.images)}")
    log.info(f"image size: {dataset.images[0].shape}")
    log.info(f"n_channels: 3")
    log.info(f"classes: 102")
    return dataset, 102, 3
