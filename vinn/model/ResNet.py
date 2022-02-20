from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from .Models import Models
import torch.nn as nn


def ResNet(model, in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    resnet = model(pretrained, **kwargs)
    resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                             kernel_size=7, stride=2, padding=3, bias=False)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, classes)
    return resnet


@Models("resnet18")
def ResNet18(in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    return ResNet(resnet18, in_channels, classes, pretrained, **kwargs)


@Models("resnet34")
def ResNet34(in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    return ResNet(resnet34, in_channels, classes, pretrained, **kwargs)


@Models("resnet50")
def ResNet50(in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    return ResNet(resnet50, in_channels, classes, pretrained, **kwargs)


@Models("resnet101")
def ResNet101(in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    return ResNet(resnet101, in_channels, classes, pretrained, **kwargs)


@Models("resnet152")
def ResNet152(in_channels=3, classes: int = 1000, pretrained: bool = False, **kwargs):
    return ResNet(resnet152, in_channels, classes, pretrained, **kwargs)