# VINN
VINN is Visual Interpretation of Neural Networks

VINN includes a series of interpretability algorithms, like LIME, Grad CAM, Integrated Gradient, saliency maps and so on.

VINN is a graduation project, and will update to new version if possible.

### Target audience

Developers, who are looking to understand what their deep models have learnt, or researchers, who want to test their new algorithms are the potential audiences for VINN.

VINN standardize the algorithm framework, so that you can follow the framework to construct your own algorithm and test it.

### Installation

##### requirements

you can create a new environment by

```bash
 conda create -f requirements
```

or make sure your own environments have

- python >= 3.6

- pytorch >= 1.2

##### installing the latest release

pip:

``` bash
pip install vinn
```

conda:

```bash
conda install vinn
```

or you can download from GitHub and install in the repository

``` bash
https://github.com/Torato-Taraka/VINN.git
pip install vinn-1.0-py3-none-any.whl
```

### Getting start

``` python
import torch
import torchvision.transforms as tfs
from torchvision.models import resnet18
import cv2

model = resnet18(pretrained=True)
transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
last_conv = model.layer4[-1]

image_path = 'test.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
gpu = True
```

Till now, necessary parameters(model, image, transform, gpu) is ready, and you can use VINN to interpret your model

```python
import vinn
gradcam = vinn.GradCAM(model=model,
                      image=image,
                      last_conv=last_conv,
                      transform=transform,
                      gpu=gpu)
gradcam.forward()
```

After forward, the algorithm finish processing, and the result is stored in `gradcam.mask`.

You can visualize the result like the following:

```python
params = {
    "original image", gradcam.image,
    "GradCAM", gradcam.mask,
    "masked image", gradcam.masked_image
}
vinn.subplot([params])
```

Note that you should pass a list of dicts, since that you may need to interpret several images at the same time.

And you can see how the image masked by calling `gradcam.masked_image`.

**But pay attention that NOT all the algorithms generates mask and original image together**, only GradCAM, GuidedGradCAM, LIME and Sensitivity have masked_image.

Of course you can generate by yourself and maybe the later version will achieve the function.

### Customize models, datasets and algorithms

VINN support customizing your model, datasets and even algorithms

#### models

You should organize your model like

```python
from vinn.model import Models

@Models("yourmodel")
def yourmodel():
    class yourmodel(nn.Module):
        def __init__(self):
        	...
    return yourmodel()
```

This is for registering your algorithm in order to management

#### datasets

Similar to customize models

```python
from vinn.datasets import DataSet

@DataSet("yourdataset")
def yourdataset():
    class yourdataset:
        def __init__(self):
            ...
            
        def __getitem__(self, index):
            image = self.transform(self.images[index])
            lable = self.lables[index]
            return image, lable
        
        def __len__(self):
            return len(self.labels)
    return yourdataset()
```

`__getitem__()` and `__len__()` is necessary, you can see the details at [torch.utils.data.Dataset]([torch.utils.data — PyTorch 1.10 documentation](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset))

And you can see all the existing dataset by

```python
from vinn import list_dataset
list_dataset()
```

#### train model

```python
from vinn import trainer
from torchvision import transforms as tfs
transform = tfs.Compose([
    ...
])
trainer(
    model_name='yourmodel',
    dataset_name='yourdataset',
    classes=...,
    transform=transform,
    train_size=...,
    test_size=...,
    lr=...,
    epochs=...,
    gpu=True
)
```

- `model_name`: the string in the @Models() you define
- `dataset_name`: the string in the @Dataset() you define
- `classes`: the classes of your dataset
- `transform`: you define it
- `train_size`: the size of a sample from your train data
- `test_size`: the size of a sample from your test data
- `lr`: learning rate
- `epochs`: training times
- `gpu`: whether use cuda or not

#### algorithms

You should first organize your algorithm like

```python
from vinn.algorithm import Algorithm 
class youralgo(Algorithm):
    def __init__(self, model, image, transform, gpu, ...):
        super().__init__(model, image, transform, gpu)
        ...
        
    def forward(self):
        ...
        self.mask = mask
```

`Algorithm` is the father class of all algorithms

The most important thing is that you must pass the result to `self.mask`, so that you can easily visualize your result.

Or you can return your result, but I don't think it good for your following operation.

### References of algorithms

- `IntegratedGradients`: [Axiomatic Attribution for Deep Networks, Mukund Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)

- `SmoothGrad`: [SmoothGrad: removing noise by adding noise, Daniel Smilkov et al. 2017](https://arxiv.org/abs/1706.03825)
- `SaliencyMap` : [Deep Inside Convolutional Networks: Visualising
  Image Classification Models and Saliency Maps, K. Simonyan, et. al. 2014](https://arxiv.org/pdf/1312.6034.pdf)
- `GradCAM`, `GuidedGradCAM`: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Ramprasaath R. Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391.pdf)
- `LIME`:["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938v3)
- `GBP(Guided Backpropagation)`:[Striving for Simplicity: The All Convolutional Net, Jost Tobias Springenberg et al. 2015](https://arxiv.org/pdf/1412.6806.pdf)
- `Sensitivity`:[Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/abs/1704.03296)

### License

VINN is MIT licensed, as found in LICENSE file.