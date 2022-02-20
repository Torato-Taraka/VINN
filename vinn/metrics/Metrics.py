from typing import Tuple
from torch import Tensor


class Metrics:
    def __init__(self, names: Tuple[str, ...] = ()):
        self._names = names

    def __call__(self, output: Tensor, target: Tensor):
        self.forward(output, target)

    def forward(self, output: Tensor, target: Tensor):
        pass

    def reset(self):
        pass

    def get_names(self):
        return self._names
    