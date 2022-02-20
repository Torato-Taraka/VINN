from typing import Tuple
from ..logger import log


class Models:
    _models = {}

    def __init__(self, name: str, alias: Tuple[str, ...] = ()):
        self.name = name
        self.alias = alias

    def __call__(self, function):
        if self._models.get(self.name):
            log.warn(f"模型{self.name}已存在")
        else:
            self._models[self.name] = function
        for alias in self.alias:
            if self._models.get(alias):
                log.warn(f"模型别名{alias}已存在")
            else:
                self._models[alias] = function

    @staticmethod
    def get_model(name: str):
        if Models._models.get(name):
            return Models._models.get(name)
        else:
            log.critical(f"模型{name}不存在")