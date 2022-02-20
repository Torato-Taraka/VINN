from typing import Tuple
import numpy as np
from torch import Tensor
from .. import log
from .Metrics import Metrics


class ConfuseMatrix(Metrics):
    def __init__(self, classes: int, names: Tuple[str, ...] = (), *args, **kwargs):
        super().__init__(names)
        self.classes = classes
        self.confuse_matrix = np.zeros((classes, classes), dtype=np.int)

    def forward(self, output: Tensor, target: Tensor):
        predicts = output.argmax(1).cpu().numpy()
        target = target.cpu().numpy()
        for i, j in zip(predicts, target):
            self.confuse_matrix[i, j] += 1

    def reset(self):
        self.confuse_matrix = np.zeros((self.classes, self.classes), dtype=np.int)

    @property
    def accuracy(self):
        correct = 0
        for n in range(self.classes):
            correct += self.confuse_matrix[n, n]
        Accuracy = correct / self.confuse_matrix.sum()
        return Accuracy

    @property
    def recall(self):
        if self.classes != 2:
            log.critical("不是二分类问题，请调用macro_recall")
            raise Exception()
        AT = self.confuse_matrix.sum(axis=0)[1]
        TP = self.confuse_matrix[1, 1]
        Recall = TP / AT
        return Recall

    @property
    def precision(self):
        if self.classes != 2:
            log.critical("不是二分类问题，请调用macro_precision")
            raise Exception()
        PT = self.confuse_matrix.sum(axis=1)[1]
        TP = self.confuse_matrix[1, 1]
        Precision = TP / PT
        return Precision

    @property
    def f1_score(self):
        Recall = self.recall
        Precision = self.precision
        F1_score = 2 * Precision * Recall / (Precision + Recall)
        return F1_score

    @property
    def micro_recall(self):
        AT = self.confuse_matrix.sum(axis=0)
        TP = np.array([self.confuse_matrix[i, i] for i in range(self.classes)])
        MicroRecall = AT.sum() / TP.sum()
        return MicroRecall

    @property
    def macro_recall(self):
        AT = self.confuse_matrix.sum(axis=0)
        TP = np.array([self.confuse_matrix[i, i] for i in range(self.classes)])
        temp = TP / AT
        log.debug(temp)
        MacroRecall = temp.mean()
        return MacroRecall

    @property
    def micro_precision(self):
        PT = self.confuse_matrix.sum(axis=1)
        TP = np.array([self.confuse_matrix[i, i] for i in range(self.classes)])
        MicroPrecision = TP.sum() / PT.sum()
        return MicroPrecision

    @property
    def macro_precision(self):
        PT = self.confuse_matrix.sum(axis=1)
        TP = np.array([self.confuse_matrix[i, i] for i in range(self.classes)])
        temp = TP / PT
        log.debug(temp)
        MacroPrecision = temp.mean()
        return MacroPrecision

