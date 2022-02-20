import matplotlib.pyplot as plt
from typing import List


def subplot(params: List[dict], cmap=None):
    plt.figure()
    rows = len(params)
    cols = max([len(i) for i in params])
    for r in range(rows):
        index = 0
        for key, value in params[r].items():
            index += 1
            plt.subplot(rows, cols, r*cols+index)
            plt.axis('off')
            if r == 0:
                plt.title(key)
            plt.imshow(value, cmap=cmap)
    plt.show()

