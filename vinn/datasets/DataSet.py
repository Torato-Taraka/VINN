class DataSet:
    def __init__(self, dataset, classes, in_channels, *args, **kwargs):
        self.dataset = dataset
        self._classes = classes
        self._in_channels = in_channels

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def __len__(self):
        return self.dataset.__len__()

    @property
    def classes(self):
        return self._classes

    @property
    def in_channels(self):
        return self._in_channels
