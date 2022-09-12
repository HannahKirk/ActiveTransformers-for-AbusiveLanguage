"""TransformersDatasetWeak

Modified small_text dataset class to allow for true and weak labels.
See https://github.com/webis-de/small-text/blob/main/small_text/data/datasets.py

"""
import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from small_text.integrations.pytorch.datasets import PytorchDataset
    from small_text.integrations.pytorch.datasets import PytorchDatasetView
except ModuleNotFoundError:
    raise PytorchNotFoundError('Could not import torchtext')


class TransformersDatasetWeak(PytorchDataset):
    """
    Dataset class for classifiers from Transformers Integration.
    """
    INDEX_TEXT = 0
    INDEX_MASK = 1
    INDEX_LABEL = 2
    INDEX_WEAKLABEL = 3

    NO_LABEL = -1

    def __init__(self, data, target_labels=None, device=None):
        """
        Parameters
        ----------
        data : list of 4-tuples (text data [Tensor], mask [Tensor], label [int], weak label [int])
            Data set.
        """
        self._data = data

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels(self.y)

        if device is None:
            self.device = None if len(data) == 0 else next(iter(data))[self.INDEX_TEXT].device
        else:
            self.device = device

        super().__init__(device=device)

    def _infer_target_labels(self, y):
        self.target_labels = np.unique([d[self.INDEX_LABEL] for d in self._data])

    @property
    def x(self):
        return [d[self.INDEX_TEXT] for d in self._data]

    @property
    def yweak(self):
        return np.array([d[self.INDEX_WEAKLABEL] for d in self._data], dtype = int)

    @x.setter
    def x(self, x):
        for i, _x in enumerate(x):
            self._data[i] = (_x, self._data[i][self.INDEX_MASK], self._data[i][self.INDEX_LABEL])

    @property
    def y(self):
        # TODO: document that None -> -1
        return np.array([d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None
                         else self.NO_LABEL
                         for d in self._data], dtype=int)

    @y.setter
    def y(self, y):
        # TODO: check same length
        for i, _y in enumerate(y):
            self._data[i] = (self._data[i][self.INDEX_TEXT], self._data[i][self.INDEX_MASK], _y)
        self._infer_target_labels(self._data)

    @property
    def data(self):
        return self._data

    @property
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        # TODO: how to handle existing labels that outside this set
        self._target_labels = target_labels

    def to(self, other, non_blocking=False, copy=False):

        data = np.array([(d[self.INDEX_TEXT].to(other, non_blocking=non_blocking, copy=copy),
                         d[self.INDEX_MASK].to(other, non_blocking=non_blocking, copy=copy),
                         d[self.INDEX_LABEL]) for d in self._data])

        if copy is True:
            target_labels = None if self.track_target_labels else self._target_labels
            return TransformersDataset(data, target_labels=target_labels, device=self.device)
        else:
            self._data = data
            return self

    def to(self, device=None, dtype=None, non_blocking=False, copy=False,
           memory_format=torch.preserve_format):

        data = [(d[self.INDEX_TEXT].to(device=device, dtype=dtype, non_blocking=non_blocking,
                                       copy=copy, memory_format=memory_format),
                d[self.INDEX_MASK].to(device=device, dtype=dtype, non_blocking=non_blocking,
                                      copy=copy, memory_format=memory_format),
                d[self.INDEX_LABEL]) for d in self._data]

        if copy is True:
            target_labels = None if self.track_target_labels else self._target_labels
            return TransformersDataset(data, target_labels=target_labels, device=device)
        else:
            self._data = data
            return self

    def __getitem__(self, item):
        return PytorchDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)