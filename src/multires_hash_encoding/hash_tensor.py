from typing import Tuple, Any, Iterable, List
from math import log, exp
from functools import reduce
import operator
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

Shape = Iterable[int]


#PRIMES = torch.tensor([1, 73856093, 19349663, 83492791])

def spatial_hash(coords: List[Tensor]) -> Tensor:
    PRIMES = (1, 2654435761, 805459861, 3674653429)
    #assert len(coords) <= len(PRIMES), "Add more PRIMES!"
    if len(coords) == 1:
        i = (coords[0] ^ PRIMES[1])
    else:
        i = coords[0] ^ PRIMES[0]
        for c, p in zip(coords[1:], PRIMES[1:]):
            i ^= c * p
    return i


class HashTensor(nn.Module):
    """
    This is a sparse array backed by simple hash table. It minimally implements an array
    interface as to be used for (nd) linear interpolation.
    There is no collision resolution or even bounds checking.

    Attributes:
      data: The hash table represented as a 2D array.
        First dim is the feature and second dim is indexed with the hash index
      shape: The shape of the array.

    NVIDIA Implementation of multi-res hash grid:
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L66-L80
    """

    def __init__(self, data, shape):
        """
        Attributes:
        data: The hash table represented as a 2D array.
            First dim is the feature and second dim is indexed with the hash index
        shape: The shape of the array.
        """
        assert data.ndim == 2, "Hash table data should be 2d"
        assert data.shape[0] == shape[0]
        super().__init__()
        self.data = data
        self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def forward(self, index):
        #feature_i, *spatial_i = i if len(i) == self.ndim else (Ellipsis, *i)
        assert len(index) == self.ndim
        feature_i, *spatial_i = index
        i = spatial_hash(spatial_i) % self.data.shape[1]
        return self.data[feature_i, i]

    def __getitem__(self, index):
        return self.forward(index)

    def __array__(self, dtype=None):
        _, *S = self.shape
        index = torch.meshgrid(*(torch.arange(s) for s in S))
        arr = self[(slice(0, None), *index)].detach().cpu().__array__(dtype)
        return arr

    def __repr__(self):
        return "HashTensor(" + str(np.asarray(self)) + ")"


def growth_factor(levels: int, minres: int, maxres: int):
    return exp((log(maxres) - log(minres)) / (levels - 1))


def _get_level_res(levels: int, minres: int, maxres: int):
    b = growth_factor(levels, minres, maxres)
    res = [int(round(minres * (b ** l))) for l in range(0, levels)]
    return res


def _get_level_res_nd(levels: int, minres: Iterable[int], maxres: Iterable[int]):
    it = (_get_level_res(levels, _min, _max)
          for _min, _max in zip(minres, maxres))
    return list(zip(*it))
