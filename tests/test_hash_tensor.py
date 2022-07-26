import pytest
import torch
from multires_hash_encoding.hash_tensor import (
    HashTensor,
)


def test_HashArray():
    data = torch.ones((1,)).reshape(1, 1)

    # 1D
    ha = HashTensor(data, (1, 2))
    assert ha.shape == (1, 2)
    assert ha.ndim == 2
    assert ha[:, 0] == 1

    # 2D
    ha = HashTensor(data, (1, 2, 2))
    assert ha[:, 0, 0] == 1

    # 3D
    ha = HashTensor(data, (1, 2, 2, 2))
    assert ha[:, 0, 0, 0] == 1
