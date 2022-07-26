from typing import Optional
from functools import partial
import torch
from torch import Tensor, dtype, device, Size
import torch.nn as nn

__all__ = ["nd_linear_interp", "Interpolate"]


def nd_corners(d: int, dtype: Optional[dtype]=None, device: Optional[device]=None) -> Tensor:
    """
    Generates the corner coordinates for an N dimensional hyper cube.
    with d=2 will generate 4 coordinates, for d=3 will generate 8 coordinates, etc ...
    """
    xi = [torch.arange(2, dtype=dtype, device=device) for i in range(d)]
    corners = torch.stack(torch.meshgrid(*xi, indexing='ij'), -1)
    return corners.reshape(1, 2**d, d)


@torch.jit.script
def weights_fn(x: Tensor, i: Tensor) -> Tensor:
    return torch.abs(x - torch.floor(x) + i - 1)


@torch.jit.script
def index_fn(x: Tensor, i: Tensor) -> Tensor:
    return torch.floor(x).to(i.dtype) + i


def nearest_index_fn(i: Tensor, shape: Size):
    """ Replaces out of bounds index with the nearest valid index """
    high = torch.tensor(shape, dtype=i.dtype, device=i.device)
    low = torch.zeros_like(high)
    return i.clamp(low, high - 1)


def nd_linear_interp(
        input: Tensor,
        coords: Tensor,
        mode: Optional[str] = 'nearest',
        corners: Optional[Tensor]=None) -> Tensor:
    assert coords.shape[-1] <= input.ndim

    *S, d = coords.shape
    corners = nd_corners(
        d, torch.int64, coords.device) if corners is None else corners
    coords = coords.reshape(-1, 1, d)   # Combine broadcast dimensions

    weights = weights_fn(coords, corners).prod(-1)      # [N, 2**D]
    index = index_fn(coords, corners)                   # [N, 2**D, D]
    if mode == None:
        pass
    elif mode == 'nearest':
        index = nearest_index_fn(index, input.shape[-d:])
    else:
        raise ValueError("only `nearest` mode or `None` is currently supported")
    values = (weights * input[(..., *index.unbind(-1))]).sum(-1)    # [..., N])
    return values.reshape(*input.shape[:-d], *S)


class Interpolate(nn.Module):
    """ N-d Interpolation class """

    def __init__(self, input: Tensor, d: int, order=1, mode='nearest'):
        """
        Args:
            input: The input array
            d: Dimension of interpolation. d=2 will perform bilinear interpolation
                on the last 2 dimensions of `input` (i.e. image with shape [3, H, W])
            order: Interpolation order. default is 1
            mode: determines how the input array is extended beyond its boundaries.
                Default is 'nearest' 
        """
        super().__init__()
        assert order == 1
        assert mode in (None, 'nearest')

        self.input = input
        self.d = d
        self.order = order
        self.mode = mode

        self.corners = nd_corners(d, torch.int64, input.device)
        self._shape = torch.tensor(
            input.shape[-d:], dtype=input.dtype, device=input.device)

    def _unnormalize(self, coords: Tensor) -> Tensor:
        """ map from [-1, 1] ---> grid coords """
        return (coords + 1) / 2 * (self._shape - 1)

    def forward(self, coords, normalized=False):
        """ If normalized -- assumes coords are in range [-1, 1]"""
        if normalized:
            coords = self._unnormalize(coords)
        return nd_linear_interp(self.input, coords, self.mode, self.corners)