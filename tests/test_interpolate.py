import numpy as np
from scipy.ndimage import map_coordinates
import torch
from multires_hash_encoding.interpolate import (
    nd_linear_interp,
    Interpolate,
)


def test_bilinear_interpolate():
    coords = torch.rand(4, 2)
    arr = torch.ones((1, 2, 2))
    out = nd_linear_interp(arr, coords)
    assert torch.allclose(out, torch.ones_like(out))

    arr = torch.tensor([[1., 2.], [1., 2.]]).reshape(1, 2, 2)
    coords = torch.tensor([[.5, .5]])

    assert torch.allclose(nd_linear_interp(arr, coords), torch.tensor([1.5]))

    coords = torch.tensor([[.5, 0]])
    assert torch.allclose(nd_linear_interp(arr, coords), torch.tensor([1.0]))


def to_np_coords(coords):
    return tuple(c.numpy() for c in coords.unbind(-1))


def test_map_coordinates():

    img = torch.rand(1, 10, 10)
    coords = torch.tensor([[5., 5.]])

    assert np.allclose(
        map_coordinates(img[0].numpy(), to_np_coords(
            coords), order=1, mode='nearest'),
        nd_linear_interp(img, coords, mode='nearest').numpy())

    num_coords = 5
    for shape in ((2, 10), (3, 10, 12), (4, 10, 12, 14)):
        for order in [1]:
            for mode in ['nearest']:
                print(f'Shape: {shape}')
                n = len(shape) - 1
                signal = torch.rand(*shape)
                coords = torch.rand(num_coords, n) * (10 - 1)
                result = nd_linear_interp(signal, coords, mode)
                assert result.shape == (shape[0], num_coords)
                target1 = map_coordinates(
                    signal[0, ...].numpy(), to_np_coords(coords), order=order, mode=mode, prefilter=False)
                target2 = map_coordinates(
                    signal[1, ...].numpy(), to_np_coords(coords), order=order, mode=mode, prefilter=False)
                assert np.allclose(result[0, :].numpy(), target1) and np.allclose(
                    result[1, :].numpy(), target2)


def test_Interpolate():

    img = torch.tensor([[1, 2, 3], [1, 2, 3]]).reshape(1, 2, 3)
    coords = torch.tensor([[0, .5]])
    interp = Interpolate(img, d=2, order=1, mode='nearest')
    out = interp(coords, normalized=False)
    assert torch.allclose(out, torch.tensor([1.5]))

    out = interp(coords, normalized=True)
    assert torch.allclose(out, torch.tensor([2.5]))
