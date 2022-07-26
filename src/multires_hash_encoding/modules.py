from typing import Iterable
from dataclasses import dataclass
import torch
import torch.nn as nn

from multires_hash_encoding.hash_tensor import HashTensor, _get_level_res_nd
from multires_hash_encoding.interpolate import Interpolate

__all__ = [
    "DenseEncodingLevel",
    "HashEncodingLevel",
    "MultiresEncoding",
    "MultiresEncodingConfig",
    "MLP",
    "ViewEncoding",
    "MultiresHashNeRF",
    ]

Shape = Iterable[int]


class DenseEncodingLevel(nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.shape = shape
        grid = nn.Parameter(torch.empty(shape, **factory_kwargs))
        self.interp = Interpolate(grid, d=len(shape) - 1, mode='nearest')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.interp.input, -1e-4, 1e-4)

    def forward(self, coords, normalized=True):
        return self.interp(coords, normalized).permute(1, 0)


class HashEncodingLevel(nn.Module):
    def __init__(self, shape, table_size, device=None, dtype=None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.shape = shape
        grid = nn.Parameter(torch.empty(
            (shape[0], table_size), **factory_kwargs))
        hash_tensor = HashTensor(grid, shape)
        assert hash_tensor.shape == shape
        self.interp = Interpolate(hash_tensor, d=len(shape) - 1, mode=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.interp.input.data, -1e-4, 1e-4)

    def forward(self, coords, normalized=True):
        return self.interp(coords, normalized).permute(1, 0)


class MultiresEncoding(nn.Module):
    def __init__(self,
                 nlevels: int = 16,
                 features: int = 2,
                 table_size: int = 2**18,
                 minres: Shape = (16, 16, 16),
                 maxres: Shape = (512, 512, 512),
                 device=None,
                 dtype=None,):
        super().__init__()
        factory_kwargs = dict(device=device, dtype=dtype)
        res_levels = _get_level_res_nd(nlevels, minres, maxres)
        level0 = DenseEncodingLevel(
            (features, *res_levels[0]), **factory_kwargs)
        levelN = (HashEncodingLevel((features, *l), table_size, **factory_kwargs)
                  for l in res_levels[1:])
        self.levels = nn.ModuleList([level0, *levelN])
        self._maxres = torch.tensor(maxres, **factory_kwargs)
        self.encoding_size = nlevels * features

    def forward(self, coords, normalized=True):
        if not normalized:
            coords = coords / (self._maxres - 1) * 2 - 1
        # Look up features at each level/resolution
        features = [l(coords, True) for l in self.levels]
        return torch.cat(features, -1)


@dataclass
class MultiresEncodingConfig:
    nlevels: int = 16
    features: int = 2
    table_size: int = 2**22
    minres: Shape = (16, 16, 16)
    maxres: Shape = (1024, 1024, 1024)


class MLP(nn.Sequential):
    def __init__(self, *features, activation=nn.ReLU()):
        l1, *ln = (nn.Linear(*f)
                   for f in zip(features[:-1], features[1:]))
        activations = (activation for _ in range(len(ln)))
        super().__init__(l1, *(m for t in zip(activations, ln) for m in t))


# source: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/a7d64eeb8844b57f5ba90463185a1506e2cbb4b8/hash_encoding.py#L75
class ViewEncoding(nn.Module):
    # Spherical Harmonic Coefficients
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (1.0925484305920792, -1.0925484305920792,
          0.31539156525252005, -1.0925484305920792,
          0.5462742152960396,)
    C3 = (-0.5900435899266435, 2.890611442640554,
          -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435,)
    C4 = (2.5033429417967046, -1.7701307697799304,
          0.9461746957575601, -0.6690465435572892,
          0.10578554691520431, -0.6690465435572892,
          0.47308734787878004, -1.7701307697799304,
          0.6258357354491761,)

    def __init__(self, degree=4):
        assert degree >= 1 and degree <= 5
        super().__init__()
        self.degree = degree
        self.encoding_size = degree ** 2

    def forward(self, input):
        result = torch.empty(
            (*input.shape[:-1], self.encoding_size), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * \
                        z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * \
                            (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * \
                            (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result


class MultiresHashNeRF(nn.Module):

    def __init__(self, mlp_width=64, color_channels=3,
                 view_encoding_degree=4,
                 multires_encoding_config=MultiresEncodingConfig()):
        self.position_encoding = MultiresEncoding(
            **vars(multires_encoding_config))
        self.view_encoding = ViewEncoding(view_encoding_degree)
        self.feature_mlp = MLP(
            self.position_encoding.encoding_size, mlp_width, mlp_width)

        self.rgb_mlp = MLP(
            mlp_width + self.view_encoding.encoding_size, mlp_width, color_channels)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [3, 3], dim=-1)
        h = self.position_encoder(input_pts)
        h = self.feature_mlp(h)
        sigma = h[..., 0]
        h = torch.cat([h, self.view_encoder(input_views)], -1)
        rgb = self.rgb_mlp(h)
        return rgb, sigma
