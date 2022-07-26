import torch
import torch.nn as nn

from multires_hash_encoding.modules import (
    DenseEncodingLevel,
    HashEncodingLevel,
    MultiresEncoding,
    MLP,
    ViewEncoding,
    MultiresEncodingConfig,
    #MultiResHashNeRF,
)


def test_DenseEncodingLevel():
    l = DenseEncodingLevel((2, 8, 10, 12))
    nn.init.ones_(l.interp.input)
    x = torch.zeros(4, 3)
    out = l(x)
    assert out.shape == (4, 2)
    assert torch.allclose(out, torch.ones_like(out))


def test_HashEncodingLevel():
    l = HashEncodingLevel((2, 8, 10, 12), table_size=1)
    nn.init.ones_(l.interp.input.data)
    x = torch.zeros(4, 3)
    out = l(x)
    assert out.shape == (4, 2)
    assert torch.allclose(out, torch.ones_like(out))


def test_MultiresEncodingLayer():
    config = MultiresEncodingConfig()
    model = MultiresEncoding(**vars(config))
    nn.init.ones_(model.levels[0].interp.input)
    for l in model.levels[1:]:
        nn.init.ones_(l.interp.input.data)
    x = torch.zeros(4, 3)
    out = model(x, normalized=True)
    assert out.shape == (4, config.features * config.nlevels)
    assert torch.allclose(out, torch.ones_like(out))


def test_MLP():
    model = MLP(3, 16, 4)
    x = torch.rand(10, 3)
    out = model(x)
    assert out.shape == (10, 4)


def test_ViewEncoding():
    model = ViewEncoding(2)
    x = torch.rand(10, 3)
    out = model(x)
    assert out.shape == (10, 4)
