import torch
from triton_bench.distributed import all_gather, reduce_scatter

import torch.distributed as dist


def test_all_gather_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 5)
    result = all_gather(x, dim=0)
    assert torch.allclose(result, x)


def test_all_gather_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def dummy_all_gather_into_tensor(out, x):
        gathered = torch.cat([x, x], dim=0)
        out.copy_(gathered)

    monkeypatch.setattr(dist, "all_gather_into_tensor", dummy_all_gather_into_tensor)

    x = torch.randn(3, 4)
    result = all_gather(x, dim=0)
    expected = torch.cat([x, x], dim=0)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_reduce_scatter_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 6)
    result = reduce_scatter(x, token_mask=None, dim=0)
    assert torch.allclose(result, x)


def dummy_reduce_scatter(out, x_list):
    out.copy_(x_list[0])


def test_reduce_scatter_distributed_no_token_mask(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(4, 6)
    expected = x.chunk(2, dim=0)[0]

    result = reduce_scatter(x, token_mask=None, dim=0)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_reduce_scatter_distributed_with_token_mask(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(2, 4)
    token_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    shape = list(x.shape)
    # Replace first dimension with token_mask's corresponding dimension.
    shape[0] = token_mask.shape[0]
    x_new = x.new_zeros(shape)
    x_new[token_mask] = x
    # Split along dim=0 (world_size=2) and take the first chunk.
    expected = x_new.chunk(2, dim=0)[0]

    result = reduce_scatter(x, token_mask=token_mask, dim=0)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)
