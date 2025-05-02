import torch
from triton_bench.distributed import all_gather, reduce_scatter, routing

import torch.distributed as dist
import triton_bench


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


def test_all_gather_distributed_dim1(monkeypatch):
    # WORLD_SIZE=3, gather along dim=1 (columns)
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 3)

    def dummy_all_gather_into_tensor_dim1(out, x):
        # simulate gathering 3 replicas along dim=1
        out.copy_(torch.cat([x, x, x], dim=1))

    monkeypatch.setattr(dist, "all_gather_into_tensor", dummy_all_gather_into_tensor_dim1)

    x = torch.randn(2, 2)
    result = all_gather(x, dim=1)
    expected = torch.cat([x, x, x], dim=1)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_reduce_scatter_distributed_dtype_cast(monkeypatch):
    # WORLD_SIZE=2, x dtype int32 => should be cast to float16
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randint(0, 10, (4, 4), dtype=torch.int32)
    expected = x.chunk(2, dim=0)[0].to(torch.float16)
    result = reduce_scatter(x, token_mask=None, dim=0)
    assert result.shape == expected.shape
    assert result.dtype == torch.float16
    assert torch.allclose(result, expected)


def test_reduce_scatter_distributed_with_token_mask_dim1(monkeypatch):
    # WORLD_SIZE=2, token_mask on dim=1 (columns)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(3, 2)
    # mask expands second dim to 4 columns, 6 True positions matching 3*2 elements
    token_mask = torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]],
                              dtype=torch.bool)
    shape = [3, 4]
    x_new = x.new_zeros(shape)
    x_new[token_mask] = x
    expected = x_new.chunk(2, dim=1)[0]
    result = reduce_scatter(x, token_mask=token_mask, dim=1)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_routing_non_distributed(monkeypatch):
    # WORLD_SIZE=1 => fallback to triton_bench.routing.routing
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(triton_bench.routing, "routing", lambda logits, n_act, expt_indx=None, EP=1: "dummy_routing")
    result, extra = routing(torch.randn(5, 4), 2)
    assert result == "dummy_routing"
    assert extra is None
