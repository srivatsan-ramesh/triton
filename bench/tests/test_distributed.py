import torch
from triton_bench.distributed import all_gather, reduce_scatter, routing

import torch.distributed as dist
import triton_bench
import torch
from triton_bench.distributed import setup, routing
from triton_bench.routing import RoutingData, GatherIndx, ScatterIndx
from pytest import MonkeyPatch


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


def test_reduce_scatter_distributed_with_token_mask_dim1(monkeypatch):
    # WORLD_SIZE=2, token_mask on dim=1 (columns)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(3, 2)
    token_mask = torch.tensor([True, False, False, True], dtype=torch.bool)
    shape = [3, 4]
    x_new = x.new_zeros(shape)
    x_new[:, token_mask] = x
    expected = x_new.chunk(2, dim=1)[0]
    result = reduce_scatter(x, token_mask=token_mask, dim=1)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_routing_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(triton_bench.routing, "routing", lambda logits, n_act, expt_indx=None, EP=1: "dummy_routing")
    result, extra = routing(torch.randn(5, 4), 2)
    assert result == "dummy_routing"
    assert extra is None


def test_routing_distributed_EP(monkeypatch):

    def dummy_all_gather_into_tensor(out, x):
        gathered = torch.cat([x, x], dim=0)
        out.copy_(gathered)

    # Test distributed routing with EP=1 (token_mask should be None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Set environment for local rank and distributed group
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather_into_tensor)

    logits = torch.tensor([[0.1, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.1]])
    n_expts_act = 2
    EP = 2
    expt_indx, topk_indx = torch.tensor([[1, 2], [0, 1], [1, 2], [0, 1]]).reshape(-1).sort(stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    topk_indx[expt_indx > 1] = -1
    gate_indx[gate_indx > 1] = -1
    rdata, gather_indx, scatter_indx, token_mask = routing(logits, n_expts_act, EP=EP)
    assert gather_indx.src_indx == topk_indx.int()
    assert gather_indx.dst_indx == gate_indx.int()
    assert scatter_indx.src_indx == gate_indx.int()
    assert scatter_indx.dst_indx == topk_indx.int()
    # rank0
    assert token_mask == torch.tensor([True, False, True, True], dtype=torch.bool)
