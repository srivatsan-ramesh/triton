import torch
from triton_bench.distributed import all_gather, reduce_scatter

import torch.distributed as dist


def test_all_gather_non_distributed(monkeypatch):
    # Ensure single-process environment
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 5)
    result = all_gather(x, dim=0)
    # Non-distributed should return the same tensor
    assert torch.allclose(result, x)


def test_all_gather_distributed(monkeypatch):
    # Simulate distributed environment with world_size=2
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Pretend that the process group is already initialized.
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    # Define a dummy function to replace all_gather_into_tensor.
    # This function simply concatenates two copies of the tensor along dim=0.
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
    # Ensure single-process environment
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 6)
    result = reduce_scatter(x, token_mask=None, dim=0)
    # For non-distributed launch, reduce_scatter returns the same tensor.
    assert torch.allclose(result, x)


# Dummy reduce_scatter function to simulate distributed reduction.
# This dummy simply copies the first chunk from the list into the output.
def dummy_reduce_scatter(out, x_list):
    out.copy_(x_list[0])


def test_reduce_scatter_distributed_no_token(monkeypatch):
    # Simulate distributed environment with world_size=2
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    # Create a tensor with shape divisible by world_size along dim 0.
    x = torch.randn(4, 6)  # 4 rows, will be split into 2 chunks of 2 rows each.
    # Expected output is the first chunk since our dummy copies x_list[0]
    expected = x.chunk(2, dim=0)[0]

    result = reduce_scatter(x, token_mask=None, dim=0)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_reduce_scatter_distributed_with_token(monkeypatch):
    # Simulate distributed environment with world_size=2
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    # For the token_mask branch:
    # Use a tensor x and a token_mask that is all True.
    x = torch.randn(4, 4)
    token_mask = torch.ones_like(x, dtype=torch.bool)
    # Build x_new as done in reduce_scatter:
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
