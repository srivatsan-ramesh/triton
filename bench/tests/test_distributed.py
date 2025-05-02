import torch
from triton_bench.distributed import all_gather

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
