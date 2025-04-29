import os
import torch
import torch.distributed as dist
from typing import Tuple


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> Tuple[int, int]:
    if _is_distributed_launch():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def torch_all_gather(x, dim):
    if _is_distributed_launch():
        bufs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, x)
        return torch.cat(bufs, dim=dim)
    else:
        return x
