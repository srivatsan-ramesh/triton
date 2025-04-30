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


def torch_all_gather(x, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        # build output shape
        shape = list(x.shape)
        shape[dim] *= world_size
        out = x.new_empty(shape)
        # gather into the single tensor
        dist.all_gather_into_tensor(out, x)
        return out
    else:
        return x


def torch_reduce_scatter(x, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = list(x.chunk(world_size, dim=dim))
        # build output shape
        shape = list(x.shape)
        shape[dim] //= world_size
        out = x.new_empty(shape)
        # reduce scatter into the single tensor
        dist.reduce_scatter(out, x_list)
        return out
    else:
        return x