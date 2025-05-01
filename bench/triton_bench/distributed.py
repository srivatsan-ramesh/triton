import os
import torch
import torch.distributed as dist
from triton_bench.routing import RoutingData, GatherIndx, ScatterIndx, routing_torch
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


def all_gather(x: torch.Tensor, dim=0):
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


def reduce_scatter(x: torch.Tensor, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = list(x.chunk(world_size, dim=dim))
        # build output shape
        shape = list(x.shape)
        shape[dim] //= world_size
        # reduce scatter into the single tensor
        # check if dtype is fp8, then convert it to float16 before reducing
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x_list = [x.to(torch.float16) for x in x_list]
            out = x.new_empty(shape, dtype=torch.float16)
        else:
            out = x.new_empty(shape, dtype=x.dtype)
        dist.reduce_scatter(out, x_list)
        return out
    else:
        return x


def routing(logits, n_expts_act, expt_indx=None):
    if _is_distributed_launch():
        assert expt_indx is None

        def topk(vals, k, expt_indx):
            # topk of experts
            if expt_indx is None:
                tk_idx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
            else:
                tk_idx = expt_indx
            tk_val = torch.take_along_dim(vals, tk_idx, dim=1)
            return tk_val, tk_idx

        _, n_expts_tot = logits.shape
        expt_scal, expt_indx = topk(logits, n_expts_act, expt_indx)
        expt_scal = torch.softmax(expt_scal, dim=-1)
        # Sort each token's selections by expert
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        expt_scal = expt_scal.reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)
        # Distributed
        expt_scal = all_gather(expt_scal, dim=0)
        expt_indx = all_gather(expt_indx, dim=0)
        # flatten topk data
        # sort by expert_id so experts are contiguous for the matmul
        topk_indx = torch.argsort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx)
        gate_scal = expt_scal[topk_indx]
        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)  # histogram of tokens over experts
        # pack the matmul data structure
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act), gather_indx, scatter_indx
    else:
        return routing_torch(logits, n_expts_act, expt_indx)
