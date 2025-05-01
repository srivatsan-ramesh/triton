import os
import torch
import torch.distributed as dist
import triton_bench.routing
from triton_bench.routing import RoutingData, GatherIndx, ScatterIndx
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


def reduce_scatter(x: torch.Tensor, local_expt_masks: torch.Tensor = None, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        if local_expt_masks is not None:
            mask = local_expt_masks
            # build the padded shape
            shape = list(x.shape)
            shape[dim] = mask.shape[dim]
            # create a zero tensor, scatter x into it where mask is True, then split
            x_list = x.new_zeros(shape).masked_scatter_(mask, x).chunk(world_size, dim=dim)
        else:
            x_list = x.chunk(world_size, dim=dim)
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


def routing(logits, n_expts_act, expt_indx=None, EP=1):
    if _is_distributed_launch():
        assert EP == 1, "Distributed routing does not support ep"

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
        # Distributed-DP
        expt_scal = all_gather(expt_scal, dim=0)
        expt_indx = all_gather(expt_indx, dim=0)
        # flatten topk data
        expt_scal = expt_scal.reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)
        # Distributed-EP
        sorted_expt_indx = torch.argsort(expt_indx, stable=True)
        # Get my own rank in distributed world
        rank = dist.get_rank()
        expt_min = n_expts_tot // EP * rank
        expt_max = n_expts_tot // EP * (rank + 1)
        local_expt_mask = (sorted_expt_indx < expt_max) & (sorted_expt_indx >= expt_min)
        gate_scal = expt_scal[local_expt_mask]
        expt_indx = expt_indx[local_expt_mask]
        # sort by expert_id so experts are contiguous for the matmul
        # For example:
        # expt_indx: [expt0 => row4, row1, row0, ..., expt1 => row2, row3, ..., ...]
        # topk_indx: [2 (row0), 1 (row1), 3 (row2), 4 (row3), 5 (row4), ...]
        topk_indx = torch.argsort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx)
        hist = torch.histc(expt_indx, bins=n_expts_tot // EP,
                           max=n_expts_tot // EP - 1)  # histogram of tokens over experts
        # pack the matmul data structure
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        return RoutingData(gate_scal, hist, n_expts_tot // EP, n_expts_act), gather_indx, scatter_indx, local_expt_mask
    else:
        return triton_bench.routing.routing(logits, n_expts_act, expt_indx, EP), None
