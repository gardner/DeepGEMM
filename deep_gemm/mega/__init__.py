import torch
from typing import List, Tuple, Optional
from ..utils.math import align, per_token_cast_to_fp8

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


def _all_gather_first_dim(tensor: torch.Tensor,
                          num_rows: int,
                          group: dist.ProcessGroup) -> Tuple[torch.Tensor, List[int]]:
    world_size = group.size()
    size = torch.tensor(num_rows, dtype=torch.long, device=tensor.device)
    sizes = [torch.empty_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size, group=group)
    sizes_list = [int(s.item()) for s in sizes]
    max_rows = max(sizes_list)

    padded = torch.zeros(
        (max_rows, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if num_rows > 0:
        padded[:num_rows].copy_(tensor[:num_rows].contiguous())

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    return torch.cat([chunk[:n] for chunk, n in zip(gathered, sizes_list)], dim=0), sizes_list


def _untranspose_sf_from_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 32, 4, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def _fp8_fp4_mega_moe_sm120(y: torch.Tensor,
                            l1_weights: Tuple[torch.Tensor, torch.Tensor],
                            l2_weights: Tuple[torch.Tensor, torch.Tensor],
                            sym_buffer: "SymmBuffer",
                            cumulative_local_expert_recv_stats: Optional[torch.Tensor],
                            recipe: Tuple[int, int, int],
                            activation: str,
                            activation_clamp: Optional[float],
                            fast_math: bool):
    _ = fast_math
    if recipe != (1, 1, 32):
        raise NotImplementedError("SM120 MegaMoE currently requires recipe=(1, 1, 32).")
    if activation != "swiglu":
        raise NotImplementedError("SM120 MegaMoE currently supports swiglu only.")

    group = sym_buffer.group
    rank = group.rank()
    world_size = group.size()
    num_tokens = y.size(0)
    num_topk = sym_buffer.num_topk
    num_experts = sym_buffer.num_experts
    num_experts_per_rank = num_experts // world_size
    hidden = sym_buffer.hidden
    intermediate_hidden = sym_buffer.intermediate_hidden

    if hidden % 128 != 0 or intermediate_hidden % 128 != 0:
        raise ValueError("SM120 MegaMoE requires hidden sizes to be multiples of 128.")

    if world_size == 1:
        token_sizes = [num_tokens]
        x_all = sym_buffer.x[:num_tokens].contiguous()
        x_sf_all = sym_buffer.x_sf[:num_tokens].contiguous()
        topk_idx_all = sym_buffer.topk_idx[:num_tokens].contiguous()
        topk_weights_all = sym_buffer.topk_weights[:num_tokens].contiguous()
    else:
        x_all, token_sizes = _all_gather_first_dim(sym_buffer.x, num_tokens, group)
        x_sf_all, _ = _all_gather_first_dim(sym_buffer.x_sf, num_tokens, group)
        topk_idx_all, _ = _all_gather_first_dim(sym_buffer.topk_idx, num_tokens, group)
        topk_weights_all, _ = _all_gather_first_dim(sym_buffer.topk_weights, num_tokens, group)

    rank_offsets = torch.tensor(
        [0, *torch.tensor(token_sizes, device="cpu").cumsum(0).tolist()[:-1]],
        dtype=torch.long,
        device=y.device,
    )
    source_ranks = torch.cat([
        torch.full((n, num_topk), r, dtype=torch.long, device=y.device)
        for r, n in enumerate(token_sizes)
    ], dim=0)
    source_tokens = torch.cat([
        torch.arange(n, dtype=torch.long, device=y.device)[:, None].expand(n, num_topk)
        for n in token_sizes
    ], dim=0)

    flat_experts = topk_idx_all.reshape(-1).long()
    flat_weights = topk_weights_all.reshape(-1).float()
    flat_source_ranks = source_ranks.reshape(-1)
    flat_source_tokens = source_tokens.reshape(-1)
    flat_source_rows = rank_offsets[flat_source_ranks] + flat_source_tokens

    local_experts = flat_experts - rank * num_experts_per_rank
    valid = (local_experts >= 0) & (local_experts < num_experts_per_rank)
    if not bool(valid.any().item()):
        y.zero_()
        return

    valid_local_experts = local_experts[valid]
    counts = torch.bincount(
        valid_local_experts,
        minlength=num_experts_per_rank,
    ).to(torch.int32)
    expected_m = int(counts.max().item())
    gemm_m = max(expected_m, 64)
    if cumulative_local_expert_recv_stats is not None:
        cumulative_local_expert_recv_stats.add_(counts)

    l1_gemm_weights = (l1_weights[0], _untranspose_sf_from_utccp(l1_weights[1]))
    l2_gemm_weights = (l2_weights[0], _untranspose_sf_from_utccp(l2_weights[1]))

    l1_acts = torch.zeros(
        (num_experts_per_rank, gemm_m, hidden),
        dtype=sym_buffer.x.dtype,
        device=y.device,
    )
    aligned_m = align(gemm_m, 4)
    l1_acts_sf = torch.empty_strided(
        (num_experts_per_rank, gemm_m, hidden // 128),
        (aligned_m * (hidden // 128), 1, aligned_m),
        dtype=torch.int32,
        device=y.device,
    ).zero_()
    source_rank_by_row = torch.zeros(
        (num_experts_per_rank, gemm_m),
        dtype=torch.long,
        device=y.device,
    )
    source_token_by_row = torch.zeros_like(source_rank_by_row)
    topk_weight_by_row = torch.zeros(
        (num_experts_per_rank, gemm_m),
        dtype=torch.float32,
        device=y.device,
    )

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    for expert_idx in range(num_experts_per_rank):
        expert_indices = valid_indices[valid_local_experts == expert_idx]
        num_expert_tokens = expert_indices.numel()
        if num_expert_tokens == 0:
            continue

        rows = flat_source_rows.index_select(0, expert_indices)
        n = int(num_expert_tokens)
        l1_acts[expert_idx, :n].copy_(x_all.index_select(0, rows))
        l1_acts_sf[expert_idx, :n].copy_(x_sf_all.index_select(0, rows))
        source_rank_by_row[expert_idx, :n].copy_(
            flat_source_ranks.index_select(0, expert_indices)
        )
        source_token_by_row[expert_idx, :n].copy_(
            flat_source_tokens.index_select(0, expert_indices)
        )
        topk_weight_by_row[expert_idx, :n].copy_(
            flat_weights.index_select(0, expert_indices)
        )

    l1_out = torch.zeros(
        (num_experts_per_rank, gemm_m, intermediate_hidden * 2),
        dtype=torch.bfloat16,
        device=y.device,
    )
    _C.m_grouped_fp8_fp4_gemm_nt_masked(
        (l1_acts, l1_acts_sf),
        l1_gemm_weights,
        l1_out,
        counts,
        gemm_m,
        recipe=recipe,
    )

    l1_view = l1_out.float().view(
        num_experts_per_rank,
        gemm_m,
        intermediate_hidden // 8,
        2,
        8,
    )
    gate = l1_view[:, :, :, 0, :].reshape(num_experts_per_rank, gemm_m, intermediate_hidden)
    up = l1_view[:, :, :, 1, :].reshape(num_experts_per_rank, gemm_m, intermediate_hidden)
    if activation_clamp is not None:
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
    l2_input = torch.nn.functional.silu(gate) * up
    l2_input = l2_input * topk_weight_by_row.unsqueeze(-1)

    l2_acts, l2_acts_sf = per_token_cast_to_fp8(
        l2_input.reshape(-1, intermediate_hidden),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=False,
    )
    l2_acts = l2_acts.view(num_experts_per_rank, gemm_m, intermediate_hidden)
    l2_acts_sf = _C.transform_sf_into_required_layout(
        l2_acts_sf.view(num_experts_per_rank, gemm_m, intermediate_hidden // 32),
        gemm_m,
        intermediate_hidden,
        (1, 32),
        num_experts_per_rank,
    )

    l2_out = torch.zeros(
        (num_experts_per_rank, gemm_m, hidden),
        dtype=torch.bfloat16,
        device=y.device,
    )
    _C.m_grouped_fp8_fp4_gemm_nt_masked(
        (l2_acts, l2_acts_sf),
        l2_gemm_weights,
        l2_out,
        counts,
        gemm_m,
        recipe=recipe,
    )

    max_tokens = max(token_sizes)
    combined = torch.zeros(
        (world_size, max_tokens, hidden),
        dtype=y.dtype,
        device=y.device,
    )
    combined.view(-1, hidden).index_add_(
        0,
        source_rank_by_row.reshape(-1) * max_tokens + source_token_by_row.reshape(-1),
        l2_out.reshape(-1, hidden),
    )
    if world_size > 1:
        dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=group)
    y.copy_(combined[rank, :num_tokens])


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 # MoE arguments
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def _interleave_l1_weights(l1_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    def interleave(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return interleave(l1_weights[0]), interleave(l1_weights[1])


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up, then transpose SF for UTCCP
    l1_interleaved = _interleave_l1_weights(l1_weights)
    l1_weights = (l1_interleaved[0], _transpose_sf_for_utccp(l1_interleaved[1]))
    # L2: only transpose SF for UTCCP
    l2_weights = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_weights, l2_weights


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    if torch.cuda.get_device_capability(y.device)[0] == 12:
        _fp8_fp4_mega_moe_sm120(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats,
            recipe,
            activation,
            activation_clamp,
            fast_math,
        )
        return

    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )
