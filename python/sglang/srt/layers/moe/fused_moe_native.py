"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

from typing import Callable, Optional

import torch
import torch_npu
from torch.nn import functional as F

from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
from sglang.srt.layers.moe.topk import TopKOutput


def fused_moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_output: TopKOutput,
    *,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    inplace: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:

    if apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    w13_weights = layer.w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = layer.w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if activation == "silu":
        x1 = F.silu(x1)
    elif activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))


def moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_output: TopKOutput,
    *,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    inplace: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:

    if apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    len_experts = layer.num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    if activation == "silu":
        act = SiluAndMul()
    elif activation == "gelu":
        act = GeluAndMul()
    else:
        raise ValueError(f"Unsupported activation: {activation=}")

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = layer.w13_weight[i]
        layer_w2_weight = layer.w2_weight[i]

        gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)
        gate_up = act(gate_up)
        expert_out = F.linear(gate_up, layer_w2_weight)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


def moe_forward_npu(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_output: TopKOutput,
    *,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    inplace: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:

    if apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output
    top_k = layer.top_k
    topk_ids = topk_ids.int()
    hidden_states, w1, w2 = x, layer.w13_weight, layer.w2_weight
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device

    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    sorted_hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    # TODO: Reorder device memory 2 times here, replace the current
    # implementation here when suitable operators become available.
    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        down_out_list,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    return final_hidden_states
