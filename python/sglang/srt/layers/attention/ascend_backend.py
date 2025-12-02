from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.npu_ops.mla_preprocess import is_mla_preprocess_enabled
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


import numpy as np

import triton
import triton.language as tl
import torch


@triton.jit
def _paged_gqa_fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    Att_Lse,
    block_table,
    stride_block_table_batch: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kpage: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vpage: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    stride_mid_ob: tl.constexpr,
    stride_mid_oh: tl.constexpr,
    stride_mid_os: tl.constexpr,
    stride_mid_lb: tl.constexpr,
    stride_mid_lh: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    MTP_STEP: tl.constexpr,
):
    """
    Forward kernel for Grouped-Query Attention (GQA) with paged KV cache.

    GQA allows multiple query heads to share the same key/value heads,
    reducing memory bandwidth while preserving expressiveness.

    Uses online softmax for numerical stability during incremental decoding.

    Args:
        Q (Tensor): Queries, shape [batch_size, q_head_num, Lk].
        K_Buffer (Tensor): Paged key cache, shape [num_blocks, page_size, kv_head_num, Lk].
        V_Buffer (Tensor): Paged value cache, shape [num_blocks, page_size, kv_head_num, Lv].
        sm_scale (float): Attention scaling factor.
        kv_seq_lens (Tensor): Sequence lengths, shape [batch_size].
        Att_Out (Tensor): Output buffer, shape [batch_size, q_head_num, Lv].
        block_table (Tensor): Logical-to-physical block mapping.
        ... (strides): Memory strides.
        q_heads_per_kv_head (int): Ratio of Q heads to KV heads.
        q_head_num (int): Total number of query heads.
        BLOCK_DMODEL (int): Tiled size for key dimension (padded).
        BLOCK_DPE (int): Optional extra dim for RoPE (can be zero).
        BLOCK_DV (int): Tiled size for value dimension (padded).
        BLOCK_N (int): Page size.
        BLOCK_H (int): Number of query heads processed per block.
        Lk (int): Actual key dimension.
        Lv (int): Actual value dimension.
    """
    cur_batch = tl.program_id(0)
    cur_head_group_id = tl.program_id(1)
    cur_kv_head = cur_head_group_id // tl.cdiv(q_heads_per_kv_head, BLOCK_H)

    if BLOCK_H < q_heads_per_kv_head:
        HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        HEAD_NUM: tl.constexpr = q_heads_per_kv_head
    cur_q_head_start = cur_head_group_id * HEAD_NUM

    # Step 1: Load Q_nope + Q_rope
    offset_h = cur_q_head_start + tl.arange(0, HEAD_NUM)
    offset_d = tl.arange(0, BLOCK_DMODEL + BLOCK_DPE)
    mask_h = offset_h < q_head_num
    mask_d = offset_d < Lk

    step_1 = 0
    step_2 = 1
    offset_q1 = (cur_batch * MTP_STEP + step_1) * stride_qbs + offset_h[:, None] * stride_qh + offset_d[None, :]
    q1 = tl.load(Q + offset_q1, mask=(mask_h[:, None]) & (mask_d[None, :]))
    offset_q2 = (cur_batch * MTP_STEP + step_2) * stride_qbs + offset_h[:, None] * stride_qh + offset_d[None, :]
    q2 = tl.load(Q + offset_q2, mask=(mask_h[:, None]) & (mask_d[None, :]))
    q = tl.zeros([2 * HEAD_NUM, BLOCK_DMODEL + BLOCK_DPE], dtype=q2.dtype)
    q = tl.insert_slice(
        q,
        q1,
        offsets=(0, 0),
        sizes=(HEAD_NUM, BLOCK_DMODEL + BLOCK_DPE),
        strides=(1, 1)
    )
    q = tl.insert_slice(
        q,
        q2,
        offsets=(HEAD_NUM, 0),
        sizes=(HEAD_NUM, BLOCK_DMODEL + BLOCK_DPE),
        strides=(1, 1)
    )

    # Step 2: Iterate over physical blocks using PagedAttention
    kv_split_id = tl.program_id(2)
    cur_split_num = tl.num_programs(2)
    cur_kv_seq_len = tl.load(kv_seq_lens + cur_batch)
    page_num = tl.cdiv(cur_kv_seq_len, BLOCK_N)
    page_num_per_split = page_num // cur_split_num
    residual_page = page_num % cur_split_num

    if kv_split_id < residual_page:
        page_num_per_split += 1
        cur_kv_split_start = kv_split_id * page_num_per_split
    else:
        cur_kv_split_start = kv_split_id * page_num_per_split + residual_page
    cur_page_start = cur_batch * stride_block_table_batch + cur_kv_split_start

    offset_page = tl.arange(0, BLOCK_N)
    offset_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offset_dv < Lv

    if page_num_per_split > 0:  # 去掉试一下
        history_max = tl.zeros([2 * HEAD_NUM], dtype=tl.float32) - float('inf')
        l = tl.zeros([2 * HEAD_NUM], dtype=tl.float32)
        acc = tl.zeros([2 * HEAD_NUM, BLOCK_DV], dtype=tl.float32)

        for page_id in range(page_num_per_split - 1):
            # Load K
            page_loc = tl.load(block_table + cur_page_start + page_id)
            offset_k = (page_loc * stride_buf_kbs
                        + offset_page[:, None] * stride_buf_kpage
                        + cur_kv_head * stride_buf_kh
                        + offset_d[None, :])
            mask_page = (cur_kv_split_start + page_id * BLOCK_N + offset_page) < cur_kv_seq_len
            k = tl.load(K_Buffer + offset_k, mask=(mask_page[:, None] & mask_d[None, :]))
            k = tl.trans(k, (1, 0))
            qk = tl.dot(q, k)

            # Load V early to overlap with computation
            offset_v = (page_loc * stride_buf_vbs
                        + offset_page[:, None] * stride_buf_vpage
                        + cur_kv_head * stride_buf_vh
                        + offset_dv[None, :])
            v = tl.load(V_Buffer + offset_v, mask=(mask_page[:, None] & mask_dv[None, :]))

            qk = qk * sm_scale
            new_e_max = tl.maximum(tl.max(qk, 1), history_max)
            re_scale = tl.exp(history_max - new_e_max)
            p_exp = tl.exp(qk - new_e_max[:, None])

            l = l * re_scale + tl.sum(p_exp, 1)
            acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
            history_max = new_e_max
        # end if
        history_max1 = tl.extract_slice(
            history_max,
            offsets=(0,),
            sizes=(HEAD_NUM,),
            strides=(1,)
        )
        history_max2 = tl.extract_slice(
            history_max,
            offsets=(HEAD_NUM,),
            sizes=(HEAD_NUM,),
            strides=(1,)
        )
        l1 = tl.extract_slice(
            l,
            offsets=(0,),
            sizes=(HEAD_NUM,),
            strides=(1,)
        )
        l2 = tl.extract_slice(
            l,
            offsets=(HEAD_NUM,),
            sizes=(HEAD_NUM,),
            strides=(1,)
        )
        acc1 = tl.extract_slice(
            acc,
            offsets=(0, 0),
            sizes=(HEAD_NUM, BLOCK_DV),
            strides=(1, 1)
        )
        acc2 = tl.extract_slice(
            acc,
            offsets=(HEAD_NUM, 0),
            sizes=(HEAD_NUM, BLOCK_DV),
            strides=(1, 1)
        )
        # =============================================== tail block start
        # qk1 qk2
        page_id = page_num_per_split - 1
        page_loc = tl.load(block_table + cur_page_start + page_id)
        offset_k = (page_loc * stride_buf_kbs
                    + offset_page[:, None] * stride_buf_kpage
                    + cur_kv_head * stride_buf_kh
                    + offset_d[None, :])
        mask_page_0 = (cur_kv_split_start + page_id * BLOCK_N + offset_page) < cur_kv_seq_len - 1
        mask_page_1 = (cur_kv_split_start + page_id * BLOCK_N + offset_page) < cur_kv_seq_len
        k = tl.load(K_Buffer + offset_k, mask=(mask_page_1[:, None] & mask_d[None, :]))
        k = tl.trans(k, (1, 0))
        qk1 = tl.dot(q1, k)
        qk2 = tl.dot(q2, k)
        offset_v = (page_loc * stride_buf_vbs
                    + offset_page[:, None] * stride_buf_vpage
                    + cur_kv_head * stride_buf_vh
                    + offset_dv[None, :])
        v = tl.load(V_Buffer + offset_v, mask=(mask_page_1[:, None] & mask_dv[None, :]))
        # softmax1
        qk1 = qk1 * sm_scale
        qk1 = tl.where((mask_h[:, None] & mask_page_0[None, :]), qk1, float("-inf"))
        new_e_max1 = tl.maximum(tl.max(qk1, 1), history_max1)
        re_scale1 = tl.exp(history_max1 - new_e_max1)
        p_exp1 = tl.exp(qk1 - new_e_max1[:, None])

        l1 = l1 * re_scale1 + tl.sum(p_exp1, 1)
        acc1 = acc1 * re_scale1[:, None] + tl.dot(p_exp1.to(v.dtype), v)
        history_max1 = new_e_max1
        # softmax2
        qk2 = qk2 * sm_scale
        qk2 = tl.where((mask_h[:, None] & mask_page_1[None, :]), qk2, float("-inf"))
        new_e_max2 = tl.maximum(tl.max(qk2, 1), history_max2)
        re_scale2 = tl.exp(history_max2 - new_e_max2)
        p_exp2 = tl.exp(qk2 - new_e_max2[:, None])

        l2 = l2 * re_scale2 + tl.sum(p_exp2, 1)
        acc2 = acc2 * re_scale2[:, None] + tl.dot(p_exp2.to(v.dtype), v)
        history_max2 = new_e_max2
        # ================================================== tail block end

        offs_mid_o = ((cur_batch * MTP_STEP + step_1) * stride_mid_ob
                      + offset_h[:, None] * stride_mid_oh
                      + kv_split_id * stride_mid_os
                      + offset_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc1 / l1[:, None], mask=(mask_h[:, None] & mask_dv[None, :]))

        offs_mid_o_lse = (cur_batch * MTP_STEP + step_1) * stride_mid_lb + offset_h * stride_mid_lh + kv_split_id
        tl.store(Att_Lse + offs_mid_o_lse, history_max1 + tl.log(l1), mask=mask_h)

        offs_mid_o = ((cur_batch * MTP_STEP + step_2) * stride_mid_ob
                      + offset_h[:, None] * stride_mid_oh
                      + kv_split_id * stride_mid_os
                      + offset_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc2 / l2[:, None], mask=(mask_h[:, None] & mask_dv[None, :]))

        offs_mid_o_lse = (cur_batch * MTP_STEP + step_2) * stride_mid_lb + offset_h * stride_mid_lh + kv_split_id
        tl.store(Att_Lse + offs_mid_o_lse, history_max2 + tl.log(l2), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_l,
    O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lb,
    stride_mid_lh,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_lb + cur_head * stride_mid_lh

    for split_kv_id in range(0, MAX_KV_SPLITS):
        acc_logits = tl.load(
            Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d
        )
        tlogic = tl.load(Mid_O_l + offs_logic + split_kv_id)
        n_e_max = tl.maximum(tlogic, e_max)

        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * acc_logits

        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def decode_gqa(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    out,
    kv_seq_lens,
    max_kv_splits,
    sm_scale,
    page_size,
    block_table,
):
    """
    Wrapper function to launch GQA forward kernel.

    Handles special cases for known architectures (e.g., DeepSeek-V3 uses split K).

    Args:
        q (Tensor): Input queries
        k_buffer (Tensor): Paged key cache
        v_buffer (Tensor): Paged value cache
        att_out (Tensor): Output buffer
        kv_seq_lens (Tensor): Sequence lengths
        sm_scale (float): Attention scale
        page_size (int): Size of each KV block
        block_table (Tensor): Block mapping table
    """
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    BLOCK_N = page_size
    BLOCK_H = 16
    # Special-case tiling for models like DeepSeek-V3 which split K into model+RoPE parts
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)
    MTP_STEP = 2

    batch, q_head_num = kv_seq_lens.shape[0], q.shape[1]
    kv_head_num = k_buffer.shape[2]
    q_heads_per_kv_head = q_head_num // kv_head_num
    assert q_head_num % kv_head_num == 0, "head_num must be divisible by kv_head_num"
    grid = (
        batch,
        triton.cdiv(q_head_num, min(BLOCK_H, q_heads_per_kv_head)),
        max_kv_splits,
    )
    _paged_gqa_fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_seq_lens,
        att_out,
        att_lse,
        block_table,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_lse.stride(0),
        att_lse.stride(1),
        q_heads_per_kv_head=q_heads_per_kv_head,
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        Lk=Lk,
        Lv=Lv,
        MTP_STEP=2,
        limit_auto_multi_buffer_only_for_local_buffer=True,
        multibuffer=False
    )

    grid = (batch * MTP_STEP, q_head_num)
    _fwd_kernel_stage2[grid](
        att_out,
        att_lse,
        out,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_lse.stride(0),
        att_lse.stride(1),
        out.stride(0),
        out.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
    )


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None


class AscendAttnBackend(AttentionBackend):

    def gen_attention_mask(self, max_seq_len: int, dtype=torch.float16):
        mask_flag = torch.tril(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        ).view(max_seq_len, max_seq_len)
        mask_flag = ~mask_flag
        if dtype == torch.float16:
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        self.mask = (
            torch.masked_fill(
                torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
            )
            .to(dtype)
            .to(self.device)
        )
        self.mask_len = max_seq_len

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.q_head_dim = (
                self.qk_rope_head_dim + model_runner.model_config.qk_nope_head_dim
            )
        self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        if not self.use_fia:
            self.gen_attention_mask(128, model_runner.dtype)
        mask_length = 2048
        self.fia_mask = ~torch.tril(
            torch.ones(
                (mask_length, mask_length),
                dtype=torch.bool,
                device=model_runner.device,
            )
        )
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.mtp_mask = torch.tril(torch.ones(2048, 2048, dtype=torch.bool)).npu()
        self.mtp_mask = ~self.mtp_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        tp_size = get_attention_tp_size()
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i * 1 for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)
        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        block_table = self.forward_metadata.block_tables
        if is_prefill:
            actual_seq_qlen = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            if self.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    actual_seq_qlen = (
                        torch.arange(
                            self.speculative_num_draft_tokens,
                            self.speculative_num_draft_tokens + q.shape[0],
                            self.speculative_num_draft_tokens,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                elif forward_batch.forward_mode.is_draft_extend():
                    actual_seq_qlen = (
                        forward_batch.extend_seq_lens.cumsum()
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q
        if self.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int

        attn_out = torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope,
            value=k_nope,
            query_rope=q_pe,
            key_rope=k_pe,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(torch.int32),
            actual_seq_lengths_kv=actual_seq_lengths_kv.to(q.device),
            block_table=block_table,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )

        return attn_out

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):

            if is_mla_preprocess_enabled():
                save_kv_cache = False
            return self.forward_mtp(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )

            else:
                if layer.qk_head_dim <= 128:
                    query = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                    attn_output = torch.empty(
                        (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                        dtype=query.dtype,
                        device=query.device,
                    )

                    torch_npu._npu_flash_attention_qlens(
                        query=query,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        mask=self.mask,
                        block_table=self.forward_metadata.block_tables,
                        seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                        context_lens=self.forward_metadata.seq_lens_cpu_int,
                        scale_value=layer.scaling,
                        num_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        out=attn_output,
                    )
                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(q)

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    causal = True
                    if (
                        layer.is_cross_attention
                        or layer.attn_type == AttentionType.ENCODER_ONLY
                    ):
                        causal = False

                    self.native_attn._run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                    )
        else:
            assert (
                layer.qk_head_dim != layer.v_head_dim
            ), "FIA only supports qk_head_dim != v_head_dim"

            # Wait for the KV transfer to complete before performing attention computation.
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            num_token_padding = q.shape[0]
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]

            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=layer.tp_q_head_num,
                input_layout="TND",
                atten_mask=self.fia_mask,
                sparse_mode=3,
                actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                scale=layer.scaling,
                next_tokens=0,
            )

            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
            if num_token_padding != forward_batch.num_token_non_padded_cpu:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        return attn_output

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
        if not self.use_mla:
            # 一段 + TND ok
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            actual_seq_lengths = np.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens + query.shape[0],
                self.speculative_num_draft_tokens,
            )  # (start, stop, step)

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                atten_mask=self.mtp_mask,
                scale=layer.scaling,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                sparse_mode=3,
            )  # torch_npu.npu_fused_infer_attention_score
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - forward_batch.num_token_non_padded_cpu, *attn_output.shape[1:]
                        ),
                    ],
                    dim=0,
                )
            return attn_output
            """ triton """
            # k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
            #     layer.layer_id).view(-1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim)
            # v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
            #     layer.layer_id).view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
            # query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
            # if not self.graph_mode:
            #     num_token_padding = query.shape[0]
            #     query = query[: forward_batch.num_token_non_padded_cpu]
            # if self.graph_mode:
            #     actual_seq_lengths_kv = self.forward_metadata.seq_lens
            # else:
            #     if self.forward_metadata.seq_lens_cpu_int is None:
            #         actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            #     else:
            #         actual_seq_lengths_kv = (
            #             self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            #         )
            #     actual_seq_lengths_kv = torch.tensor(actual_seq_lengths_kv, dtype=torch.int32, device=query.device)
            #
            # max_kv_splits = 1
            # attn_output = torch.empty_like(query)
            # # 用empty初始化性能劣化
            # attn_logits = torch.empty(
            #     query.shape[0], layer.tp_q_head_num, max_kv_splits, layer.v_head_dim,
            #     device=query.device, dtype=query.dtype
            # )
            # attn_lse = torch.empty(
            #     query.shape[0], layer.tp_q_head_num, max_kv_splits, device=query.device, dtype=query.dtype
            # )
            # sm_scale = 1.0 / (layer.qk_head_dim ** 0.5)
            # decode_gqa(
            #     query,
            #     k_cache,
            #     v_cache,
            #     attn_logits,
            #     attn_lse,
            #     attn_output,
            #     actual_seq_lengths_kv,
            #     max_kv_splits,
            #     sm_scale,
            #     self.page_size,
            #     self.forward_metadata.block_tables,
            # )
            #
            # attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            # if (
            #     not self.graph_mode
            #     and forward_batch.num_token_non_padded_cpu != num_token_padding
            # ):
            #     attn_output = torch.cat(
            #         [
            #             attn_output,
            #             attn_output.new_zeros(
            #                 num_token_padding - forward_batch.num_token_non_padded_cpu, *attn_output.shape[1:]
            #             ),
            #         ],
            #         dim=0,
            #     )
            # return attn_output
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_rope_cache = k_rope.view(
                -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
            )
            c_kv_cache = c_kv.view(
                -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
            )

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + q_nope.shape[0],
                    self.speculative_num_draft_tokens,
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
            )
            attn_output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                workspace=workspace,
                out=[attn_output, softmax_lse],
            )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0], *attn_output.shape[1:]
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if not self.use_mla:
            num_tokens = q.shape[0]
            """PA will support bs<tp in the later version of CANN"""
            if True:
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
                query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                num_tokens = query.shape[0]
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query,
                        k_cache,
                        v_cache,
                        block_table=self.forward_metadata.block_tables,
                        block_size=self.page_size,
                        num_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="BSH",
                        scale=layer.scaling,
                        actual_seq_lengths_kv=actual_seq_len_kv,
                    )
                )
                output = torch.empty(
                    (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
                torch_npu.npu_fused_infer_attention_score.out(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSH",
                    scale=layer.scaling,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    workspace=workspace,
                    out=[output, softmax_lse],
                )
                return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
            else:
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = torch.from_numpy(
                        np.array(self.forward_metadata.seq_lens_cpu_list).astype(
                            np.int32
                        )
                    )
                else:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=actual_seq_len_kv,
                    out=attn_output,
                )
                return attn_output.view(
                    num_tokens, layer.tp_q_head_num * layer.v_head_dim
                )
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_rope_cache = k_rope.view(
                -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
            )
            c_kv_cache = c_kv.view(
                -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
            )

            q_nope = q.view(-1, layer.tp_q_head_num, 1, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, 1, self.qk_rope_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if is_mla_preprocess_enabled():
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )

        if self.graph_mode:
            return self.forward_decode_graph(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            num_tokens = q.shape[0]
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            if self.use_fia:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        forward_batch.batch_size,
                        -1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim
                    ),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                    scale=layer.scaling,
                )
            else:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                kv_c = kv_c.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )
                k_pe = k_pe.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
            else:
                assert (
                    self.graph_mode == False
                )  # _npu_paged_attention_mla not support graph mode
                q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(AscendAttnBackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
