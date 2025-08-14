"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""

from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionBlock,
    Qwen2_5_VisionPatchMerger,
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLMLP,
)
from sglang.srt.utils import add_prefix, is_npu

_is_npu = is_npu()

if _is_npu:
    import torch_npu


class AscendQwen2_5_VisionAttention(VisionAttention):

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        bsz, s, _ = x.shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        # [b, s, embed_dim] --> [b, s, embed_dim]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [b, s, embed_dim] --> [b, s, head, head_size]
        q = q.reshape(bsz, s, head, -1).contiguous()
        k = k.reshape(bsz, s, kv_head, -1).contiguous()
        v = v.reshape(bsz, s, kv_head, -1).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = torch_npu.npu_rotary_mul(q, cos, sin)
            k = torch_npu.npu_rotary_mul(k, cos, sin)

        # [b, s, head, head_size] --> [b * s, head, head_size]
        q, k, v = [rearrange(x, "b s ... -> (b s) ...").contiguous() for x in (q, k, v)]

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
        )

        assert output.dim() == 3, output.shape

        # [b * s, h, head_size] --> [b, s, h * head_size]
        output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

        # [b, s, h * head_size] --> [b, s, h * head_size]
        output, _ = self.proj(output)

        return output


class AscendQwen2_5_VisionBlock(Qwen2_5_VisionBlock):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
    ) -> None:
        super(Qwen2_5_VisionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.norm2 = RMSNorm(dim, eps=1e-6)

        softmax_in_single_precision = False
        qkv_backend = "ascend"
        flatten_batch = True

        self.attn = AscendQwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=flatten_batch,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class AscendQwen2_5_VisionPatchMerger(Qwen2_5_VisionPatchMerger):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, context_dim, spatial_merge_size, quant_config, prefix)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)


class AscendQwen2_5_VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.matmul(
            self.proj.weight.data.view(self.embed_dim, -1).transpose(0, 1)
        )
        return hidden_states


class AscendQwen2_5_VisionTransformer(Qwen2_5_VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(Qwen2_5_VisionTransformer, self).__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = vision_config.intermediate_size
        self.patch_embed = AscendQwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                AscendQwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(depth)
            ]
        )
        self.merger = AscendQwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()

        cos_new = torch.cat((cos, cos), dim=-1)
        sin_new = torch.cat((sin, sin), dim=-1)
        cos_new = cos_new.reshape(1, -1, 1, cos_new.shape[-1])
        sin_new = sin_new.reshape(1, -1, 1, sin_new.shape[-1])
        return cos_new, sin_new

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        position_embeddings = self.cal_cos_sin(rotary_pos_emb)

        # compute cu_seqlens
        cu_seqlens = (
            torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        ).to(torch.int32)

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)

        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x
