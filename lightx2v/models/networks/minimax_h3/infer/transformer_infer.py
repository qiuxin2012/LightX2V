import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[:, None, :]
    sin = sin.to(hidden_states.dtype)[:, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    rotary = rotary * cos + rotated * sin
    return torch.cat((rotary, passthrough), dim=-1).contiguous()


class MiniMaxH3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.hidden_size = int(config.get("hidden_size", 5376))
        self.tp_group = config.get("device_mesh").get_group(mesh_dim="tensor_p") if config.get("tensor_parallel", False) else None
        self.tp_size = dist.get_world_size(self.tp_group) if self.tp_group is not None else 1
        global_num_heads = int(config.get("num_attention_heads", 56))
        if global_num_heads % self.tp_size:
            raise ValueError(f"MiniMax-H3 attention heads {global_num_heads} are not divisible by TP size {self.tp_size}")
        self.num_heads = global_num_heads // self.tp_size
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.seq_parallel = config.get("seq_parallel", False)
        self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p") if self.seq_parallel else None
        self.seq_p_a2a_backend = config.get("parallel", {}).get("seq_p_a2a_backend", "torch")
        if self.seq_p_a2a_backend not in {"torch", "round_robin"}:
            raise ValueError(
                "MiniMax-H3 parallel.seq_p_a2a_backend must be 'torch' or "
                f"'round_robin', got {self.seq_p_a2a_backend!r}"
            )
        self.block_offload = config.get("cpu_offload", False) and config.get("offload_granularity") == "block"
        self.debug_sync = os.getenv("MINIMAX_H3_DEBUG_SYNC", "0") == "1"
        if self.block_offload:
            self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
        self.init_compile(config)

    def _debug_stage(self, stage, tensor):
        if not self.debug_sync:
            return
        logger.info(
            f"[H3 debug][rank={dist.get_rank()}] {stage}; "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
        torch_device_module.synchronize()

    def _attention(self, weights, hidden_states, rotary_emb, sp_local_video_length=0):
        self._debug_stage("attention: before QKV projections", hidden_states)
        q = weights.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        self._debug_stage("attention: after QKV projections", q)
        q = _apply_rotary_emb(weights.norm_q.apply(q), *rotary_emb)
        k = _apply_rotary_emb(weights.norm_k.apply(k), *rotary_emb)
        self._debug_stage("attention: after QK norm and RoPE", q)
        if self.seq_parallel:
            main = slice(0, sp_local_video_length)
            aux = slice(sp_local_video_length, None)
            out, aux_out = weights.parallel.apply_new(
                q=q[main],
                k=k[main],
                v=v[main],
                aux_q=q[aux],
                aux_k=k[aux],
                aux_v=v[aux],
                attention_module=weights.calculate,
                seq_p_group=self.seq_p_group,
                a2a_backend=self.seq_p_a2a_backend,
                aux_first=False,
                attention_kwargs={"causal": False},
            )
            out = torch.cat((out, aux_out), dim=0)
        else:
            seq_len = q.shape[0]
            cu_seqlens = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
            out = weights.calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_kv=seq_len,
                causal=False,
            )
        self._debug_stage("attention: after attention kernel", out)
        out = weights.to_out.apply(out.to(GET_DTYPE()))
        self._debug_stage("attention: after output projection", out)
        if self.tp_group is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            self._debug_stage("attention: after TP all-reduce", out)
        return out

    def _ff(self, weights, hidden_states):
        self._debug_stage("feed-forward: before input projection", hidden_states)
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        out = weights.out_proj.apply(value * F.silu(gate))
        self._debug_stage("feed-forward: after output projection", out)
        if self.tp_group is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            self._debug_stage("feed-forward: after TP all-reduce", out)
        return out

    def infer_block(self, weights, hidden_states, pre_infer_out):
        # Activation is evaluated in fp32, then cast immediately before the
        # checkpoint's bf16 AdaLN projection.
        modulation = weights.adaln.apply(F.silu(pre_infer_out.temb).to(GET_DTYPE()))
        if self.tp_group is not None:
            gathered_modulation = [torch.empty_like(modulation) for _ in range(self.tp_size)]
            dist.all_gather(gathered_modulation, modulation, group=self.tp_group)
            modulation = torch.cat(gathered_modulation, dim=-1)
        modulation = modulation.view(-1, 6 * self.hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        indices = pre_infer_out.adaln_indices

        residual = hidden_states
        normed = weights.norm1.apply(hidden_states)
        normed = normed * (1.0 + scale_msa.index_select(0, indices))
        normed = normed + shift_msa.index_select(0, indices)
        hidden_states = residual + gate_msa.index_select(0, indices) * self._attention(
            weights.attn,
            normed,
            pre_infer_out.rotary_emb,
            pre_infer_out.sp_local_video_length,
        )

        residual = hidden_states
        normed = weights.norm2.apply(hidden_states)
        normed = normed * (1.0 + scale_mlp.index_select(0, indices))
        normed = normed + shift_mlp.index_select(0, indices)
        hidden_states = residual + gate_mlp.index_select(0, indices) * self._ff(weights.ff, normed)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        blocks = block_weights.blocks
        debug_rank = dist.get_rank() if self.debug_sync else -1
        for block_index, block in enumerate(blocks):
            if self.debug_sync:
                logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: begin")
                torch_device_module.synchronize()
            if self.block_offload:
                if self.offload_manager.need_init_first_buffer:
                    if self.debug_sync:
                        logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: initialize first weight buffer")
                    self.offload_manager.init_first_buffer(blocks)
                if self.debug_sync:
                    logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: prefetch next weight buffer")
                self.offload_manager.prefetch_weights((block_index + 1) % len(blocks), blocks)
                block = self.offload_manager.cuda_buffers[0]
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    if self.debug_sync:
                        logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: compute")
                    hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
                if self.debug_sync:
                    torch_device_module.synchronize()
                    logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: compute complete")
                self.offload_manager.swap_blocks()
            else:
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
                if self.debug_sync:
                    torch_device_module.synchronize()
            if self.debug_sync:
                logger.info(f"[H3 debug][rank={debug_rank}] block={block_index}: end")
        return hidden_states
