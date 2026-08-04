import os

import torch
import torch.distributed as dist
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.weights import (
    MiniMaxH3PostWeights,
    MiniMaxH3PreWeights,
    MiniMaxH3TransformerWeights,
)
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3Model(BaseTransformerModel):
    """LightX2V-native MiniMax-H3 joint audio/video transformer."""

    pre_weight_class = MiniMaxH3PreWeights
    transformer_weight_class = MiniMaxH3TransformerWeights
    post_weight_class = MiniMaxH3PostWeights

    def __init__(self, model_path, config, device):
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError(
                "MiniMax-H3 requires DTYPE=BF16. The native loader preserves the released checkpoint's 626 BF16 tensors and 12 FP32 projection/time/head tensors without dtype conversion."
            )
        parallel = config.get("parallel", {})
        tp_size = int(parallel.get("tensor_p_size", 1))
        sp_size = int(parallel.get("seq_p_size", 1))
        num_heads = int(config.get("num_attention_heads", 56))
        if num_heads % tp_size:
            raise ValueError(f"MiniMax-H3 tensor_p_size must divide {num_heads} attention heads, got {tp_size}")
        if config.get("seq_parallel", False) and (num_heads // tp_size) % sp_size:
            raise ValueError(
                "MiniMax-H3 Ulysses requires seq_p_size to divide TP-local attention heads: "
                f"global_heads={num_heads}, tensor_p_size={tp_size}, local_heads={num_heads // tp_size}, seq_p_size={sp_size}"
            )
        if config.get("cfg_parallel", False) or config.get("enable_cfg", False):
            raise ValueError("MiniMax-H3 is guidance-distilled and does not have a CFG/unconditional branch")
        if config.get("dit_quantized", False):
            raise NotImplementedError("MiniMax-H3 currently supports the released mixed BF16/FP32 weights")
        if config.get("lora_configs"):
            raise NotImplementedError("MiniMax-H3 LoRA loading is not implemented yet")
        offload_granularity = config.get("offload_granularity", "module")
        if config.get("cpu_offload", False) and offload_granularity not in {"model", "module", "block"}:
            raise ValueError(f"MiniMax-H3 offload_granularity must be 'module' or 'block', got {offload_granularity!r}")

        transformer_path = config.get("dit_original_ckpt") or os.path.join(model_path, "transformer")
        super().__init__(transformer_path, config, device)
        self.sensitive_layer = {
            "proj_in",
            "audio_proj_in",
            "time_embedder",
            "proj_out",
            "audio_proj_out",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load the released mixed-precision tensors without generic dtype coercion."""
        del unified_dtype, sensitive_layer
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"MiniMax-H3 native loading expects the released safetensors checkpoint; got {file_path}")
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        converted = {}
        with safe_open(file_path, framework="pt", device=str(self.device)) as source:
            for key in source.keys():
                if any(remove_key in key for remove_key in remove_keys):
                    continue
                if preserve_keys is not None and not any(preserve_key in key for preserve_key in preserve_keys):
                    continue
                self._convert_released_weight(converted, key, source.get_tensor(key))
        return converted

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        weights = super()._load_ckpt(unified_dtype, sensitive_layer)
        required = {
            "proj_in.weight",
            "audio_proj_in.weight",
            "context_embedder.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.49.attn.to_v.weight",
            "proj_out.weight",
            "audio_proj_out.weight",
        }
        missing = sorted(required.difference(weights))
        if missing or len(weights) != 638:
            raise ValueError(
                "MiniMax-H3 checkpoint conversion did not produce the released 638-tensor contract: "
                f"got {len(weights)}, missing={missing}. Check that all official transformer shards are present."
            )
        return weights

    def _store_converted_weight(self, destination, key, tensor):
        """Store only this rank's TP shard while streaming the checkpoint."""
        if not self.use_tp or not key.startswith("transformer_blocks."):
            destination[key] = tensor
            return

        column_parallel = (".attn.to_q.", ".attn.to_k.", ".attn.to_v.", ".adaln_proj.linear.")
        row_parallel = (".attn.to_out.0.", ".ff.net.2.")
        if any(marker in key for marker in column_parallel):
            if tensor.shape[0] % self.tp_size:
                raise ValueError(f"Cannot column-shard {key} with shape {tuple(tensor.shape)} over TP={self.tp_size}")
            tensor = torch.chunk(tensor, self.tp_size, dim=0)[self.tp_rank].contiguous()
        elif ".ff.net.0.proj." in key:
            if tensor.shape[0] % (2 * self.tp_size):
                raise ValueError(f"Cannot shard fused value/gate tensor {key} with shape {tuple(tensor.shape)} over TP={self.tp_size}")
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat(
                (torch.chunk(value, self.tp_size, dim=0)[self.tp_rank], torch.chunk(gate, self.tp_size, dim=0)[self.tp_rank]),
                dim=0,
            ).contiguous()
        elif any(marker in key for marker in row_parallel):
            if tensor.shape[1] % self.tp_size:
                raise ValueError(f"Cannot row-shard {key} with shape {tuple(tensor.shape)} over TP={self.tp_size}")
            tensor = torch.chunk(tensor, self.tp_size, dim=1)[self.tp_rank].contiguous()
        destination[key] = tensor

    def _convert_released_weight(self, destination, key, tensor):
        """Map the official fused H3 checkpoint to the native weight layout."""
        direct_names = {
            "video_patch_proj": "proj_in",
            "audio_patch_proj": "audio_proj_in",
            "condition_proj": "context_embedder",
            "time_embedder.proj_in": "time_embedder.linear_1",
            "time_embedder.proj_out": "time_embedder.linear_2",
            "final_layer.norm": "norm_out.norm",
            "final_layer.adaln_proj.linear": "norm_out.linear",
            "final_layer.video_out": "proj_out",
            "final_layer.audio_out": "audio_proj_out",
        }
        for source_name, target_name in direct_names.items():
            if key == source_name or key.startswith(source_name + "."):
                self._store_converted_weight(destination, target_name + key[len(source_name) :], tensor)
                return

        if key == "rope.inv_freq":
            # RoPE frequencies are reconstructed from config in pre-infer.
            return

        if key.startswith("token_refiner.blocks."):
            key = key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
        elif key.startswith("blocks."):
            key = key.replace("blocks.", "transformer_blocks.", 1)

        replacements = (
            (".attn.q_norm.", ".attn.norm_q."),
            (".attn.k_norm.", ".attn.norm_k."),
            (".attn.out_proj.", ".attn.to_out.0."),
            (".mlp.fc1.", ".ff.net.0.proj."),
            (".mlp.fc2.", ".ff.net.2."),
        )
        for source_name, target_name in replacements:
            key = key.replace(source_name, target_name)

        qkv_marker = ".attn.qkv_proj."
        if qkv_marker in key:
            if tensor.shape[0] % 3:
                raise ValueError(f"MiniMax-H3 fused QKV tensor has invalid shape: {key}={tuple(tensor.shape)}")
            prefix, suffix = key.split(qkv_marker, 1)
            q, k, v = tensor.chunk(3, dim=0)
            self._store_converted_weight(destination, f"{prefix}.attn.to_q.{suffix}", q)
            self._store_converted_weight(destination, f"{prefix}.attn.to_k.{suffix}", k)
            self._store_converted_weight(destination, f"{prefix}.attn.to_v.{suffix}", v)
            return

        self._store_converted_weight(destination, key, tensor)

    def _should_load_weights(self):
        # Every rank streams the official shards and immediately retains only
        # its TP-local tensors. This avoids materializing a full 62 GiB model
        # on rank 0 and works for CPU/block offload without device staging.
        return True

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        del is_weight_loader
        return weight_dict

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("MiniMax-H3 feature caching is not implemented")
        self.pre_infer_class = MiniMaxH3PreInfer
        self.transformer_infer_class = MiniMaxH3TransformerInfer
        self.post_infer_class = MiniMaxH3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()
            # Buffer construction uses the active device.  H3 loads the text
            # encoder and VAEs afterwards, so return the buffers to CPU until
            # the denoising stage actually starts.
            self.transformer_weights.offload_buffers_to_cpu()

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if not infer_condition:
            raise ValueError("MiniMax-H3 does not execute an unconditional pass")
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        pre = self.pre_infer.infer(self.pre_weight, prompt_embeds)
        if self.config.get("seq_parallel", False):
            pre = self._seq_parallel_pre_process(pre)
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre)
        output = self.post_infer.infer(self.post_weight, hidden_states, pre)
        if self.config.get("seq_parallel", False):
            self._current_sp_video_tail_length = pre.sp_video_tail_length
            output = self._seq_parallel_post_process(output)
        return output

    @torch.no_grad()
    def infer(self, inputs):
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        self.scheduler.video_noise_pred = output.video
        self.scheduler.audio_noise_pred = output.audio

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)

        video_indices = pre_infer_out.video_indices
        sharded_video_length = video_indices.numel() // world_size * world_size
        if sharded_video_length == 0:
            raise ValueError("MiniMax-H3 sequence parallel requires at least one video row per rank")
        main_indices = video_indices[:sharded_video_length]
        tail_indices = video_indices[sharded_video_length:]
        local_main = main_indices.chunk(world_size)[rank]

        # Preserve the relative order of all replicated non-main rows.  Dense,
        # non-causal attention is permutation equivariant, while RoPE remains
        # attached to the corresponding row.
        is_aux = torch.ones(pre_infer_out.hidden_states.shape[0], dtype=torch.bool, device=video_indices.device)
        is_aux[main_indices] = False
        aux_indices = is_aux.nonzero(as_tuple=False).flatten()
        order = torch.cat((local_main, aux_indices))

        pre_infer_out.hidden_states = pre_infer_out.hidden_states.index_select(0, order)
        pre_infer_out.timestep_indices = pre_infer_out.timestep_indices.index_select(0, order)
        pre_infer_out.adaln_indices = pre_infer_out.adaln_indices.index_select(0, order)
        pre_infer_out.rotary_emb = tuple(value.index_select(0, order) for value in pre_infer_out.rotary_emb)

        local_len = local_main.numel()
        tail_len = tail_indices.numel()
        aux_global_to_local = torch.full(
            (is_aux.numel(),), -1, dtype=torch.long, device=video_indices.device
        )
        aux_global_to_local[aux_indices] = torch.arange(aux_indices.numel(), device=video_indices.device) + local_len
        pre_infer_out.video_indices = torch.cat(
            (torch.arange(local_len, device=video_indices.device), aux_global_to_local[tail_indices])
        )
        pre_infer_out.audio_indices = aux_global_to_local[pre_infer_out.audio_indices]
        pre_infer_out.text_indices = aux_global_to_local[pre_infer_out.text_indices]
        pre_infer_out.sp_local_video_length = local_len
        pre_infer_out.sp_video_tail_length = tail_len
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, output):
        tail_len = self._current_sp_video_tail_length
        local_main = output.video[:-tail_len] if tail_len else output.video
        tail = output.video[-tail_len:] if tail_len else None
        gathered = [torch.empty_like(local_main) for _ in range(dist.get_world_size(self.seq_p_group))]
        dist.all_gather(gathered, local_main.contiguous(), group=self.seq_p_group)
        video = torch.cat(gathered, dim=0)
        if tail is not None:
            video = torch.cat((video, tail), dim=0)
        return type(output)(video=video, audio=output.audio)
