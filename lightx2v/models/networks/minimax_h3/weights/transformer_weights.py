import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def _linear(name, bias=False, create_cuda_buffer=False):
    return MM_WEIGHT_REGISTER["Default"](
        f"{name}.weight",
        f"{name}.bias" if bias else None,
        create_cuda_buffer=create_cuda_buffer,
    )


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("to_q", _linear(f"{prefix}.to_q", create_cuda_buffer=create_cuda_buffer))
        self.add_module("to_k", _linear(f"{prefix}.to_k", create_cuda_buffer=create_cuda_buffer))
        self.add_module("to_v", _linear(f"{prefix}.to_v", create_cuda_buffer=create_cuda_buffer))
        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        self.add_module("norm_q", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm_q.weight", create_cuda_buffer=create_cuda_buffer, eps=qk_eps))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm_k.weight", create_cuda_buffer=create_cuda_buffer, eps=qk_eps))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[config.get("attn_type", "flash_attn3")]())
        if config.get("seq_parallel", False):
            attn_type = config.get("parallel", {}).get("seq_p_attn_type", "ulysses")
            if attn_type != "ulysses":
                raise ValueError(f"MiniMax-H3 sequence parallel currently supports only Ulysses, got {attn_type!r}")
            self.add_module("parallel", ATTN_WEIGHT_REGISTER["ulysses"]())
        self.add_module("to_out", _linear(f"{prefix}.to_out.0", create_cuda_buffer=create_cuda_buffer))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, create_cuda_buffer=False):
        super().__init__()
        self.add_module("in_proj", _linear(f"{prefix}.net.0.proj", create_cuda_buffer=create_cuda_buffer))
        self.add_module("out_proj", _linear(f"{prefix}.net.2", create_cuda_buffer=create_cuda_buffer))


class MiniMaxH3TransformerBlockWeights(WeightModule):
    def __init__(self, index, config, create_cuda_buffer=False):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module("norm1", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm1.weight", create_cuda_buffer=create_cuda_buffer, eps=eps))
        self.add_module("attn", MiniMaxH3AttentionWeights(f"{prefix}.attn", config, create_cuda_buffer))
        self.add_module("norm2", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm2.weight", create_cuda_buffer=create_cuda_buffer, eps=eps))
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", create_cuda_buffer))
        self.add_module("adaln", _linear(f"{prefix}.adaln_proj.linear", bias=True, create_cuda_buffer=create_cuda_buffer))


class MiniMaxH3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        if config.get("lazy_load", False):
            raise NotImplementedError(
                "MiniMax-H3 reads the official sharded checkpoint directly; per-block lazy_load requires a block-sharded checkpoint and is not supported yet. Use cpu_offload with module or block granularity."
            )
        blocks = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config) for i in range(int(config.get("num_layers", 50)))])
        if config.get("cpu_offload", False) and config.get("offload_granularity") == "block":
            self.offload_block_cuda_buffers = WeightModuleList(
                [MiniMaxH3TransformerBlockWeights(i, config, create_cuda_buffer=True) for i in range(2)]
            )
            # Buffers must load before the CPU-resident blocks consume entries
            # from the checkpoint dictionary.
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", blocks)

    def _iter_offload_buffer_leaves(self):
        if not hasattr(self, "offload_block_cuda_buffers"):
            return

        def walk(module):
            for child in getattr(module, "_modules", {}).values():
                if child is not None:
                    yield from walk(child)
            for parameter in getattr(module, "_parameters", {}).values():
                if parameter is not None:
                    yield parameter
            if not hasattr(module, "_modules") and not hasattr(module, "_parameters"):
                yield module

        yield from walk(self.offload_block_cuda_buffers)

    def move_offload_buffers(self, device):
        """Move the two reusable block buffers without touching 50 CPU blocks."""
        for leaf in self._iter_offload_buffer_leaves():
            for buffer_name in ("weight_cuda_buffer", "bias_cuda_buffer"):
                buffer = getattr(leaf, buffer_name, None)
                if not isinstance(buffer, torch.Tensor):
                    continue
                buffer = buffer.to(device)
                setattr(leaf, buffer_name, buffer)
                setattr(leaf, buffer_name.removesuffix("_cuda_buffer"), buffer)

    def offload_buffers_to_device(self):
        self.move_offload_buffers(AI_DEVICE)

    def offload_buffers_to_cpu(self):
        self.move_offload_buffers("cpu")
