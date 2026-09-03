import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.offload.block_slab import pack_cpu_block_slab
from lightx2v.models.networks.minimax_h3.infer.triton_ops import MiniMaxH3TritonRope  # noqa: F401
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER


def _linear(config, name, bias=False, create_cuda_buffer=False, tp_split=None):
    lora_prefix = "transformer_blocks"
    if config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        tp_mm_type = config.get("tp_mm_type", "TensorParallel")
        weight = MM_WEIGHT_REGISTER[tp_mm_type](
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=config.get("dit_quant_scheme", "Default"),
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
            lora_column_chunks=2 if ".ff.net.0.proj" in name else 1,
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix=lora_prefix,
        )
    else:
        weight = MM_WEIGHT_REGISTER[config.get("dit_quant_scheme", "Default")](
            f"{name}.weight",
            f"{name}.bias" if bias else None,
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix=lora_prefix,
        )
    weight.set_config(config)
    return weight


def _rms(config, name, eps, create_cuda_buffer=False):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](
        name,
        create_cuda_buffer=create_cuda_buffer,
        eps=eps,
    )


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("to_q", _linear(config, f"{prefix}.to_q", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_k", _linear(config, f"{prefix}.to_k", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_v", _linear(config, f"{prefix}.to_v", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        self.add_module(
            "norm_q",
            _rms(
                config,
                f"{prefix}.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "norm_k",
            _rms(
                config,
                f"{prefix}.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "torch_real_rope")](
                layout="split_half",
                compute_dtype=torch.float32,
            ),
        )
        attn_type = config.get("attn_type", "flash_attn3")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            calculate = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            calculate = attention_cls()
        if attn_type == "sol_attn":
            calculate.set_config(config.get("sol_attn_setting", {}))
        self.add_module("calculate", calculate)
        if config.get("seq_parallel", False):
            parallel = config.get("parallel", {})
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[parallel.get("seq_p_attn_type", "ulysses")](a2a_backend=parallel.get("seq_p_a2a_backend", "torch")),
            )
        self.add_module("to_out", _linear(config, f"{prefix}.to_out.0", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("in_proj", _linear(config, f"{prefix}.net.0.proj", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("out_proj", _linear(config, f"{prefix}.net.2", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3TransformerBlockWeights(WeightModule):
    def __init__(self, index, config, create_cuda_buffer=False):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module(
            "norm1",
            _rms(
                config,
                f"{prefix}.norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("attn", MiniMaxH3AttentionWeights(f"{prefix}.attn", config, create_cuda_buffer))
        self.add_module(
            "norm2",
            _rms(
                config,
                f"{prefix}.norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", config, create_cuda_buffer))
        # AdaLN is the largest per-block projection in H3.  Its output is
        # column-sharded here and gathered once per block before modulation.
        self.add_module("adaln", _linear(config, f"{prefix}.adaln_proj.linear", bias=True, create_cuda_buffer=create_cuda_buffer, tp_split="col"))


class MiniMaxH3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        if config.get("lazy_load", False):
            raise NotImplementedError(
                "MiniMax-H3 reads the official sharded checkpoint directly; disk lazy_load requires a converted block-sharded checkpoint and is not supported yet. Use lazy_load=false with model or block CPU offload."
            )
        self.blocks = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config) for i in range(int(config.get("num_layers", 50)))])
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "block":
            self.offload_block_cuda_buffers = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config, create_cuda_buffer=True) for i in range(2)])
            # Register device buffers before source blocks: buffer allocation
            # needs checkpoint metadata that normal CPU loading consumes.
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", self.blocks)

    def prepare_offload_block_slabs(self):
        """Pack each CPU block into one pinned transfer allocation."""
        if not self.config.get("offload_use_block_slab", False):
            return {}
        if hasattr(self, "offload_block_slabs"):
            return self.offload_block_slabs

        slabs = {}
        for block_idx, block in enumerate(self.blocks):
            state = block.state_dict()
            if not state:
                raise ValueError(f"MiniMax-H3 block {block_idx} has no tensors to pack")
            non_cpu = [name for name, tensor in state.items() if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"]
            if non_cpu:
                raise ValueError(f"MiniMax-H3 block slab requires CPU tensors; block {block_idx} has invalid entries {non_cpu}")
            slabs[block_idx] = pack_cpu_block_slab(
                state,
                pin_memory=True,
                strict_pin=True,
            )
            self._replace_block_cpu_tensors_with_slab_views(block, slabs[block_idx])

        self.offload_block_slabs = slabs
        total_bytes = sum(slab.layout.nbytes for slab in slabs.values())
        logger.info(
            "MiniMax-H3 packed {} block-offload slabs ({:.2f} GiB per rank)",
            len(slabs),
            total_bytes / (1024**3),
        )
        return slabs

    @staticmethod
    def _replace_block_cpu_tensors_with_slab_views(block, slab):
        """Let weight modules share the slab storage instead of duplicating it."""
        visited = set()

        def visit(module):
            if module is None or id(module) in visited:
                return
            visited.add(id(module))
            storage = getattr(module, "_mm", module)
            for source_name, attr_name, _ in getattr(storage, "base_attrs", ()):
                pin_attr = f"pin_{attr_name}"
                if source_name in slab.views and getattr(storage, pin_attr, None) is not None:
                    setattr(storage, pin_attr, slab.views[source_name])
            for child in getattr(module, "_parameters", {}).values():
                visit(child)
            for child in getattr(module, "_modules", {}).values():
                visit(child)

        visit(block)
