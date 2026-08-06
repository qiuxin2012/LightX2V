"""MiniMax-H3 tensor-parallel weight helpers.

The common :class:`MMWeightTP` wrapper owns a concrete MM implementation in
``_mm``.  H3 also uses the wrapper with model/block offload, so its tensor
lifecycle must be delegated to that concrete implementation.
"""

import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP


class MiniMaxH3TensorParallelLinear(MMWeightTP):
    """Tensor-parallel linear that remains compatible with H3 offload."""

    def __init__(self, *args, use_all_gather_reduce=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_all_gather_reduce = use_all_gather_reduce

    def apply(self, input_tensor):
        """Apply the projection and combine row-parallel partial results."""
        output = self._mm.apply(input_tensor)

        if self.split_dim == "row" and self.reduce_output and self.tp_size > 1 and self.tp_group is not None:
            # On XPU, TP all-reduce interleaved with Ulysses SP collectives can
            # stall oneCCL.  Gathering the partials and summing locally is
            # mathematically equivalent to a SUM all-reduce.
            if self.use_all_gather_reduce and output.device.type == "xpu":
                partials = [torch.empty_like(output) for _ in range(self.tp_size)]
                dist.all_gather(partials, output.contiguous(), group=self.tp_group)
                output.copy_(partials[0])
                for partial in partials[1:]:
                    output.add_(partial)
            else:
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)

            if self._row_split_bias is not None:
                output = output + self._row_split_bias

        return output

    def set_config(self, config=None):
        config = {} if config is None else config
        self.config = config
        self._mm.set_config(config)

    def state_dict(self, destination=None):
        return self._mm.state_dict(destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return self._mm.load_state_dict(destination, block_index, adapter_block_index)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        return self._mm.load_state_dict_from_disk(block_index, adapter_block_index)

    def to_cuda(self, non_blocking=False):
        return self._mm.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        return self._mm.to_cpu(non_blocking=non_blocking)


def unwrap_tp_linear(module):
    """Return the concrete tensor-owning MM implementation."""
    return module._mm if isinstance(module, MiniMaxH3TensorParallelLinear) else module


__all__ = ["MiniMaxH3TensorParallelLinear", "unwrap_tp_linear"]
