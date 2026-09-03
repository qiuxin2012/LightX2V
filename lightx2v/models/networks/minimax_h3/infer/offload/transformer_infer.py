import torch
from loguru import logger

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class MiniMaxH3OffloadTransformerInfer(MiniMaxH3TransformerInfer):
    """Run H3 blocks through the same double-buffered prefetch path as Wan."""

    def __init__(self, config):
        super().__init__(config)
        offload_granularity = config.get("offload_granularity", "model")
        if offload_granularity == "block":
            self.use_event_offload = bool(config.get("use_event_offload", False))
            manager_cls = EventSlotWeightAsyncStreamManager if self.use_event_offload else WeightAsyncStreamManager
            self.offload_manager = manager_cls(offload_granularity="block")
            self.infer_func = self.infer_with_event_offload if self.use_event_offload else self.infer_with_blocks_offload
            if self.use_event_offload:
                logger.info("MiniMax-H3 block offload is using event-protected ping-pong slots")
        elif offload_granularity != "model":
            raise NotImplementedError(f"MiniMax-H3 does not support offload_granularity={offload_granularity!r}")

    def get_compile_block_key(self, block_idx, block):
        # block offload
        if hasattr(self, "offload_manager"):
            return id(block)
        # model offload
        return super().get_compile_block_key(block_idx, block)

    def _prefetch_weights_without_adaln(self, block_index, blocks):
        with torch_device_module.stream(self.offload_manager.cuda_load_stream):
            if hasattr(self.offload_manager, "cpu_buffers"):
                source_block = self.offload_manager.cpu_buffers[0]
            else:
                source_block = blocks[block_index]
            block_state_dict = source_block.state_dict()
            weights_without_adaln = {}
            for name, tensor in block_state_dict.items():
                if ".adaln_proj." not in name:
                    weights_without_adaln[name] = tensor
            self.offload_manager.cuda_buffers[1].load_state_dict(weights_without_adaln, block_index)

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        num_blocks = len(blocks)
        if self.use_adaln_cache and not self._adaln_cache_hit:
            # The previous forward may have prefetched block 0 without AdaLN.
            # Reload the full block when the current timestep misses.
            self.offload_manager.need_init_first_buffer = True
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        for block_index in range(num_blocks):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            next_block_index = (block_index + 1) % num_blocks
            if self.use_adaln_cache and self._adaln_cache_hit:
                self._prefetch_weights_without_adaln(next_block_index, blocks)
            else:
                self.offload_manager.prefetch_weights(next_block_index, blocks)
            block = self.offload_manager.cuda_buffers[0]
            self.block_idx = block_index
            if AI_DEVICE == "xpu":
                # Match Wan's XPU offload path: overlap the next weight copy on
                # the load stream with current-block compute on the default
                # stream, then let swap_blocks() perform the device-wide sync.
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            self.offload_manager.swap_blocks()

        return hidden_states

    def infer_with_event_offload(self, blocks, hidden_states, pre_infer_out):
        """Overlap block H2D copies and compute without host/device-wide barriers."""
        manager = self.offload_manager
        num_blocks = len(blocks)
        if num_blocks == 0:
            return hidden_states

        current_stream = torch_device_module.current_stream()
        # Keep XPU collectives and compute on the caller stream, matching the
        # proven Qwen3-VL event-offload path.  Only H2D copies use the separate
        # load stream. Other backends retain the dedicated compute stream.
        compute_stream = current_stream if AI_DEVICE == "xpu" else manager.compute_stream
        if compute_stream is not current_stream:
            compute_stream.wait_stream(current_stream)

        scheduled_slots = {}
        next_block_idx = 0

        def prefetch_next(slot_idx):
            nonlocal next_block_idx
            if next_block_idx >= num_blocks:
                return
            manager.prefetch_to_slot(slot_idx, next_block_idx, blocks)
            scheduled_slots[next_block_idx] = slot_idx
            next_block_idx += 1

        try:
            for slot_idx in range(min(manager.slot_count, num_blocks)):
                prefetch_next(slot_idx)

            for block_idx in range(num_blocks):
                slot_idx = scheduled_slots.pop(block_idx)
                block = manager.wait_ready(slot_idx, stream=compute_stream)
                self.block_idx = block_idx
                with torch_device_module.stream(compute_stream):
                    hidden_states = self.run_block(block_idx, block, hidden_states, pre_infer_out)
                manager.record_free(slot_idx, stream=compute_stream)
                prefetch_next(slot_idx)

            if compute_stream is not current_stream:
                with torch_device_module.stream(compute_stream):
                    final_done = compute_stream.record_event()
                current_stream.wait_event(final_done)
                hidden_states.record_stream(current_stream)
            return hidden_states
        except Exception:
            # Prevent an in-flight copy from overwriting a slot during retry
            # or teardown after a failed block.
            torch_device_module.synchronize()
            manager.reset_slots()
            raise
