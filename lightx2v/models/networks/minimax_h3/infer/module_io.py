from dataclasses import dataclass

import torch


@dataclass
class MiniMaxH3PreInferOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    timestep_indices: torch.Tensor
    adaln_indices: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    # In sequence-parallel mode hidden_states is laid out as
    # [local sharded video rows | replicated auxiliary rows].  The auxiliary
    # rows contain every non-video token plus the (at most sp_size - 1) video
    # rows needed to make the sharded prefix evenly divisible.
    sp_local_video_length: int = 0
    sp_video_tail_length: int = 0


@dataclass
class MiniMaxH3VelocityOutput:
    video: torch.Tensor
    audio: torch.Tensor
