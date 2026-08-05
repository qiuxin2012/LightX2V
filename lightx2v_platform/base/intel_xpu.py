"""
Intel XPU Device implementation for LightX2V.

Intel XPU provides GPU acceleration through Intel's hardware (Arc GPU, Ponte Vecchio, etc.).
This module handles Intel-specific configurations including:
- XPU device initialization
- Distributed training with Intel oneCCL backend
"""

import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Detect Intel XPU platform
IS_INTEL_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@PLATFORM_DEVICE_REGISTER("intel_xpu")
class IntelXpuDevice:
    """
    Intel XPU Device implementation for LightX2V.

    Intel XPU uses torch.xpu APIs for GPU acceleration.
    Distributed training uses Intel oneCCL backend.
    """

    name = "intel_xpu"

    @staticmethod
    def init_device_env():
        """
        Initialize Intel XPU optimizations.

        This is called from lightx2v_platform.set_ai_device when platform is intel_xpu.
        Currently no specific optimizations needed for Intel XPU.
        """
        logger.info("Intel XPU platform detected, initializing environment...")
        logger.info(f"  - Available XPU devices: {torch.xpu.device_count()}")

    @staticmethod
    def is_available() -> bool:
        """Check if Intel XPU is available."""
        return IS_INTEL_XPU

    @staticmethod
    def get_device() -> str:
        """Get the device type string. Returns 'xpu' for Intel XPU."""
        return "xpu"

    @staticmethod
    def init_parallel_env():
        """Initialize a single-node distributed environment for Intel XPU."""
        local_rank = int(os.environ["LOCAL_RANK"])
        device_map_value = os.environ.get("LIGHTX2V_XPU_DEVICE_MAP", "").strip()
        if device_map_value:
            device_map = [int(value.strip()) for value in device_map_value.split(",")]
            world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", len(device_map))))
            if len(device_map) != world_size or sorted(device_map) != list(range(world_size)):
                raise ValueError(
                    "LIGHTX2V_XPU_DEVICE_MAP must be a permutation containing one device "
                    f"per local rank; expected 0..{world_size - 1}, got {device_map}"
                )
            device_index = device_map[local_rank]
            logger.info(f"Mapping local rank {local_rank} to physical xpu:{device_index}")
        else:
            device_index = local_rank
        torch.xpu.set_device(device_index)
        dist.init_process_group(backend="xccl")


# Register alias "xpu" for backward compatibility
PLATFORM_DEVICE_REGISTER["xpu"] = IntelXpuDevice
