# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, dependency-free loader for MiniMax-H3 safetensors components.

The released H3 component directories may contain either one safetensors file
or several indexed shards.  The VAE inference wrappers are decode-only, so
loading an entire shard into a temporary state dict would waste many gigabytes
on encoder weights.  This helper scans the files and assigns only the exact
parameters/buffers present in the target module, one tensor at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


@dataclass(frozen=True)
class SafetensorsSubsetReport:
    component_dir: Path
    files: tuple[Path, ...]
    loaded_keys: tuple[str, ...]
    ignored_keys: int


def _component_files(component_dir: str | Path) -> tuple[Path, ...]:
    component_dir = Path(component_dir)
    if not component_dir.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 component directory does not exist: {component_dir}")
    files = tuple(sorted(component_dir.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in MiniMax-H3 component directory: {component_dir}")
    return files


def _get_parent(module: nn.Module, key: str) -> tuple[nn.Module, str]:
    parts = key.split(".")
    parent: nn.Module = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _assign_tensor(module: nn.Module, key: str, tensor: torch.Tensor) -> None:
    parent, name = _get_parent(module, key)
    if name in parent._parameters:
        old_parameter = parent._parameters[name]
        requires_grad = False if old_parameter is None else old_parameter.requires_grad
        parent._parameters[name] = nn.Parameter(tensor, requires_grad=requires_grad)
    elif name in parent._buffers:
        parent._buffers[name] = tensor
    else:
        raise KeyError(f"Cannot assign checkpoint tensor {key!r}: it is not a parameter or buffer")


_SAFETENSORS_DTYPES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _expected_specs(module: nn.Module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {key: (tuple(value.shape), value.dtype) for key, value in module.state_dict().items()}


def _checkpoint_targets(key: str) -> tuple[str, ...]:
    """Return native names for an official (occasionally fused) VAE tensor."""
    if key == "decoder.mask_token":
        # The native decoder constructs the always-zero class token at runtime.
        return ()
    key = key.replace("decoder.x_embedder.", "decoder.proj_in.")
    key = key.replace(".attn.to_out.", ".attn.to_out.0.")
    key = key.replace(".ff.w1.", ".ff.net.0.proj.")
    key = key.replace(".ff.w2.", ".ff.net.2.")
    key = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", key)
    key = key.replace(".nin_shortcut.", ".conv_shortcut.")
    marker = ".attn.to_qkv."
    if marker in key:
        prefix, suffix = key.split(marker, 1)
        return tuple(f"{prefix}.attn.to_{name}.{suffix}" for name in ("q", "k", "v"))
    return (key,)


def validate_safetensors_subset(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Validate exact key and shape parity without materializing checkpoint tensors."""

    files = _component_files(component_dir)
    expected = _expected_specs(module)
    expected_roots = {key.partition(".")[0] for key in expected}
    found: set[str] = set()
    unexpected: list[str] = []
    ignored = 0

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                targets = _checkpoint_targets(key)
                if not targets:
                    ignored += 1
                    continue
                tensor_slice = checkpoint.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                checkpoint_dtype = str(tensor_slice.get_dtype())
                if len(targets) == 3:
                    if not shape or shape[0] % 3:
                        raise ValueError(f"Invalid fused QKV shape for {key!r}: {shape}")
                    shape = (shape[0] // 3, *shape[1:])
                for target in targets:
                    if target not in expected:
                        if target.partition(".")[0] in expected_roots:
                            unexpected.append(key)
                        else:
                            ignored += 1
                        continue
                    expected_shape, expected_dtype = expected[target]
                    if shape != expected_shape:
                        raise ValueError(f"Shape mismatch for {target!r}: model expects {expected_shape}, checkpoint contains {shape}")
                    if checkpoint_dtype != _SAFETENSORS_DTYPES.get(expected_dtype):
                        raise TypeError(f"Dtype mismatch for {target!r}: model expects {expected_dtype}, checkpoint contains {checkpoint_dtype}")
                    if target in found:
                        unexpected.append(f"duplicate:{target}")
                    found.add(target)

    missing = sorted(set(expected) - found)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        raise RuntimeError("MiniMax-H3 checkpoint does not match the native module: " + ", ".join(details))

    return SafetensorsSubsetReport(Path(component_dir), files, tuple(sorted(found)), ignored)


def load_safetensors_subset(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Load exactly ``module.state_dict()`` from one or more original H3 shards.

    This also supports a module constructed on the ``meta`` device: each meta
    parameter is replaced directly by its CPU checkpoint tensor, avoiding a
    second full-size initialized copy of the VAE.
    """

    report = validate_safetensors_subset(module, component_dir)
    expected = set(module.state_dict())
    loaded: set[str] = set()

    for filename in report.files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                targets = tuple(target for target in _checkpoint_targets(key) if target in expected)
                if not targets:
                    continue
                tensor = checkpoint.get_tensor(key)
                tensors = tensor.chunk(3, dim=0) if len(targets) == 3 else (tensor,)
                for target, value in zip(targets, tensors):
                    _assign_tensor(module, target, value)
                    loaded.add(target)

    # validate_safetensors_subset already checked this, but keep the invariant
    # local to the mutating operation as well.
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Failed to load MiniMax-H3 tensors: {missing[:20]}")
    return SafetensorsSubsetReport(report.component_dir, report.files, tuple(sorted(loaded)), report.ignored_keys)
