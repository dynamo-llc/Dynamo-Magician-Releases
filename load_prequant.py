#!/usr/bin/env python3
"""
Load pre-quantized bitsandbytes NF4 models without needing base weights.

This module provides utilities to load pre-quantized WanModel weights
directly from safetensors/pt files, without downloading or loading
the original FP16/BF16 base model weights.

Usage:
    from load_prequant import load_quantized_model

    model = load_quantized_model("lingbot-world-base-cam/high_noise_model_bnb_nf4")
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState

# ── bitsandbytes version guard ────────────────────────────────────────────────
# A very old or mismatched bitsandbytes can segfault (access violation) inside
# Linear4bit.__init__ before Python gets a chance to handle any exception.
# Fail loudly here so the error event reaches the UI instead of silently
# killing the server process.
def _require_bnb_version(min_tuple=(0, 45, 0)):
    try:
        parts = tuple(int(x) for x in bnb.__version__.split(".")[:3] if x.isdigit())
        if parts < min_tuple:
            raise RuntimeError(
                f"bitsandbytes {bnb.__version__} is too old (need >={'.'.join(map(str, min_tuple))}).\n"
                "Please run setup.bat to reinstall the correct environment."
            )
    except RuntimeError:
        raise
    except Exception:
        pass  # Non-standard version string; allow through

_require_bnb_version()

# Add parent to path for wan imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.model import WanModel, rope_params
from native_bnb import NativeLinear4bit

def is_blackwell():
    """Detect if the current GPU is Blackwell (CC 12.0+).

    Always returns True for sm_120+ regardless of PyTorch arch_list.
    Blackwell GPUs always use the NativeLinear4bit path (pure-PyTorch
    dequant + static FP16 migration), which avoids bitsandbytes CUDA
    kernel incompatibilities on Blackwell regardless of bnb version.
    """
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        return props.major >= 12
    except:
        return False


def replace_linears_with_bnb_nf4(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
    quant_type: str = "nf4",
) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Replace all nn.Linear layers with empty bnb.nn.Linear4bit layers.

    This creates the structure needed to load pre-quantized weights.
    The layers are created without weights - they will be populated
    by load_state_dict afterwards.

    Args:
        model: The model to modify in-place
        compute_dtype: Compute dtype for the quantized layers
        compress_statistics: Whether to use double quantization
        quant_type: Quantization type ('nf4' or 'fp4')

    Returns:
        Tuple of (num_replaced, dict mapping layer_name to (in_features, out_features))
    """
    replaced = 0
    layer_shapes = {}

    # Collect all linear layers first to avoid modifying during iteration
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for name, module in linear_layers:
        # Get parent module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        # Store original shape for reconstruction
        layer_shapes[name] = (module.in_features, module.out_features)

        # Use Native NF4 on Blackwell (Native sm_120 Support)
        if is_blackwell():
            nf4_linear = NativeLinear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16 # Standard stable F16
            )
        else:
            # Create empty NF4 linear layer with same shape
            nf4_linear = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )

        # Replace the layer (weights will be loaded from state_dict)
        setattr(parent, child_name, nf4_linear)
        replaced += 1

    return replaced, layer_shapes


def build_model_from_config(config: Dict[str, Any]) -> WanModel:
    """
    Build a WanModel instance from config dictionary.

    Uses the meta device to avoid allocating ~56 GB of FP32 weights that
    would immediately be thrown away by replace_linears_with_bnb_nf4.

    Args:
        config: Dictionary with model configuration

    Returns:
        WanModel instance (non-Linear params materialised on CPU)
    """
    # Build on meta device: creates the full module tree with zero memory.
    # init_weights() runs but is a no-op on meta tensors.
    with torch.device("meta"):
        model = WanModel(
            model_type=config.get("model_type", "i2v"),
            patch_size=tuple(config.get("patch_size", (1, 2, 2))),
            text_len=config.get("text_len", 512),
            in_dim=config.get("in_dim", 16),
            dim=config.get("dim", 2048),
            ffn_dim=config.get("ffn_dim", 8192),
            freq_dim=config.get("freq_dim", 256),
            text_dim=config.get("text_dim", 4096),
            out_dim=config.get("out_dim", 16),
            num_heads=config.get("num_heads", 16),
            num_layers=config.get("num_layers", 32),
            window_size=tuple(config.get("window_size", (-1, -1))),
            qk_norm=config.get("qk_norm", True),
            cross_attn_norm=config.get("cross_attn_norm", True),
            eps=config.get("eps", 1e-6),
        )

    # Materialise only non-Linear parameters (norms, embeddings, etc.) on CPU.
    # Linear layers stay on meta — they are about to be replaced with NF4.
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            # Skip nn.Linear weight/bias — will be replaced by NF4 layers
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
            else:
                parent = model
            attr = parts[-1] if len(parts) == 2 else parts[0]
            is_linear = isinstance(parent, nn.Linear)
            if not is_linear:
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                setattr(parent, attr, new_param)

    # Materialise buffers on CPU
    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta"):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
            else:
                parent = model
            attr = parts[-1] if len(parts) == 2 else parts[0]
            parent.register_buffer(
                attr,
                torch.empty(buf.shape, dtype=buf.dtype, device="cpu"),
            )

    # Recompute plain tensor attributes that were created as meta tensors.
    # model.freqs is NOT a parameter/buffer (by design, to avoid dtype
    # conversion during .to()), so it must be recomputed on CPU explicitly.
    dim = config.get("dim", 2048)
    num_heads = config.get("num_heads", 16)
    d = dim // num_heads
    model.freqs = torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1)

    return model


def reconstruct_params4bit_from_components(
    weight_components: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> bnb.nn.Params4bit:
    """
    Reconstruct a Params4bit object from serialized components using QuantState.from_dict.

    This uses bitsandbytes' own deserialization method for correctness.

    Args:
        weight_components: Dict with keys like 'weight', 'absmax', 'quant_map',
                          'nested_absmax', 'nested_quant_map', 'quant_state_data'
        device: Device to load to

    Returns:
        Reconstructed Params4bit
    """
    # Build the dict that QuantState.from_dict expects
    qs_dict = {
        "absmax": weight_components["absmax"],
        "quant_map": weight_components["quant_map"],
    }

    # Add nested quantization components if present (double quantization)
    if "nested_absmax" in weight_components:
        qs_dict["nested_absmax"] = weight_components["nested_absmax"]
        qs_dict["nested_quant_map"] = weight_components["nested_quant_map"]

    # Add the packed quant_state data (contains shape, dtype, etc.)
    if "quant_state_data" in weight_components:
        qs_dict["quant_state.bitsandbytes__nf4"] = weight_components["quant_state_data"]

    # Use bitsandbytes' own deserialization
    quant_state = QuantState.from_dict(qs_dict, device=torch.device(device))

    # Get quantized weight and move to device
    quantized_weight = weight_components["weight"].to(device)

    # Create Params4bit with the quantized data
    param = bnb.nn.Params4bit(
        data=quantized_weight,
        requires_grad=False,
        quant_state=quant_state,
        bnb_quantized=True,  # Already quantized, don't re-quantize on .to()
    )

    return param


def load_quantized_state(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str = "cpu",
) -> nn.Module:
    """
    Load quantized weights into a model with bnb.Linear4bit layers.

    This function handles the special bitsandbytes serialization format
    where weights are decomposed into quantized data + metadata tensors.

    Uses QuantState.from_dict for proper deserialization.

    Args:
        model: Model with Linear4bit layers already in place
        weights_path: Path to model.safetensors or model.pt
        layer_shapes: Dict mapping layer names to (in_features, out_features)
        device: Device to load quantized weights to

    Returns:
        Model with loaded weights
    """
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(weights_path)
    else:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Group keys by their base weight name
    weight_components = defaultdict(dict)
    other_keys = {}

    # Quantization-related suffixes
    quant_suffixes = [".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4"]

    for key, tensor in sd.items():
        base_key = None
        component = None

        if ".weight.absmax" in key:
            base_key = key.replace(".weight.absmax", "")
            component = "absmax"
        elif ".weight.quant_map" in key:
            base_key = key.replace(".weight.quant_map", "")
            component = "quant_map"
        elif ".weight.nested_absmax" in key:
            base_key = key.replace(".weight.nested_absmax", "")
            component = "nested_absmax"
        elif ".weight.nested_quant_map" in key:
            base_key = key.replace(".weight.nested_quant_map", "")
            component = "nested_quant_map"
        elif ".weight.quant_state.bitsandbytes__nf4" in key:
            base_key = key.replace(".weight.quant_state.bitsandbytes__nf4", "")
            component = "quant_state_data"
        elif key.endswith(".weight"):
            # Check if this is a quantized linear weight or regular weight
            potential_base = key[:-7]  # Remove ".weight"
            has_quant_metadata = any(f"{potential_base}.weight{suffix}" in sd for suffix in quant_suffixes)

            if has_quant_metadata:
                base_key = potential_base
                component = "weight"
            else:
                other_keys[key] = tensor
                continue
        else:
            other_keys[key] = tensor
            continue

        if base_key and component:
            weight_components[base_key][component] = tensor

    # Load quantized weights into model
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, NativeLinear4bit)):
            if name in weight_components:
                components = weight_components[name]

                if "weight" in components:
                    if isinstance(module, NativeLinear4bit):
                        # Use bnb's full QuantState deserialization to properly
                        # recover the true float absmax.  With double quantization
                        # (compress_statistics=True) the stored "absmax" is uint8
                        # and must be dequantized using nested_absmax/quant_map.
                        param = reconstruct_params4bit_from_components(
                            components, device="cpu"
                        )
                        qs = param.quant_state

                        # Recover true float absmax (pure PyTorch, no CUDA)
                        if hasattr(qs, 'nested') and qs.nested:
                            s2 = qs.state2
                            code = s2.code.float()          # (256,) lookup table
                            indices = qs.absmax.long()       # (n,) quantized uint8
                            lookup = code[indices]           # (n,) normalized
                            bs2 = s2.blocksize
                            n = indices.numel()
                            scale = s2.absmax.float().repeat_interleave(bs2)[:n]
                            true_absmax = lookup * scale
                            if hasattr(qs, 'offset') and qs.offset is not None:
                                true_absmax += qs.offset.float()
                        else:
                            true_absmax = qs.absmax.float()

                        # Store packed weight (reshape flat → 2D)
                        module.weight.copy_(
                            param.data.reshape(module.weight.shape).to(device)
                        )

                        # Store true absmax in block-aligned 2D layout
                        block_size = qs.blocksize  # typically 64
                        absmax_2d = true_absmax.reshape(
                            module.out_features,
                            module.in_features // block_size,
                        ).to(module.compute_dtype).to(device)
                        module.register_buffer("absmax", absmax_2d)

                        # Store the model's actual NF4 code table
                        if "quant_map" in components:
                            module.register_buffer(
                                "quant_map",
                                components["quant_map"].float().to(device),
                            )
                        loaded_count += 1
                    else:
                        # Use bitsandbytes' own deserialization via QuantState.from_dict
                        param = reconstruct_params4bit_from_components(components, device=device)
                        module.weight = param
                        loaded_count += 1

            # Load bias if present
            bias_key = f"{name}.bias"
            if bias_key in other_keys and module.bias is not None:
                module.bias.data.copy_(other_keys[bias_key].to(device))

    # Load non-quantized weights (embeddings, norms, biases, etc.)
    non_linear_sd = {}
    for key, tensor in other_keys.items():
        non_linear_sd[key] = tensor

    if non_linear_sd:
        missing, unexpected = model.load_state_dict(non_linear_sd, strict=False)
        expected_missing = {f"{name}.weight" for name in layer_shapes.keys()}
        critical_missing = [k for k in missing if k not in expected_missing and not k.endswith("freqs")]
        if critical_missing:
            print(f"Warning: Missing non-quantized keys: {critical_missing[:10]}...")

    print(f"  Loaded {loaded_count} quantized linear layers")
    return model


def load_quantized_model(
    model_dir: str,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> WanModel:
    """
    Load a pre-quantized WanModel from a directory.

    This function:
    1. Reads config.json to get model architecture
    2. Builds an empty WanModel
    3. Replaces Linear layers with bnb.Linear4bit
    4. Loads the pre-quantized weights with proper reconstruction
    5. Moves model to device

    Args:
        model_dir: Directory containing config.json and model.safetensors/model.pt
        device: Device to load model to
        compute_dtype: Compute dtype for quantized layers

    Returns:
        Loaded and ready WanModel
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check for quantization metadata (optional but recommended)
    meta_path = model_dir / "quantization_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        quant_config = meta.get("quant", {})
        compute_dtype_str = quant_config.get("compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

    # Find weights file (prefer safetensors)
    safetensors_path = model_dir / "model.safetensors"
    pt_path = model_dir / "model.pt"

    if safetensors_path.exists():
        weights_path = str(safetensors_path)
    elif pt_path.exists():
        weights_path = str(pt_path)
    else:
        raise FileNotFoundError(
            f"No weights found in {model_dir}. "
            "Expected model.safetensors or model.pt"
        )

    print(f"Loading pre-quantized model from {model_dir}")
    print(f"  Config: {config_path}")
    print(f"  Weights: {weights_path}")

    # Build model from config (creates initialized weights we'll replace)
    model = build_model_from_config(config)

    # Replace Linear → Linear4bit (empty, ready for state_dict)
    replaced, layer_shapes = replace_linears_with_bnb_nf4(model, compute_dtype=compute_dtype)
    print(f"  Replaced {replaced} linear layers")

    # Load quantized weights with proper reconstruction
    # This loads quantized weights directly to the target device
    model = load_quantized_state(model, weights_path, layer_shapes, device=device)

    # --- Blackwell Optimization: One-Time Static Migration ---
    # When loading to CPU (as the pipeline does), skip the FP16 migration to
    # keep the model compact in NF4 format (~9 GB vs ~28 GB).  The pipeline's
    # _prepare_model_for_timestep will perform the migration when the model is
    # first moved to GPU.
    if is_blackwell() and device != "cpu":
        import gc
        print("  Blackwell detected! Migrating to 100% Native FP16 layers...")
        native_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, NativeLinear4bit)
        ]
        for name, module in native_layers:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            # Perform one-time CPU de-quantization and static conversion
            static_linear = module.to_native_linear(device=device)
            setattr(parent, child_name, static_linear)

            # Free the NF4 buffers immediately so peak RAM doesn't double
            del module
        del native_layers
        gc.collect()
        print("  Migration COMPLETE. bitsandbytes kernels bypassed.")

    # Move non-quantized parts to device and set eval mode
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"  Model ready on {device}")

    return model


def verify_quantized_model(model: nn.Module) -> Dict[str, Any]:
    """
    Verify that a model has been properly quantized.

    Args:
        model: Model to verify

    Returns:
        Dictionary with verification results
    """
    total_params = 0
    quantized_params = 0
    linear4bit_count = 0
    regular_linear_count = 0

    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear4bit_count += 1
            if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
                quantized_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            regular_linear_count += 1

    for param in model.parameters():
        total_params += param.numel()

    return {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "linear4bit_count": linear4bit_count,
        "regular_linear_count": regular_linear_count,
        "is_quantized": linear4bit_count > 0 and regular_linear_count == 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test loading pre-quantized model")
    parser.add_argument("model_dir", type=str, help="Path to quantized model directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load to")
    args = parser.parse_args()

    model = load_quantized_model(args.model_dir, device=args.device)

    info = verify_quantized_model(model)
    print(f"\nVerification:")
    print(f"  Linear4bit layers: {info['linear4bit_count']}")
    print(f"  Regular Linear layers: {info['regular_linear_count']}")
    print(f"  Is properly quantized: {info['is_quantized']}")
