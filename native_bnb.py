import torch
import torch.nn as nn

# NF4 Lookup Table (Official)
NF4_LUT = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491719007492065,
    -0.28444138169288635, -0.18477343022823334, -0.08820830285549164, 0.0,
    0.07107656449079514, 0.14263366162776947, 0.22430233657360077, 0.3203405737876892,
    0.4407068192958832, 0.5905631184577942, 0.796090841293335, 1.0
])

def dequantize_nf4(packed_weight, absmax, out_shape):
    """
    De-quantizes NF4 weights back to float using pure PyTorch.
    packed_weight: uint8 tensor of shape (out_features, in_features // 2)
    absmax: float tensor of shape (out_features, num_blocks)
    out_shape: (out_features, in_features)
    """
    device = packed_weight.device
    dtype = absmax.dtype
    lut = NF4_LUT.to(device=device, dtype=dtype)
    
    # Unpack higher and lower 4 bits using arithmetic ops (more stable kernels)
    # x >> 4 is div(x, 16)
    # x & 0xF is remainder(x, 16)
    higher = torch.div(packed_weight, 16, rounding_mode='floor').to(torch.int64)
    lower = torch.remainder(packed_weight, 16).to(torch.int64)
    
    # Interleave higher and lower
    # bitsandbytes usually stores as [lower0, higher0, lower1, higher1...]
    # or sometimes sharded. For standard pre-quantized files, we reshape correctly.
    unpacked = torch.stack([lower, higher], dim=-1).reshape(out_shape)
    
    # Lookup values from LUT
    dequant = lut[unpacked]
    
    # Apply absmax scaling
    # Standard block size for bnb is 64
    block_size = 64
    if absmax.numel() == out_shape[0]:
        # Single absmax per feature: (out_features, 1) * (out_features, in_features)
        dequant = dequant * absmax.view(-1, 1)
    else:
        # Block-wise absmax
        # Reshape dequant to (out_features, num_blocks, block_size)
        num_blocks = dequant.shape[1] // block_size
        dequant = dequant.view(dequant.shape[0], -1, block_size)
        
        # Scale each block. absmax shape is (out_features, num_blocks)
        dequant = dequant * absmax.unsqueeze(-1)
        dequant = dequant.reshape(out_shape)
        
    return dequant

class NativeLinear4bit(nn.Module):
    """
    A drop-in replacement for bnb.nn.Linear4bit using only PyTorch ops.
    """
    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        
        # These will be loaded from the state_dict
        # weight is packed uint8 (out, in//2)
        self.register_buffer("weight", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("absmax", torch.zeros((out_features, 1), dtype=compute_dtype))
        
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features, dtype=compute_dtype)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # On-the-fly de-quantization for standard Torch matmul
        # This bypasses bnb's custom CUDA kernels entirely.
        w = dequantize_nf4(self.weight, self.absmax, (self.out_features, self.in_features))
        
        # Cast to compute dtype if needed
        w = w.to(x.dtype)
        
        return torch.nn.functional.linear(x, w, self.bias)

    def to_native_linear(self, device="cuda"):
        """
        Convert this layer to a standard torch.nn.Linear layer.
        De-quantization is performed on the CPU for maximum stability.
        """
        # 1. Move to CPU for de-quantization (guaranteed stable kernels)
        packed = self.weight.to("cpu")
        absmax = self.absmax.to("cpu")
        
        # 2. De-quantize to the target compute dtype
        w = dequantize_nf4(packed, absmax, (self.out_features, self.in_features))
        w = w.to(self.compute_dtype)
        
        # 3. Create standard linear layer
        new_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        new_linear.weight.data.copy_(w)
        if self.bias is not None:
            new_linear.bias.data.copy_(self.bias.data.to("cpu").to(self.compute_dtype))
            
        return new_linear.to(device)

    @classmethod
    def from_bnb(cls, bnb_layer):
        """Helper to convert an existing bnb layer to native."""
        new_layer = cls(bnb_layer.in_features, bnb_layer.out_features, 
                        bias=bnb_layer.bias is not None,
                        compute_dtype=bnb_layer.compute_dtype)
        
        # Copy data
        new_layer.weight.copy_(bnb_layer.weight.data)
        if hasattr(bnb_layer.weight, 'quant_state'):
            new_layer.absmax.copy_(bnb_layer.weight.quant_state[0]) # index 0 is absmax
            
        if bnb_layer.bias is not None:
            new_layer.bias.data.copy_(bnb_layer.bias.data)
            
        return new_layer.to(bnb_layer.weight.device)
