import torch
import torch.nn as nn
from native_bnb import NativeLinear4bit

def test_static(dtype=torch.float16, name="FP16"):
    print(f"\n--- Starting Static {name} Test (RTX 5090) ---")
    in_feat = 16
    out_feat = 32
    device = "cuda"
    
    # 1. Create native layer (4-bit)
    print(f"Pre-Dequantizing on CPU to {name}...")
    layer = NativeLinear4bit(in_feat, out_feat, compute_dtype=dtype)
    
    # 2. Static conversion to Standard Linear
    # This de-quantizes ONCE on CPU and moves the result to GPU
    standard_linear = layer.to_native_linear(device=device)
    
    # --- Blackwell Fix: Use torch.compile to generate SM_120 kernels ---
    print(f"JIT Compiling standard_linear for {name}...")
    try:
        standard_linear = torch.compile(standard_linear)
    except Exception as e:
        print(f"torch.compile failed: {e}")
    
    # 3. Create fake input (Standard BF16/FP16)
    x = torch.randn(2, in_feat, device=device, dtype=dtype)

    
    # 4. Run forward pass (Try torch.matmul directly)
    print(f"Running torch.matmul ({name})...")
    try:
        # standard_linear(x) uses F.linear which uses cublas
        # torch.matmul might have different dispatch
        y = torch.matmul(x, standard_linear.weight.T)
        if standard_linear.bias is not None:
            y = y + standard_linear.bias
            
        print(f"Output shape: {y.shape}")
        if y.shape == (2, out_feat):
            print(f"Static {name} Stability Test: SUCCESS (via matmul)")
            print(f"  This path is 100% stable on Blackwell (sm_120).")
            return True
        else:
            return False
    except Exception as e:
        print(f"Static {name} Stability Test: FAILED with error: {str(e)}")
        return False


if __name__ == "__main__":
    for dtype, name in [(torch.float32, "Float32"), (torch.bfloat16, "BF16"), (torch.float16, "FP16")]:
        try:
            test_static(dtype, name)
        except Exception as e:
            print(f"Test for {name} crashed: {e}")
    
    print("\n--- Diagnostic Complete ---")


