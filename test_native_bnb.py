import torch
from native_bnb import NativeLinear4bit

def test():
    print("Starting Native NF4 Test...")
    in_feat = 16
    out_feat = 32
    device = "cuda"
    dtype = torch.float16
    
    # Create native layer
    layer = NativeLinear4bit(in_feat, out_feat, compute_dtype=dtype).to(device)
    
    # Create fake input
    x = torch.randn(2, in_feat, device=device, dtype=dtype)
    
    # Run forward pass (this triggers dequantize_nf4)
    print("Running forward pass...")
    try:
        y = layer(x)
        print(f"Output shape: {y.shape}")
        if y.shape == (2, out_feat):
            print("Native NF4 Forward Pass: SUCCESS")
        else:
            print(f"Native NF4 Forward Pass: FAILED (Wrong shape {y.shape})")
    except Exception as e:
        print(f"Native NF4 Forward Pass: FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
