import torch
import torch.nn as nn
import sys

def test_op(name, func):
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        func()
        print("PASS")
        return True
    except Exception as e:
        print(f"FAILED (Error: {str(e)})")
        return False

def run_diagnostics():
    device = "cuda"
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {props.name} (CC {props.major}.{props.minor}, {props.total_memory//1024**3}GB VRAM)")
    else:
        print("CUDA not available!")
        return

    # --- Test 1: Basic Float Math ---
    test_op("Float32 Matmul", lambda: torch.matmul(torch.randn(10, 10).cuda(), torch.randn(10, 10).cuda()))
    test_op("Float16 Matmul", lambda: torch.matmul(torch.randn(10, 10).cuda().half(), torch.randn(10, 10).cuda().half()))
    test_op("BFloat16 Matmul", lambda: torch.matmul(torch.randn(10, 10).cuda().to(torch.bfloat16), torch.randn(10, 10).cuda().to(torch.bfloat16)))

    # --- Test 2: Bitwise Ops (Used in NF4) ---
    def bitwise_test():
        u = torch.tensor([255, 0], dtype=torch.uint8).cuda()
        res1 = u >> 4
        res2 = u & 0xF
    test_op("Uint8 Bitwise Shift (>>)", lambda: bitwise_test())

    # --- Test 3: Indexing/LUT (Used in NF4) ---
    def lut_test():
        lut = torch.randn(16).cuda()
        idx = torch.tensor([0, 15, 7], dtype=torch.long).cuda()
        res = lut[idx]
    test_op("GPU Indexing (LUT)", lambda: lut_test())

    # --- Test 4: Linear Layer (F.linear) ---
    def linear_test(dtype):
        x = torch.randn(2, 16).cuda().to(dtype)
        w = torch.randn(32, 16).cuda().to(dtype)
        b = torch.randn(32).cuda().to(dtype)
        return torch.nn.functional.linear(x, w, b)
    
    test_op("Linear F32", lambda: linear_test(torch.float32))
    test_op("Linear F16", lambda: linear_test(torch.float16))
    test_op("Linear BF16", lambda: linear_test(torch.bfloat16))

if __name__ == "__main__":
    run_diagnostics()
