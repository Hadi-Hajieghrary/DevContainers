import torch
import sys

def test_pytorch():
    print("--- PyTorch Installation Check ---")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

        # Simple tensor operation on GPU
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("✓ GPU Tensor Operation: Success")

        # Test basic neural network operation
        linear = torch.nn.Linear(3, 2).cuda()
        output = linear(x)
        print(f"✓ Neural Network Operation: Success (output shape: {output.shape})")
    else:
        print("✗ GPU not found. Check nvidia-container-toolkit installation on host.")
        print("  PyTorch will run on CPU only.")

if __name__ == "__main__":
    print(f"Python Interpreter: {sys.executable}\n")
    test_pytorch()
    print("\nPyTorch installation test complete.")