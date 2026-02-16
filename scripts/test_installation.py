#!/usr/bin/env python3
"""Test script to diagnose PyTorch and CUDA installation issues."""

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("=== PyTorch Diagnostic ===")

    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {getattr(torch, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    # Check CUDA support
    if hasattr(torch, 'cuda'):
        try:
            cuda_available = torch.cuda.is_available()
            cuda_devices = torch.cuda.device_count() if cuda_available else 0
            print("✅ torch.cuda module available")
            print(f"   CUDA available: {cuda_available}")
            print(f"   CUDA devices: {cuda_devices}")

            if cuda_available:
                for i in range(cuda_devices):
                    try:
                        print(f"     Device {i}: {torch.cuda.get_device_name(i)}")
                    except Exception as e:
                        print(f"     Device {i}: Unknown ({e})")
        except Exception as e:
            print(f"❌ CUDA check failed: {e}")
    else:
        print("❌ torch.cuda module NOT available - This is a CPU-only PyTorch build")

    return True

def check_ultralytics():
    """Check if ultralytics can be imported."""
    print("\n=== Ultralytics Diagnostic ===")

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False

def main():
    """Run all diagnostics."""
    print("PyTorch/CUDA Diagnostic Tool")
    print("=" * 50)

    pytorch_ok = check_pytorch()
    ultralytics_ok = check_ultralytics()

    print("\n=== Summary ===")
    if not pytorch_ok:
        print("❌ PyTorch is not properly installed")
    if pytorch_ok and not hasattr(torch, 'cuda'):
        print("❌ PyTorch lacks CUDA support")
    if pytorch_ok and hasattr(torch, 'cuda') and not torch.cuda.is_available():
        print("❌ CUDA is not available at runtime")
    if not ultralytics_ok:
        print("❌ Ultralytics YOLO cannot be imported")

    if pytorch_ok and hasattr(torch, 'cuda') and torch.cuda.is_available() and ultralytics_ok:
        print("✅ Everything looks good for GPU-accelerated YOLO!")
    else:
        print("❌ Issues detected - GPU acceleration not available")

if __name__ == "__main__":
    main()