#!/usr/bin/env bash
set -euo pipefail

# install_deps.sh
# Creates venv and installs all dependencies for vision-location-detector on Jetson Orin Nano
# Assumes you are in the project root directory (~/vision-location-detector)
# Assumes requirements.txt already contains:
#   numpy<2
#   ultralytics
#   torch @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
#   torchvision @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
#   onnxruntime-gpu @ https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
#   ... other project dependencies

echo "============================================================="
echo "  Installing dependencies for vision-location-detector"
echo "  Jetson Orin Nano - JetPack 6.x compatible"
echo "============================================================="

# ──────────────────────────────────────────────────────────────────────────────
# 1. Clean up old venv if it exists
# ──────────────────────────────────────────────────────────────────────────────
if [ -d "venv" ]; then
    echo "Removing existing venv..."
    rm -rf venv
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2. Create fresh venv with system-site-packages
# ──────────────────────────────────────────────────────────────────────────────
echo "Creating new virtual environment..."
python3 -m venv venv --system-site-packages

# ──────────────────────────────────────────────────────────────────────────────
# 3. Activate venv
# ──────────────────────────────────────────────────────────────────────────────
source venv/bin/activate

# ──────────────────────────────────────────────────────────────────────────────
# 4. Upgrade pip & friends inside venv
# ──────────────────────────────────────────────────────────────────────────────
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ──────────────────────────────────────────────────────────────────────────────
# 5. Install project dependencies (including pinned numpy<2 & wheels)
#    This should respect numpy<2 from requirements.txt
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# 6. Force-reinstall numpy<2 (extra safety in case resolver ignored pin)
# ──────────────────────────────────────────────────────────────────────────────
echo "Ensuring NumPy stays <2 (critical for Torch wheel compatibility)..."
pip install "numpy==1.26.4" --force-reinstall --no-deps

# ──────────────────────────────────────────────────────────────────────────────
# 7. Quick verification
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installations:"

python -c "import numpy; print('NumPy version:     ', numpy.__version__)"
python -c "import torch; print('Torch version:     ', torch.__version__); print('CUDA available:    ', torch.cuda.is_available())"
python -c "import torchvision; print('Torchvision:       ', torchvision.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:      ', onnxruntime.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics:       OK')"

# Critical test: torch.from_numpy must work without "Numpy is not available"
python -c "
import numpy as np
import torch
a = np.zeros((1,))
t = torch.from_numpy(a)
print('torch.from_numpy:  Success')
"

echo ""
echo "If all checks above show reasonable versions and 'Success', setup is complete."
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  ./run.sh"
echo ""
echo "To export a model to TensorRT (first time may take several minutes):"
echo "  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, imgsz=640, workspace=4)\""
echo ""
echo "Done!"