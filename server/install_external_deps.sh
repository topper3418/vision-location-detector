#!/bin/bash
set -euo pipefail

# install_global_deps.sh
# Installs system-level (non-venv) dependencies for vision-location-detector on Jetson Orin Nano
# JetPack 6.x compatible — focuses on cuSPARSELt fix + basic prep
# Run this BEFORE running install_deps.sh (the venv script)

echo "============================================================="
echo "  Installing GLOBAL (system-level) dependencies"
echo "  Jetson Orin Nano - JetPack 6.x / CUDA 12.6 compatible"
echo "============================================================="
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Update system packages
# ──────────────────────────────────────────────────────────────────────────────
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ──────────────────────────────────────────────────────────────────────────────
# 2. Install basic Python tools (if not already present)
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing python3-pip and venv support..."
sudo apt install -y python3-pip python3-venv

# Upgrade pip globally (optional but recommended)
pip install --user -U pip

# ──────────────────────────────────────────────────────────────────────────────
# 3. Install cuSPARSELt (Jetson/tegra-specific version)
#    This is the critical fix for libcusparseLt.so.0 missing error
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "Installing cuSPARSELt (tegra local installer for JetPack 6.x)..."

CUSPARSELT_VERSION="0.7.0"  # Update this if a newer version is available
DEB_FILE="cusparselt-local-tegra-repo-ubuntu2204-${CUSPARSELT_VERSION}_1.0-1_arm64.deb"

if [ ! -f "$DEB_FILE" ]; then
    echo "Downloading cuSPARSELt local installer..."
    wget "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${DEB_FILE}"
else
    echo "cuSPARSELt deb already downloaded, skipping wget."
fi

echo "Installing deb package..."
sudo dpkg -i "$DEB_FILE"

echo "Copying keyring and updating apt..."
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-${CUSPARSELT_VERSION}/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

echo "Installing libcusparselt packages..."
sudo apt-get install -y libcusparselt0 libcusparselt-dev

# ──────────────────────────────────────────────────────────────────────────────
# 4. Refresh library cache so torch can find libcusparseLt.so.0
# ──────────────────────────────────────────────────────────────────────────────
echo "Updating ldconfig..."
sudo ldconfig

# ──────────────────────────────────────────────────────────────────────────────
# 5. Verification steps
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "Verification:"

# Check if the library is now visible
LIB_PATH=$(find /usr -name libcusparseLt.so.0 2>/dev/null || true)
if [ -n "$LIB_PATH" ]; then
    echo "SUCCESS: libcusparseLt.so.0 found at: $LIB_PATH"
    ls -l "$LIB_PATH"
else
    echo "WARNING: libcusparseLt.so.0 NOT found after install!"
    echo "  → Double-check the NVIDIA cuSPARSELt download page for the latest version"
    echo "    https://developer.nvidia.com/cusparselt-downloads"
    echo "    (select: Linux → aarch64-jetson → Ubuntu 22.04 → deb local)"
fi

echo ""
echo "You can now safely run:"
echo "  ./install_deps.sh"
echo "  # which creates/activates venv and installs everything project-specific"
echo ""
echo "Done!"