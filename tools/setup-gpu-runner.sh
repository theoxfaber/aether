#!/usr/bin/env bash
# Setup a GitHub Actions self-hosted runner with GPU support for Aether.
#
# Usage:
#   1. Add a self-hosted runner in your repo:
#      Settings → Actions → Runners → New runner
#   2. Run this script on the GPU machine BEFORE configuring the runner.
#   3. Configure the runner with the token from GitHub.
#   4. Label the runner with "gpu" so the gpu-tests CI job picks it up.
#
# Requires: sudo, an NVIDIA GPU with compute capability ≥ 5.0 (for WGPU/Vulkan).
# Tested on: Ubuntu 22.04 LTS

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Aether GPU CI Runner Setup ==="
echo ""

# ── 1. System packages ──────────────────────────────────────────────────
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    wget \
    vulkan-tools \
    mesa-vulkan-drivers \
    linux-headers-$(uname -r)

# ── 2. Vulkan SDK (for lavapipe software fallback) ─────────────────────
echo "[2/4] Installing Vulkan SDK..."
if ! command -v vulkaninfo &>/dev/null; then
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc > /dev/null
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
        https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq vulkan-sdk
fi

# ── 3. NVIDIA driver + CUDA (optional, for native GPU acceleration) ────
echo "[3/4] Checking NVIDIA GPU..."
if lspci | grep -i nvidia > /dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing driver + CUDA toolkit..."
    sudo apt-get install -y -qq nvidia-driver-550 nvidia-utils-550
    # CUDA via runfile or apt — we use apt for simplicity
    if ! command -v nvcc &>/dev/null; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y -qq cuda-toolkit-12-4
        rm -f cuda-keyring_1.1-1_all.deb
    fi
    echo "CUDA $(nvcc --version | grep release | cut -d, -f2) installed"
else
    echo "No NVIDIA GPU detected. GPU tests will use software rendering (lavapipe)."
fi

# ── 4. Verify ────────────────────────────────────────────────────────────
echo "[4/4] Verifying setup..."
vulkaninfo --summary 2>&1 | grep -E "GPU|deviceName|driverInfo" || echo "(no Vulkan GPU info available — sw rendering may be used)"
echo ""
echo "=== GPU Runner Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. cd ~/actions-runner  (or wherever you extracted the runner)"
echo "  2. ./config.sh --url https://github.com/YOUR_ORG/aether --token YOUR_TOKEN --labels gpu"
echo "  3. ./run.sh"
echo ""
echo "The runner will pick up the 'gpu-tests' CI job."
