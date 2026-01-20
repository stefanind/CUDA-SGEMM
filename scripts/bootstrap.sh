#!/usr/bin/env bash

# -e exit on error
# -u error on undefined variables
# -o pipefall fail if any piped command fails
set -euo pipefail

# no CUDA install or NVIDIA drivers on purpose
sudo apt-get update
sudo apt-get install -y build-essential cmake

# some checks after running 
echo "NOTE: You may still need NVIDIA driver + CUDA toolkit installed."
echo "Check: nvidia-smi && nvcc --version"
