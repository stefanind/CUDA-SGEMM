#!/usr/bin/env bash
set -euo pipefail

# -p if dir exists, do nothing; if parent dir missing, create it
mkdir -p build

# -S . == source directory
# -B build tells build/ to contain the makefiles
# Release is mandatory for CUDA benchmarking
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# use the build system CMake generated in build/ and compile
# -j triggers a parallel build 
cmake --build build -j"$(nproc)"


# sanity info
command -v nvidia-smi >/dev/null && nvidia-smi || true
command -v nvcc >/dev/null && nvcc --version || true