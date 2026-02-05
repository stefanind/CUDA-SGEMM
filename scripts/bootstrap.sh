#!/usr/bin/env bash

# -e exit on error
# -u error on undefined variables
# -o pipefall fail if any piped command fails
set -euo pipefail

apt-get update
apt-get install -y build-essential cmake git

nvcc --version
