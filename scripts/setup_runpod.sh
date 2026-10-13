#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$workspace/.venv"
export PIP_CACHE_DIR="$workspace/.cache/pip"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

mkdir -p "$workspace" "$PIP_CACHE_DIR"
if [[ ! -f "$venv/bin/activate" ]]; then
  python -m venv --system-site-packages "$venv"
fi

source "$venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install "einops==0.8.1" "transformers==4.48.2"

# Compile against the CUDA-enabled PyTorch supplied by the RunPod image.
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
  --force-reinstall --no-deps \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.2.post1" \
  --no-build-isolation
python -m pip install "mamba-ssm==2.2.5" --no-build-isolation
python "$workspace/scripts/verify_mamba.py"

cat > "$workspace/ACTIVATE_PROJECT.txt" <<EOF
Run: source $venv/bin/activate
Project: $workspace
EOF
