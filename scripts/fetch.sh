#!/usr/bin/env bash
# Resumable downloads for delta-mem-mlx. Safe to re-run: huggingface-cli resumes partial
# files, and this loop retries until each download succeeds (Ctrl-C to stop).
# Everything lands under ./weights/ (git-ignored). Nothing here is needed for the tests.
set -u
cd "$(dirname "$0")/.."
mkdir -p weights
export HF_HUB_ENABLE_HF_TRANSFER=0          # plain HTTP resumes more reliably on flaky wifi

HF=$(command -v hf || command -v huggingface-cli || { echo "pip install -U huggingface_hub first"; exit 1; })

fetch () {  # fetch <repo> <local-dir>
  local repo="$1" dir="$2" n=0
  until "$HF" download "$repo" --local-dir "$dir" >/dev/null 2>&1; do
    n=$((n+1)); echo "[$(date +%H:%M:%S)] $repo: retry $n in 15 s"; sleep 15
  done
  echo "[$(date +%H:%M:%S)] $repo: complete → $dir"
}

# 1. the community MLX conversion of the authors' Qwen3-4B sidecar (small, ~20 MB)
fetch ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter weights/delta-mem-qwen3-4b-instruct-mlx-adapter

# 2. the matching backbone, 4-bit (~2.3 GB). Skip with:  SKIP_BASE=1 scripts/fetch.sh
if [ -z "${SKIP_BASE:-}" ]; then
  fetch mlx-community/Qwen3-4B-Instruct-2507-4bit weights/Qwen3-4B-Instruct-2507-4bit
fi
echo "done. tests need none of this: python tests/test_identity.py"
