#!/usr/bin/env bash
# Runs the suite twice: tiny random models (offline), then a real model if one is given.
#   scripts/run_tests.sh                                   # offline
#   scripts/run_tests.sh mlx-community/Qwen2.5-7B-Instruct-4bit
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
python tests/test_identity.py 2>&1 | tee logs/test_tiny.log || exit 1
if [ -n "${1:-}" ]; then
  python tests/test_identity.py --model "$1" 2>&1 | tee logs/test_real.log
fi
