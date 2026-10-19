#!/usr/bin/env bash
# Runs every test file. Each run ends GREEN (exit 0), SKIP (exit 77: the weights or the reference
# clone are not on disk) or RED; the script exits 1 if anything is RED, 0 otherwise.
#   scripts/run_tests.sh              1  tests/test_identity.py on tiny random models (offline, ~10 s)
#                                     2  tests/test_identity.py on the real Qwen3-4B base   (SKIP if not on disk)
#                                     3  tests/test_release_adapter.py                     (SKIP if weights absent)
#                                     4  tests/test_crosscheck.py                          (SKIP if weights or the reference clone absent)
#   scripts/run_tests.sh <model>      run 2 on <model> instead (an mlx-lm path or HF repo, as before)
# DELTA_MEM_BASE, DELTA_MEM_ADAPTER, DELTA_MEM_REF override the weight paths; PY overrides the interpreter
# (default: the repo venv, .venv/bin/python, resolved to an absolute path before the cd below).
# The cross-check writes results/crosscheck.latest.json and compares it with the committed results/crosscheck.json,
# which it never rewrites.
set -u
PY="${PY:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY — create the venv (see README) or set PY=/path/to/python"; exit 1; }
cd "$(dirname "$0")/.."
mkdir -p logs
BASE=${DELTA_MEM_BASE:-$HOME/mlx-models/Qwen3-4B-Instruct-2507-4bit}
ADAPTER=${DELTA_MEM_ADAPTER:-$HOME/mlx-models/delta-mem-qwen3-4b-adapter}
REF=${DELTA_MEM_REF:-$HOME/ror/_ref/delta-mem-mlx-sidecar}
MODEL=${1:-}

GREEN=$'\033[32mGREEN\033[0m'; RED=$'\033[31mRED\033[0m'; SKIP=$'\033[33mSKIP\033[0m'
summary=(); red=0

run_one() {   # run_one <name> <log file> <command...>
  local name=$1 log=$2; shift 2
  echo; echo "### $name"
  "$@" 2>&1 | tee "logs/$log"
  local rc=${PIPESTATUS[0]}
  case $rc in
    0)  summary+=("$GREEN  $name");;
    77) summary+=("$SKIP   $name");;
    *)  summary+=("$RED    $name (exit $rc)"); red=1;;
  esac
}

run_one "identity, tiny random models" test_tiny.log "$PY" tests/test_identity.py
if [ -n "$MODEL" ]; then
  run_one "identity, $MODEL" test_real.log "$PY" tests/test_identity.py --model "$MODEL"
elif [ -f "$BASE/config.json" ]; then
  run_one "identity, $(basename "$BASE")" test_real.log "$PY" tests/test_identity.py --model "$BASE"
else
  run_one "identity, real model" test_real.log sh -c "echo 'SKIP: no model given and $BASE is not on disk'; exit 77"
fi
run_one "release adapter" test_release_adapter.log "$PY" tests/test_release_adapter.py --base "$BASE" --adapter "$ADAPTER"
run_one "cross-check vs reference port" test_crosscheck.log "$PY" tests/test_crosscheck.py \
  --base "$BASE" --adapter "$ADAPTER" --ref "$REF"

echo; echo "### summary"
printf '  %s\n' "${summary[@]}"
exit $red
