#!/usr/bin/env bash
# git push that survives a bad connection: retries every 20 s until it succeeds.
set -u
cd "$(dirname "$0")/.."
n=0
until git push "${@:-origin HEAD}"; do
  n=$((n+1)); echo "[$(date +%H:%M:%S)] push failed, retry $n in 20 s"; sleep 20
done
echo "pushed."
