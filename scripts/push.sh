#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
n=0
until git push -u origin HEAD; do
  n=$((n+1)); echo "[$(date +%H:%M:%S)] push failed, retry $n in 20 s"; sleep 20
done
echo "pushed."
