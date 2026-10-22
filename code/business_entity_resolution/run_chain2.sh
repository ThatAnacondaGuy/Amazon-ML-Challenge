#!/usr/bin/env bash
# After version 2: cycle 3 (noise-operation features, normal data) then combined (shift + features + more trees).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python; DATA=../../student_resource/dataset; export PYTHONPATH=src
echo "[chain2] augment work_v2 train $(date +%H:%M)"
$PY -m ber.augment --data-dir $DATA --work-dir work_v2 > augment_v2.log 2>&1 || { echo "[chain2] augment v2 FAILED"; exit 1; }
echo "[chain2] waiting for version 2 chain $(date +%H:%M)"
while pgrep -f run_chain.sh >/dev/null || pgrep -f "ber.stage2" >/dev/null; do sleep 20; done
echo "[chain2] cycle 3 (aug, normal data) $(date +%H:%M)"
mkdir -p ../../output_c3
$PY -m ber.stage2 --data-dir $DATA --work-dir work --out-dir ../../output_c3 --aug --tag c3_ > chain_c3.log 2>&1 || echo "[chain2] cycle 3 FAILED"
echo "[chain2] cycle 3 done $(date +%H:%M)"
echo "[chain2] combined (shift + aug + more trees) $(date +%H:%M)"
mkdir -p ../../output_combined
$PY -m ber.stage2 --data-dir $DATA --work-dir work_v2 --out-dir ../../output_combined --aug --tag c3_ --rounds 2500 > chain_combined.log 2>&1 || echo "[chain2] combined FAILED"
echo "[chain2] combined done $(date +%H:%M)"
