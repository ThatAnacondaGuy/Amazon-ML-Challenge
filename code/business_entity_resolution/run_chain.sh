#!/usr/bin/env bash
# Sequential chain (memory-safe): finish stage-2 -> cache test -> stage-2 outputs -> version 2 (shift + more trees).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
DATA=../../student_resource/dataset
export PYTHONPATH=src
echo "[chain] waiting for running stage-2 to exit $(date +%H:%M)"
while pgrep -f "ber.stage2" >/dev/null; do sleep 15; done
echo "[chain] 1/4 caching test partitions $(date +%H:%M)"
$PY -m ber.pipeline --data-dir $DATA --work-dir work --out-dir ../../output --test-only > chain_test.log 2>&1 || { echo "[chain] test caching FAILED"; exit 1; }
echo "[chain] 2/4 stage-2 outputs (reuses saved models) $(date +%H:%M)"
$PY -m ber.stage2 --data-dir $DATA --work-dir work --out-dir ../../output_stage2 > chain_stage2.log 2>&1 || { echo "[chain] stage-2 FAILED"; exit 1; }
echo "[chain] stage-2 DONE $(date +%H:%M)"
echo "[chain] 3/4 version 2 train partitions (shift) $(date +%H:%M)"
mkdir -p work_v2 ../../output_v2
cp work/translit_dict.json work_v2/
for f in work/test_*; do ln -sf "$(pwd)/$f" work_v2/; done
$PY -m ber.pipeline --data-dir $DATA --work-dir work_v2 --out-dir ../../output_v2 --shift --train-only > chain_v2_train.log 2>&1 || { echo "[chain] v2 train FAILED"; exit 1; }
echo "[chain] 4/4 version 2 stage-2 (more trees) $(date +%H:%M)"
$PY -m ber.stage2 --data-dir $DATA --work-dir work_v2 --out-dir ../../output_v2 --rounds 2500 > chain_v2_stage2.log 2>&1 || { echo "[chain] v2 stage-2 FAILED"; exit 1; }
echo "[chain] version 2 DONE $(date +%H:%M)"
