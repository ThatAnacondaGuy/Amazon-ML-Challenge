#!/usr/bin/env bash
# After cycle 3: group-level test (reuses cycle-3 stage-1) -> combined (shift + noise-op + group + more trees).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python; DATA=../../student_resource/dataset; export PYTHONPATH=src
echo "[chain3] waiting for cycle 3 $(date +%H:%M)"
while pgrep -f "ber.stage2" >/dev/null; do sleep 20; done
echo "[chain3] group-level run (reuses cycle-3 stage-1) $(date +%H:%M)"
mkdir -p ../../output_group
$PY -m ber.stage2 --data-dir $DATA --work-dir work --out-dir ../../output_group --aug --group --tag g3_ --s1-tag c3_ > chain_group.log 2>&1 || echo "[chain3] group FAILED"
echo "[chain3] group done $(date +%H:%M)"
echo "[chain3] combined (shift + noise-op + group + more trees) $(date +%H:%M)"
mkdir -p ../../output_combined
$PY -m ber.stage2 --data-dir $DATA --work-dir work_v2 --out-dir ../../output_combined --aug --group --tag cg_ --rounds 2500 > chain_combined.log 2>&1 || echo "[chain3] combined FAILED"
echo "[chain3] combined done $(date +%H:%M)"
