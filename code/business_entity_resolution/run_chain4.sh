#!/usr/bin/env bash
# NO shift (it hurt the leaderboard). After cycle 3: group-level (reuse cycle-3 stage-1) -> combined on NORMAL data.
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python; DATA=../../student_resource/dataset; export PYTHONPATH=src
echo "[chain4] waiting for cycle 3 $(date +%H:%M)"
while pgrep -f "ber.stage2" >/dev/null; do sleep 20; done
echo "[chain4] group-level run (reuses cycle-3 stage-1) $(date +%H:%M)"
mkdir -p ../../output_group
$PY -m ber.stage2 --data-dir $DATA --work-dir work --out-dir ../../output_group --aug --group --tag g3_ --s1-tag c3_ > chain_group.log 2>&1 || echo "[chain4] group FAILED"
echo "[chain4] group done $(date +%H:%M)"
echo "[chain4] combined NORMAL data (noise-op + group + more trees) $(date +%H:%M)"
mkdir -p ../../output_combined
$PY -m ber.stage2 --data-dir $DATA --work-dir work --out-dir ../../output_combined --aug --group --tag cg_ --rounds 2500 > chain_combined.log 2>&1 || echo "[chain4] combined FAILED"
echo "[chain4] combined done $(date +%H:%M)"
