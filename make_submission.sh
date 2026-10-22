#!/usr/bin/env bash
# Build <team>_submission.zip with an explicit allow-list (no venv, caches, probes or data).
# Usage: ./make_submission.sh <team_name> [output_dir]
set -euo pipefail
TEAM="${1:?team name required}"
OUT="${2:-output}"
cd "$(dirname "$0")"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$STAGE/output/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  code/business_entity_resolution/src code/business_entity_resolution/tests \
  code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt \
  "$STAGE/code/business_entity_resolution/"
cp Documentation_template.md "$STAGE/"
rm -f "${TEAM}_submission.zip"
(cd "$STAGE" && zip -qr - .) > "${TEAM}_submission.zip"
rm -rf "$STAGE"
unzip -l "${TEAM}_submission.zip" | tail -n +1 | awk 'NR<=40'
