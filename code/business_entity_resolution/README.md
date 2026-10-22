# Business Entity Resolution — classical pipeline

Resolves Source-2/Source-3 business records to the deduplicated Source-1 entities using only classical methods:
rule-based normalization (stdlib `unicodedata` romanization for all Indic scripts), sparse TF-IDF candidate
retrieval with conjunctive name×address keys, string/address similarity features, and LightGBM (MIT) matchers.
No external data, no APIs, no embeddings, no pretrained models.

## Environment

Python 3.12 (tested 3.12.13 on macOS arm64, 10 cores, 24 GB RAM).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Reproduce (data → blocking → matching → output)

Run from this directory; `--data-dir` points at the challenge `dataset/` folder (with `train/` and `test/`).

```bash
# 1. stage-1 pipeline: learns resources, generates candidates + features for train and test, trains the stage-1
#    model, validates, and writes a first submission (also caches everything under work/)
PYTHONPATH=src python -m ber.pipeline --data-dir ../../student_resource/dataset --work-dir work --out-dir ../../output

# 2. noise-operation features for every cached train/test pair (typo vs word-substitution, number truncation vs
#    nudge, invented-name flag) -> work/<split>_<country>_aug.parquet
PYTHONPATH=src python -m ber.augment --data-dir ../../student_resource/dataset --work-dir work

# 3. final model: out-of-fold stage-1 (5 folds by Source-1 entity) + collective/sibling features + stage-2 model,
#    validation report (work/c3_validation_stage2.json) and the submission files
PYTHONPATH=src python -m ber.stage2 --data-dir ../../student_resource/dataset --work-dir work --out-dir ../../output --aug --tag c3_

# 3b. (final submission) combined model = step 3 with --group --tag cg_ --rounds 2500, then remove pure
#     template-swap matches in the unlabelled France partition (see src/ber/postprocess.py for the rationale)
PYTHONPATH=src python -m ber.stage2 --data-dir ../../student_resource/dataset --work-dir work --out-dir ../../output_combined --aug --group --tag cg_ --rounds 2500
PYTHONPATH=src python -m ber.postprocess --data-dir ../../student_resource/dataset --work-dir work --in-dir ../../output_combined --out-dir ../../output_swap2 --partitions France --any-number
PYTHONPATH=src python -m ber.addback --data-dir ../../student_resource/dataset --work-dir work --in-dir ../../output_swap2 --out-dir ../../output --partition France --tag cg_

# 4. validate the submission format
python3 ../../student_resource/utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../student_resource/dataset/test --check-ids

# unit tests
PYTHONPATH=src python tests/test_contracts.py
```

Outputs: `output/matching_results.tsv` (scored file) and `output/candidate_pairs.tsv` (the exact pruned candidate
set that the matching models score). Logs, validation metrics (`validation*.json`) and models are in `work/`.

## Layout

| Module | Role |
|---|---|
| `ber/io.py` | ingest (raw tab split + RFC-4180 unquote, schema checks), submission writers |
| `ber/text.py` | cleaning, script-agnostic romanization, name/address parsing, phonetic skeleton |
| `ber/learn.py` | transliteration dictionary learned from training pairs |
| `ber/blocking.py` | per-country TF-IDF inverted index (64-bit hashed features, df cap), numba top-k, ratio pruning |
| `ber/features.py` | ~45 country-agnostic pairwise features (incl. house-number, co-location, noise-token features) |
| `ber/decide.py` | exact macro F0.5, argmax exclusivity, cardinality-matched thresholds |
| `ber/pipeline.py` | stage-1 end-to-end driver |
| `ber/augment.py` | noise-operation features: typo vs whole-word substitution, house-number truncation vs nudge, invented names |
| `ber/stage2.py` | out-of-fold stacking with collective and sibling-agreement features (final model) |

Country is used only as a partition key (open set; no country-specific code paths). Runtime and memory figures
are logged in `work/run.log` and `work/stage2.log`.
