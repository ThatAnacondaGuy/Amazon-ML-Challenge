# Amazon ML Challenge — Business Entity Resolution

A classical machine-learning pipeline for matching business records across data sources. The implementation combines text normalization and transliteration, TF-IDF candidate retrieval, pairwise similarity features, LightGBM matching, and a second-stage model with collective features.

## Repository Structure

- `code/business_entity_resolution/` — pipeline source, dependencies, tests, and execution instructions.
- `design_probes/` — diagnostic and experimental scripts.
- `student_resource/utils/` — submission validation utility.
- `SOLUTION_PLAN.md` — solution design and experimental findings.
- `Documentation_template.md` — challenge documentation template.
- `README_data.md` — data documentation reference.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Run Tests

From the repository root:

```bash
cd code/business_entity_resolution
PYTHONPATH=src python tests/test_contracts.py
```

## Data Requirements

The challenge datasets are not included in this repository. Obtain them through the authorized challenge channel and follow the competition's data-use and redistribution rules.

The full pipeline expects a dataset directory containing `train/` and `test/` subdirectories. See `code/business_entity_resolution/README.md` for the staged execution commands and expected outputs.

## Reproducibility

The complete pipeline may require substantial time and memory. Validate generated submissions with the supplied challenge validator before relying on the results.

Reported experimental metrics in `SOLUTION_PLAN.md` are design and validation findings, not guaranteed performance on other machines or data splits.

## Data Safety and Reuse

Do not commit credentials, private evaluation data, virtual environments, or large intermediate artifacts.

No license is declared here. Confirm the competition rules and obtain the necessary permissions before reusing or redistributing code, datasets, or results.

