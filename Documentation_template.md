# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A fully classical pipeline is used, with no embeddings, no pretrained models and no external data.
- **Normalization:** rule-based. Every Indic script is romanized using only Python's stdlib `unicodedata`, and a transliteration dictionary learned from training pairs refines the result.
- **Candidate generation:** each S2/S3 record looks up its best Source-1 entities in the same country, using a sparse TF-IDF inverted index with *conjunctive name × address keys*.
- **Matching:** about 45 string, address and house-number features feed a LightGBM matcher (MIT). A stacked second stage reasons about competing candidates and a business's other matched records.
- **Decisions:** each record goes to at most one Source-1 entity, and a threshold tuned for F0.5 is calibrated per country without labels.
- **Result:** final public leaderboard F0.5 = **0.981538**. The progression 0.97271 → 0.97563 → 0.97758 tracked validation gains; the last step (+0.0040) came from a diagnosed France decoy pattern.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Names don't identify businesses.** 45–54% of Source-1 entities share their normalized name with another entity (for example, 546 × "Meridian" in US train). Only 47% of true pairs have identical names. So name and address evidence must be combined, both in blocking and in matching.
- **Postcodes are nearly absent** (PIN present in ≤0.8% of records). House numbers, street and locality tokens, and cities are the address anchors.
- **Scripts:** 23% (S2) and 13% (S3) of India names are written in 9 Indic scripts, while Source 1 is 100% Latin.
- **Name noise:**
  - Legal-form changes and word-order changes.
  - Typos, including leetspeak (`5ervices`).
  - Inserted words (`Services`, `Center`, `Smt`, `M/s`).
  - Domain and handle names (`aleenhicksgranite.com`, `#KUMARBIOTECH`).
  - Pseudo-word trade names (`Ciraarcjax`) that only the address can link.
- **Address noise:**
  - Components reordered, abbreviations, `<NULL>` tokens.
  - Zero-padded, ranged or truncated house numbers.
  - State codes, full names and regional script used interchangeably.
  - 2.7–3.4% of addresses empty.
- **Structure:**
  - Every S2/S3 record matches **at most one** Source-1 entity (0 violations in 7.64M training pairs).
  - Entities have 3.46 matches on average (maximum 11), and 5.6% are singletons.
  - 26% of S2/S3 records are distractors.
  - The test set has 5.5–5.8 records per entity against 4.7 in train. Diagnostics showed the extra test records are distractors.
- **Distractors are "siblings":** perturbed copies of real businesses with one name word swapped or the house number nudged (`D-901` vs `D-904`). They caused 86% of false positives in early error analysis.
- **France** (test only):
  - Templated names: 46% share a name.
  - 33.5% of entities share an exact address with another entity, against about 7% in train. This co-location makes a model trained on US and India over-merge.
  - French abbreviations (`R.`, `AV`, `IMP`, `BIS`) and dotted legal forms (`S.A.S`).
- **Data quirks:** 813 test rows contain CSV-quoted fields (`"""ehpad Club SAS"`), and there is some mojibake and control characters.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (two-stage stacked GBDT) + constrained decision layer  
**Core Innovation:** conjunctive name × address retrieval keys (twin-proof blocking at about 9–12k records/s); a script-agnostic stdlib romanizer plus a transliteration dictionary learned from training pairs; **noise-operation features** that identify *how* a record differs from a candidate (typo vs whole-word substitution, house-number truncation vs nudge, invented trade name), separating the data generator's genuine noise from its decoy perturbations; stacked collective reasoning over a business's other matched records; label-free per-country threshold calibration.

---

## 3. Candidate Generation (Blocking)

**Representation.** Each record has two views, each L2-normalized and weighted 0.5/0.5:
- **Name view:** character 4-grams of the space-free core name, core tokens, phonetic-skeleton tokens, plus conjunctive keys `skeleton(name token) × house number` and `skeleton(name token) × locality token`.
- **Address view:** address tokens, their skeletons, house numbers, plus the conjunctive key `house number × street token`.

**Weighting and hashing.** TF-IDF is computed from the country partition's own Source 1. Features use 64-bit hashes, compacted to an exact vocabulary.

**Retrieval.**
- An inverted index over Source 1 excludes stop-features with document frequency above 2,000 from traversal.
- A numba parallel kernel returns the top 10 Source-1 entities per S2/S3 record (reverse direction, which matches the one-S1-per-record structure).
- A candidate is kept only if its score is at least 0.5 × the record's best score.

**Blocking keys used:** character 4-grams, name tokens, phonetic skeletons, address tokens, house numbers, and conjunctive name × number, name × locality and number × street keys. Postcode keys were not used (almost absent).

**Candidate pairs generated:** 23.2M on train (India 9.84M, US 13.32M) and 24.4M on test (France 4.83M, India 11.94M, US 7.66M). That is 2.2–3.4 per S2/S3 record, a reduction ratio of about 0.999997 against the within-country cross product.

**How true matches were not lost:**
- Candidate pair recall on train: **India 98.25%, US 98.57%**.
- Name-only retrieval reached only 51–64% recall@10 in design experiments, because of name twins. The conjunctive keys fixed this.
- Ratio pruning removes 62% of candidates but loses under 0.01% of true pairs.
- The remaining misses are mostly records with empty addresses whose names are shared by many entities, which cannot be resolved reliably.

---

## 4. Matching Model

**Features used** (country is never a feature):
- **Name features:**
  - rapidfuzz ratio, token-sort ratio, token-set ratio (core), Jaro-Winkler (core).
  - Ratio and partial ratio on the space-free core, which handles domains and concatenations.
  - Token and phonetic-skeleton Jaccard, TF-IDF name cosine, initials match.
  - Counts of extra and missing tokens.
  - Count of injected noise tokens (a lexicon learned without labels from tokens over-represented in S2/S3 relative to S1).
- **Address features:**
  - Token-set ratio, Jaccard, TF-IDF address cosine, exact address-key equality.
  - House numbers: any match, first-number match, prefix/suffix containment, log numeric distance, full compound-token match (`2-16-133-6`, `d904`), counts.
- **Other:**
  - Retrieval rank, score, ratio and gap to the record's best candidate; candidate count.
  - Name-twin count of the Source-1 entity; co-location features (how many Source-1 entities share the candidate's address, and whether the candidate is the best name match at that address); twin candidates per record.
  - Source (S2/S3), domain/handle flag, non-Latin flag.
  - **Noise-operation features (from error analysis):**
    - Unmatched-token alignment: typo / whole-vocabulary-word substitution / other substitution / short-initials mismatch; tokens only on one side that are vocabulary words.
    - House-number relation on the *non-shared* numbers: equal / truncation (prefix-suffix) / one-digit nudge / numerically close; extra numbers on either side.
    - Invented-name flag and out-of-vocabulary share (vocabulary = tokens used by ≥20 S1 names in the partition).
  - Stage-2 adds collective features from out-of-fold stage-1 scores: rank, margin to the runner-up entity, the entity's support from other records, and **sibling agreement** (similarity of the record to the entity's other confident matches).

**Model type:** LightGBM 4.7.0 (MIT license), binary objective, 127 leaves, learning rate 0.05, up to 1,500 trees, trained from scratch on the provided training data. Stage-2 is stacked on 5-fold out-of-fold stage-1 predictions, with folds grouped by Source-1 entity. There are far fewer than 8B parameters (<5M leaf values and thresholds).

**Threshold selection method:**
1. **Exclusivity:** each record is assigned only to its highest-scoring entity.
2. **Global threshold t\*:** maximizes the worst country's macro F0.5 on a 20% validation holdout grouped by Source-1 entity (t\* = 0.70).
3. **Per-country, label-free calibration:** within [t\* − 0.15, t\* + 0.25], each test country's threshold is chosen so its predicted matches per entity equal the validation reference (3.317). This is justified because match cardinality is identical across countries in train (3.462 vs 3.461).
4. **Chosen thresholds:** India 0.71, US 0.84, France 0.95. France over-merged at 0.70 (3.52 predicted matches per entity), consistent with its co-location.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** validation uses a 20% holdout grouped by Source-1 entity and the exact challenge formula, including singletons and blocking misses.

| Model | Validation India | Validation US | Public leaderboard |
|---|---|---|---|
| Stage-1 | 0.9774 | 0.9808 | 0.97271 |
| + stacked stage-2 (collective/sibling) | 0.9814 | 0.9835 | 0.97563 |
| + noise-operation features | 0.9836 | 0.9848 | 0.97758 |
| + group features, 2,500 trees (combined) + France template-swap filter (**final**) | **0.9839** | **0.9849** | **0.981538** |

  - A variant trained with a simulated higher distractor density (24% of training S1 dropped) scored *lower* on the leaderboard (~0.974), although it improved on a matching simulated validation. The simulation mislabelled genuine noisy copies as distractors, so it was discarded.
- **France template-swap filter (final step, `ber/postprocess.py`).** In the unlabelled France partition, error analysis of predicted pairs showed decoys that swap exactly one template word for another at the same address (`Kilti Amis` vs `Kilti Culture`). Genuine copies instead drop a word and/or add an injected noise suffix (`Et Fils`, `Services`, `& Associés`, `France`); these are protected by a label-free noise lexicon (tokens over-represented in S2/S3 vs S1 names, ratio ≥ 1.15). The filter removed 28,656 France matches (3.3%) and raised the leaderboard from 0.97758 to 0.98154; the final version also removes swaps whose house numbers differ (France randomizes numbers on genuine copies), 29,091 in total. A final France recall step (`ber/addback.py`) re-adds 1,893 records whose name and street/city words are identical to the S1 but whose house number differs, because France randomizes house numbers on genuine copies. It is applied only to France, because in the labelled countries such swaps are 98–99% true matches.
- **Common false positives (wrong merges):**
  - "Sibling" distractors: the same street with the house number nudged (`D-901`/`D-904`, `2-16-133/2`/`/6`), or one distinctive name word swapped (`Superior Infra` vs `Superior Kumar`).
  - Records with empty addresses matched to one of many same-name entities.
  - In France, separate businesses registered at the same address.
- **Common false negatives (missed matches):**
  - Heavy benign noise: inserted generic words (`Services`, `Center`), honorifics, leetspeak, truncated house numbers (`19-A-89` vs `919-A-89`).
  - Pseudo-word trade names with partial addresses.
  - Records with empty addresses (a blocking ceiling of about 1.5% of pairs).

---

## 6. Conclusion
- Two ingredients made classical entity resolution work at 10M-record scale on a laptop: combining name and address evidence at the *retrieval* stage (conjunctive keys), and decisions that exploit the data's structure (one entity per record, country-invariant cardinality).
- The main lesson: textbook remedies for domain shift (dropping features, monotone constraints, self-training) measured *negative*. Diagnosing the concrete mechanism (co-location in France) and adding targeted, label-free features and calibration did not.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/ber/` and `tests/`. To reproduce, from that folder:
1. `PYTHONPATH=src python -m ber.pipeline --data-dir <dataset> --work-dir work --out-dir <output>` builds candidates and features for train and test, trains stage-1, validates, and writes the submission.
2. `PYTHONPATH=src python -m ber.stage2 ...` produces the stacked final model and outputs.

`README.md` has the exact commands, and `requirements.txt` pins versions (Python 3.12). A full run takes about 68 min for stage-1 and about 90 min for stage-2 on a 10-core, 24 GB laptop; peak RSS was about 10 GB.

**External data and compliance:** only the provided files are used. There are no APIs, gazetteers or pretrained weights. Unlabelled test inputs are used only for unsupervised statistics (IDF, twin counts, the noise lexicon) and for label-free threshold calibration. No model is trained on test data.

### B. Additional Results
Validation F0.5 by threshold (stage-1):
- **India:** 0.40: 0.9731 · 0.50: 0.9756 · 0.60: 0.9770 · 0.70: 0.9774 · 0.80: 0.9765 · 0.90: 0.9724
- **US:** 0.40: 0.9779 · 0.50: 0.9797 · 0.60: 0.9805 · 0.70: 0.9808 · 0.80: 0.9802 · 0.90: 0.9778

Design experiments (leave-one-country-out transfer, blocking recall and cost trade-offs, ablations) are documented in `SOLUTION_PLAN.md`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
