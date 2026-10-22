# Business Entity Resolution — Design Review & Solution Plan

**Status:** Phases 0–3 are complete. This document is the Phase-3 deliverable and is **awaiting approval**. No production pipeline code has been written yet.
**Date:** 2026-09-26 · **Machine:** Apple M5 (10 cores), 24 GB RAM · **Environment:** project venv, Python 3.12.13

> **About the numbers.** Every number here was measured on the real data using throwaway *design probes* (small scripts, not the pipeline). They ran on real-scale indexes, meaning full-country S1 indexes with millions of queries. Appendix A lists each probe and its result. Where a number is an estimate rather than a measurement, the text says so.

---

## 0. Executive summary

**What we will build.** A classical, fully offline entity-resolution pipeline. It uses no embeddings and no pretrained models.

1. **Robust ingest.** Split each row on tabs, then apply RFC-4180 unquoting. This handles the 813 CSV-quoted rows in test.
2. **Rule-based normalization.** It is script-agnostic:
   - A romanizer for all 9 Indic scripts, driven only by stdlib `unicodedata` character names.
   - A transliteration dictionary learned from training pairs.
   - An abbreviation learner that works on any country's *unlabelled* text with no labels.
3. **Reverse-direction blocking.** For every S2/S3 record, retrieve its top-k S1 entities within the same country. Retrieval uses a df-capped sparse TF-IDF inverted index over name, address and **conjunctive name×address keys**. A numba kernel runs it.
4. **About 50 country-agnostic string, address, retrieval-context and ambiguity features**, scored by a **LightGBM** (MIT) stage-1 model. A **stage-2** model then adds collective features built from out-of-fold stage-1 scores.
5. **Precision-first decisions.** Each S2/S3 record may join at most one S1. The ground truth shows 0 violations of that rule in 7.64M pairs. On top of that we apply a threshold chosen to maximize macro F0.5 under test-like conditions, then **calibrated per country without labels** by matching the predicted matches per entity. The generator's cardinality is measured to be country-invariant, which justifies this.

**Key measured evidence (train, probes):**

| Quantity | US | India |
|---|---|---|
| Blocking pair recall @5 (full country index) | **98.22%** | 96.95% → **97.79%** with the learned dictionary |
| Blocking recall @10, df-cap 2k, conjunctive keys | 97.5% | 97.6% (4-grams) |
| Retrieval throughput (10 cores) | ~12k queries/s | ~10k queries/s |
| First-cut matcher, macro F0.5 (28 features, 1 stage) | **0.9737** | 0.9530 → **0.9630** with the learned dictionary |
| + stage-2 collective features (out-of-fold stacking) | — | **0.9663** (+0.39 over OOF stage-1 0.9624; with dictionary) |
| Oracle F0.5 with a perfect matcher on our candidates (blocking ceiling) | 0.9946 | 0.9893 → 0.9929 with the dictionary |
| Unseen-country transfer (trained on the other country only) | India→US **0.9563** with the label-free threshold (in-domain 0.9740) | US→India 0.8507 → 0.8992 with the dictionary → **0.9131** with the label-free threshold (in-domain 0.9632) |

**Honest test estimate:** macro F0.5 of about **0.96 (plausible range 0.953–0.969)**. The spread is driven almost entirely by France (§4.9), which has no labels.

**Biggest risks.**
1. **Transfer to France.** The mechanism is diagnosed: **33.5% of France S1 share an exact address with another business, against about 7% in train.** That makes a US/India-trained model over-merge (predicted 3.55 matches per S1 vs about 3.3 expected). The mitigations are all label-free and all measured: co-location features, the dictionary, and a cardinality-matched threshold (§4.8).
2. Test has a higher ratio of records to S1 entities (5.5–5.8 against 4.67 in train), which probably means more distractors. We simulate this in validation.

**Runtime and memory.** A full train+test run takes about 1.5–2 h on this laptop. Test-only inference with trained models takes about 45 min. Peak memory target is 12 GB or less (§4.10).

---

## 1. Phase 0 — Ground truth on the problem

### 1.1 What was read

Everything was read in full:
- `6ab5628d5a817_amazon_ml_challenge_problem_statement.md`
- `student_resource/README.md` (same spec with cleaner formatting)
- `Documentation_template.md`
- `utils/validate_submission.py` (all 350 lines)

The data was then loaded and profiled directly (§1.3). The raw bytes were scanned before any parser was trusted.

### 1.2 Constraint & scoring checklist

| # | Rule | Source | How the design satisfies it (by construction) |
|---|---|---|---|
| C1 | `matching_results.tsv` header is exactly `source1_entity_id⇥matched_entity_ids` | README "Output Format"; validator `MATCHING_HEADER` | The writer uses constants, and the validator runs as a gate |
| C2 | Tab-separated UTF-8 with no quoting. IDs are comma-joined with no spaces. **Lines must end in LF only**: the validator strips only `\n`, so a stray `\r` would corrupt the last ID | README; validator L132–146 | A custom writer (not the `csv` module) writes `"\n"` explicitly. A unit test asserts no `\r` and no `"` |
| C3 | Exactly one row per test S1 entity (1,732,544), with no extras and no duplicate rows | README "Important"; validator L191–197 | Output is left-joined from the loaded S1 table. We assert row count equals S1 count and that IDs are unique |
| C4 | Matched IDs are S2-/S3- only, exist in test, have no duplicates within a list, and the list is empty for singletons. No self-matches | README; validator L151–161 | IDs come only from the loaded test S2/S3 tables. Each list is set-deduplicated and sorted. The validator runs with `--check-ids` (it is **off by default**, which is easy to miss) |
| C5 | `candidate_pairs.tsv` is the **last-stage candidate set the model actually scores**, not an early blocking pass. It follows the same format rules, and matches must be a subset of candidates | README "candidate_pairs.tsv" | It is written from the exact pruned pair table that stage-1 scores. Stage-2 re-scores the same set and never adds pairs. The subset relation is asserted in code and warned by the validator |
| C6 | `country` is an open set. Do not hard-code, filter or one-hot to {US, India}. **France appears only in test** and every France S1 must be output | README "Data Description" | Country is used only as a partition key. It is **never a feature**, and there is **no branching on country names**. Each partition is processed by the same code |
| C7 | The final model must be MIT or Apache-2.0 licensed with at most 8B parameters | README Constraint 5 | LightGBM **4.7.0 (MIT)**. At most a few million leaf values and thresholds (<5M; §4.6), more than 3 orders of magnitude below the cap. No pretrained weights exist anywhere in the pipeline |
| C8 | No external databases, APIs, lookups, geocoding or internet data. Violation means disqualification | "Academic Integrity" | Inputs are only the provided files. No network access at runtime. Learned resources come only from provided data (§4.11) |
| C9 | The zip contains `output/` (both TSVs), `code/business_entity_resolution/{src/, README.md, requirements.txt}` and a filled `Documentation_template.md`. The code must regenerate both outputs from the data | "Final Submission Package" | §4.13 mapping. A clean-room rerun is the final checklist step |
| C10 | Metric: F0.5 = 1.25·P·R/(0.25·P+R) per S1, **macro-averaged over all S1**. A singleton scores 1 if predicted empty, else 0. A non-singleton predicted empty scores 0 | "Evaluation Criteria" | The eval module reimplements this exactly and is unit-tested on the README example (0.714). All thresholds optimize this metric |
| C11 | The public leaderboard is a subset of test; the private leaderboard (the rest) decides the ranking | "Leaderboard" | No tuning on public-leaderboard feedback. Internal validation drives every decision |
| C12 | **Brief constraint:** classical methods only. No embeddings (we also exclude dense LSA/SVD vectors), no transformers, no pretrained neural networks, in any stage. The matcher is a GBDT | User brief | Sparse TF-IDF, hashing, string metrics, rule-based phonetics and transliteration, LightGBM |

**Easy-to-miss items found in the data and the validator** (not stated in the docs):

| # | Finding | Evidence | Response |
|---|---|---|---|
| E1 | 813 test rows (10 in train) contain CSV-quoted fields with doubled quotes, e.g. `"""ehpad Club SAS"` → `"ehpad Club SAS` | raw byte scan | Split raw on tabs (every row has exactly 4 fields), then unquote each field |
| E2 | Control characters and mojibake: `\x1a` (a corrupted apostrophe or dash) and `Â\x80\x93` (a double-encoded en-dash), in about 730 rows | raw scan | A cleaning map runs before normalization |
| E3 | Literal `<NULL>`, `NULL` and `N/A` appear *inside* addresses | samples | Null-token removal |
| E4 | By default the validator does **not** check that IDs exist | validator L30–39 | Always run with `--check-ids` before submitting |
| E5 | Every S2/S3 record belongs to **at most one** S1 (0 violations in 7,638,365 pairs) | ground-truth profile | The decision layer enforces exclusivity |
| E6 | Test has **5.53–5.82** S2/S3 records per S1 in every country, against **4.67** in train | profile | A distractor-shift simulation in validation (§4.9) |

### 1.3 Data profile (measured)

**Files**

| File | Rows | Countries | Empty address | Non-Latin names |
|---|---|---|---|---|
| train S1 | 2,206,821 | US 1,323,633 · India 883,188 | 0% | 0% |
| train S2 | 5,034,616 | US 3,016,817 · India 2,017,799 | 3.36% | India 23.2% |
| train S3 | 5,285,603 | US 3,170,056 · India 2,115,547 | 3.33% | India 12.6% |
| test S1 | 1,732,544 | India 809,986 · US 663,106 · **France 259,452** | 0% | 0% |
| test S2 | 4,887,273 | India 2,312,565 · US 1,871,330 · France 703,378 | 2.65% | India 23.5% |
| test S3 | 5,082,316 | India 2,405,000 · US 1,945,701 · France 731,615 | 2.68% | India 12.8% |

- IDs are unique and prefixes are correct in every file. Every row has exactly 4 fields. There are no CRs and no invalid UTF-8.
- **Scripts in India S2/S3 names:** Devanagari 12.8% (S2), then Kannada, Telugu, Tamil, Gujarati, Bengali, Malayalam, Oriya and Gurmukhi (9 scripts in total). S1 is 100% Latin. Some address tails are also in regional script, e.g. `महाराष्ट्र`, `ಕರ್ನಾಟಕ`.
- **Test country mix:** India 46.8%, US 38.3%, France 15.0% of test S1. **France is 15% of the score with zero labels.**

**Ground truth (train)**

| Property | Value |
|---|---|
| Positive pairs | 7,638,365 (S2: 3,693,619 · S3: 3,944,746); no duplicates |
| Matches per S1 | mean **3.46**, max **11**, at most 5 from S2 and at most 6 from S3 |
| Cardinality distribution | 0: 5.58% · 1: 5.40% · 2: 17.0% · 3: 24.1% · 4: 21.9% · 5: 14.6% · 6: 7.5% · 7: 2.9% · 8+: 1.1% |
| Singletons | **5.58%** (India 5.57%, US 5.59%) |
| S2/S3 records under more than one S1 | **0**, so exclusivity holds |
| S2/S3 records with no S1 (distractors) | 2,681,854, i.e. **26%** (S2 73.4% matched, S3 74.6% matched) |
| Cross-country pairs | **0**, so partitioning by country costs no recall |
| ID / row-order leakage | Spearman 0.0001 / 0.0001. Nothing to exploit or overfit |

**Ambiguity: names do not identify businesses**

| | India S1 | US S1 | France S1 (test) |
|---|---|---|---|
| S1 entities sharing their normalized name with ≥1 other S1 | **53.6%** | **45.2%** | **45.8%** |
| Largest same-name group | 194 (`mumbai india`) | 546 (`meridian`) | 493 (`bordeaux club`) |

- Only **47.3%** of true pairs have identical crude-normalized names.
- **8.6%** of Latin true pairs share *zero* name tokens, even after phonetic skeletons. These are domains and handles (`aleenhicksgranite.com`, `#KUMARBIOTECH`), vowel-dropped forms (`Plsft` for Pasoft), and pseudo-word trade names (`Ciraarcjax`, `Veovio`) that only the address can link.
- 20.6% of distractors have exactly the name of some S1 in the same country.
- **Distractors are "siblings"** (error analysis, §3 Round 2): perturbed copies of real businesses with one name token swapped and/or the house number nudged, e.g. `Laxmi Estate Steel @ D-904` vs S1 `Laxmi Estate @ D-901`.

**Address signals**

| Signal | Value |
|---|---|
| 6-digit PIN present | S1 0.1%, S2/S3 0.5–0.8%. **Postcode blocking is useless** |
| Any 5-digit number | 4.4–6.7% (mostly US street numbers) |
| Landmark references (`near`, `opp`, `behind`) | 2.9–5.2% |
| Uppercase addresses in S2 | 53–67% |
| House-number formats | `05526` (zero-padded), `756-758` (range), `##657`, `2-16-133/6`, `D-904`, `B3/108`, `14 BIS`; truncation, e.g. `919-A-89` → `19-A-89` |
| State forms | code, full name and regional script all occur (`MH` / `Maharashtra` / `महाराष्ट्र`) |

**Vocabulary transfer:** 99.56% (US) and 99.67% (India) of test S1 name-token occurrences appear in the train vocabulary. The name generator reuses its vocabulary, so dictionaries learned from train transfer to test.

**France (test, unlabelled).**
- *Names* are templated from a small vocabulary: top tokens are `sarl, sas, club, france, de, eurl, amicale, ecole, maison…`. They carry diacritic noise (`Pàrents`, `Çlub`), dotted legal forms (`S.A.S`, `E.U.R.L.`), initials as names (`DL` = Danimation Loisirs) and pseudo-word trade names.
- *Addresses* show:
  - Street-type abbreviations: `R.`, `AV`, `IMP`, `ALL`, `BD`.
  - Number suffixes: `14 BIS`, `2 QUATER`.
  - Number markers: `N°`, `NO`, `#`.
  - Departments substituted for regions (`Nord` for `Hauts-de-France`).
  - Joined apostrophes (`lAlcazar`).
  - **Several different businesses at one address** (e.g. `30 Rue Caventou, Lille`). *Measured later (R3-F11):* **33.5% of France S1 entities share their exact address with another S1 (largest group 114), against 5.5–7.5% in US/India.**

### 1.4 What the profile implies for design

1. **Joint name+address evidence is mandatory**, both for blocking and for matching. Name-only retrieval cannot pick the right entity among 45–54% name twins (measured in Round 1).
2. **The blocking direction should be S2/S3 → S1** (reverse). Because of exclusivity, each record needs only a few S1 candidates, and total candidate volume is bounded by k × |S2∪S3|.
3. **Address anchors are house numbers, street/locality tokens and cities**, not postcodes. House numbers must be parsed as full compound tokens (sub-parts, letters, suffixes), because distractors differ precisely there.
4. **Cross-script handling must be rule-based and data-learned** (romanizer plus a dictionary learned from train pairs). About 20% of India S2/S3 names are non-Latin.
5. **An address-only path is needed** for the 2–3% of true pairs with pseudo-word or unrelated trade names.
6. **The decision layer must enforce "≤1 S1 per record"** and be precision-first. Distractors are 26% of train records and probably about 40% of test records (E6).
7. **Everything must be country-agnostic** and must carry a real, measured strategy for an unseen country. France is 15% of the score.
8. **Scale:** 1.7M S1 against 10M S2/S3 in test, on a 24 GB laptop. Only inverted-index retrieval with stop-feature pruning is tractable. Near-uncapped retrieval (df-cap 50k) was measured at 457–926 queries/s, i.e. 3–6 hours for test.

---

## 2. Phase 1 — The team in the room

| Specialist | Owns | Is specifically responsible for catching |
|---|---|---|
| **BLK** — Blocking & Retrieval Lead | Candidate generation, recall ceiling, reduction ratio, candidate budget | Recall lost to key design, twin-name collapse, empty-address records, candidate explosion, train/test candidate parity |
| **NRM** — Multilingual Normalization Specialist | Unicode, transliteration of 9 Indic scripts, French and other unseen conventions, phonetics, name and address parsing | Script-specific romanization errors, asymmetric normalization, hard-coded language lists, destructive normalization rules |
| **FEA** — Feature Engineering & GBDT Specialist | Feature set, labels, LightGBM setup, stacking | Leakage (entity- and fold-level), train/serve skew, redundant or fragile features, country-identifying features, overfitting to generator quirks |
| **DEC** — Decision Theory & Metric Specialist | F0.5 macro, thresholds, exclusivity, singleton policy | Rules misaligned with per-entity macro F0.5, ignored exclusivity, over-merging on singletons and twins |
| **VAL** — Validation & Generalization Scientist | Holdout design, unseen-country simulation, distribution shift, honest estimates | Optimistic validation, no France plan, threshold overfitting, train→test shift |
| **SCL** — Scalability & Infra Reviewer | Runtime and memory on the M5 with 24 GB, parallelism, determinism | O(n·m) stages, Python-object memory blow-ups, single-threaded hot loops, non-deterministic hashing |
| **CMP** — Compliance, Fair-Play & Release Officer (**absolute gate**) | External-data ban, licenses, the 8B cap, the classical-only constraint, output contract, reproducible package | Any external lookup or data, any pretrained or embedding component, GPL contamination, format violations, irreproducible artifacts |

CMP holds a veto. A CMP-Critical issue blocks convergence regardless of its accuracy benefit.

---

## 3. Phase 2 — Design iteration log

**How the loop ran.** Each round critiqued the current draft from every seat. Wherever possible, a critique was **tested with a probe on real data before it was accepted or rejected**, so most critiques cite a measured number.

**Severity scale:**
- **Critical:** violates a hard constraint, costs 0.5 or more F0.5 points, or makes the run infeasible on the target machine.
- **Major:** costs 0.1–0.5 points, or carries a real robustness or maintainability risk.
- **Minor:** everything else.

**Convergence rule:** a full round with **no new Critical issue from any seat**.

### 3.1 Draft v0 — the "reasonable first instinct"

| Stage | v0 choice |
|---|---|
| Ingest | `pandas.read_csv(sep="\t")` with default quoting |
| Normalize | Lowercase, strip punctuation, `unidecode` for non-Latin, hard-coded English abbreviation map (Pvt→Private, Rd→Road…), strip legal suffixes |
| Blocking | Per country: character 3-gram TF-IDF on **names**, then `sparse_dot_topn` top-20 S2/S3 per **S1** (forward), plus exact PIN/ZIP key blocks |
| Features | rapidfuzz ratio / partial / token_sort / token_set on name and address, Jaro-Winkler, Levenshtein, Jaccard, Soundex/Metaphone equality of the first name token, PIN/ZIP equality, **country one-hot**, lengths |
| Model | XGBoost binary with default parameters; **random 80/20 split over pairs** |
| Decision | Match if p ≥ 0.5; every S1 gets all candidates above threshold |
| Validation | F0.5 on the random pair split |

### 3.2 Round 1 — critiques of v0

| ID | Seat | Sev. | Critique | Evidence | Resolution |
|---|---|---|---|---|---|
| R1-B1 | BLK | **Critical** | Name-only retrieval cannot find the right S1 among name twins | India, full 883k index: recall@10 is **51.2%** (cap 2k), **61.2%** (cap 10k), 64.3% (cap 50k). Joint name+address at cap 10k reaches **95.4%** | **Accepted.** Joint name+address scoring |
| R1-B2 | BLK | Major | PIN/ZIP key blocks contribute almost nothing | PIN appears in 0.1% of S1 and 0.5–0.8% of S2/S3 | **Accepted.** Dropped |
| R1-B3 | BLK | **Critical** | A full-vocabulary sparse product is intractable, because common grams (`ent`, `pri`, `ing`) have enormous posting lists | At cap 50k, about 280k postings per query → **457 q/s**, i.e. about 3 h for India test alone | **Accepted.** Stop-feature df-cap plus a custom inverted-index top-k |
| R1-B4 | BLK | Major | Forward top-20 per S1 ignores exclusivity and fans out on twin names: the same record appears in hundreds of twin lists | Structural argument, plus E5 | **Accepted.** Reverse direction (S2/S3 → S1, top-k) |
| R1-B5 | BLK | Major | Pseudo-word trade names (`Ciraarcjax`) are invisible to name blocking | 8.6% of true pairs have zero name overlap, of which roughly a third are pseudo-words | **Accepted.** Address features are part of retrieval |
| R1-N1 | NRM | **Critical** | Default CSV quoting on 813 test rows. Quoted fields like `"""ehpad Club SAS"` need RFC-4180 unquoting, and a permissive parser can run across rows | Raw scan; all rows have exactly 4 tab fields | **Accepted.** Raw tab split plus explicit unquote |
| R1-N2 | NRM | **Critical** | `unidecode` gives generic romanization with no script rules, and Indic names stay unmatchable | Indic true pairs: token-sort p50 **0.59** vs random-pair p99 **0.53**, i.e. no separation | **Accepted.** Stdlib romanizer (see R2-N5) |
| R1-N3 | NRM | Major | Mojibake, control characters and `<NULL>` tokens pollute tokens | E2, E3 | **Accepted.** Cleaning map |
| R1-N4 | NRM | Major | An English-only abbreviation map cannot cover France (`R.`, `AV`, `IMP`, `BIS`, `S.A.S`) or any other unseen country | France samples | **Accepted.** Solved generically in R2-N7 |
| R1-F1 | FEA | **Critical** | A country one-hot violates C6 and has no value for France | C6 | **Accepted.** Removed; country is never a feature |
| R1-F2 | FEA | **Critical** | A random *pair* split leaks: pairs of the same S1, and competing candidates of the same record, land on both sides | Structural | **Accepted.** Split by S1 entity |
| R1-F3 | FEA | Major | First-token Soundex/Metaphone equality is English-centric and single-token | Romanized Hindi breaks English rules (`praivet`) | **Accepted.** Multi-token, Indic-aware phonetic skeleton overlap |
| R1-F4 | FEA | Major | No context features (rank among candidates, gap to best, twin count), so identical names cannot be discriminated | Later confirmed: context features carry **74–78%** of LightGBM gain | **Accepted** |
| R1-F5 | FEA | Minor | XGBoost vs LightGBM | Both licenses fine | **Accepted** LightGBM (faster on CPU at 10M+ rows, native NaN, monotone constraints). See T6 |
| R1-D1 | DEC | **Critical** | Fixed p ≥ 0.5 ignores the precision weighting | India probe: t=0.5 gives **0.9477**, t=0.7 gives **0.9525** | **Accepted.** Tune t on macro F0.5 |
| R1-D2 | DEC | Major | Independent per-pair thresholds can assign one record to several twins, although ground truth proves exclusivity | E5 | **Accepted.** Argmax per record |
| R1-D3 | DEC | Major | Any false positive on a singleton (5.6%) scores 0 | C10 | **Accepted.** Precision-first rule plus ambiguity features (singleton loss later measured at 0.37 pts) |
| R1-V1 | VAL | **Critical** | No plan for the unseen country (15% of score) | Test mix | **Accepted.** Leave-one-country-out (LOCO) simulation plus unsupervised France diagnostics |
| R1-V2 | VAL | Major | A pair-level metric hides blocking misses and singletons | C10 | **Accepted.** Evaluate the full pipeline with the exact macro formula |
| R1-S1 | SCL | Major | pandas object strings for about 24M rows use 15–20 GB | Estimate | **Accepted.** polars/Arrow |
| R1-S2 | SCL | Major | Python loops over tens of millions of pairs take hours | — | **Accepted.** `rapidfuzz.process.cpdist` (C++, multi-threaded) plus numba |
| R1-C1 | CMP | **Critical** | `unidecode` and `python-Levenshtein` are **GPL**. Shipping GPL code in the submission is a license-contamination risk | License metadata | **Accepted.** Stdlib romanizer and rapidfuzz (MIT) only |
| R1-C2 | CMP | Major | "Just for normalization" gazetteers (PIN→city, ZIP databases, state tables downloaded from the web) count as external data augmentation | Fair-play rules | **Accepted.** Nothing is downloaded. Mappings are learned from provided data (R2-N7) |
| R1-X1 | FEA | — | *Proposal:* add Soundex, NYSIIS and Double-Metaphone equality for every token | — | **Rejected for now.** English-centric and redundant with the skeleton. Kept as an optional ablation in Phase 4 |
| R1-X2 | BLK | — | *Proposal:* add a forward pass (per-S1 top-K) for recall | Reverse @5 already gives 97–98%; a forward pass would re-add twin fan-out | **Rejected** |

**Revision v1**

| Stage | v1 choice |
|---|---|
| Ingest | polars raw tab split, RFC-4180 unquote, cleaning map |
| Normalization | Stdlib Brahmic romanizer plus phonetic skeleton; seed English abbreviations |
| Blocking | Reverse direction; per-country TF-IDF over name (3-grams, tokens, skeletons) and address (tokens, skeletons, numbers) with weights 0.5/0.5; inverted index with df-cap; numba top-k, k=10 |
| Features | About 28 country-agnostic features |
| Model | LightGBM |
| Decision | Argmax per record, then a tuned global threshold |
| Validation | S1-grouped split, exact macro F0.5 on the whole pipeline, LOCO planned |

### 3.3 Round 2 — critiques of v1

| ID | Seat | Sev. | Critique | Evidence | Resolution |
|---|---|---|---|---|---|
| R2-B6 | BLK | **Critical** | v1's cap trade-off is still bad. Single features cannot separate twins cheaply | India, joint 0.5/0.5: cap 2k gives recall@10 **88.5%** at 19.6k q/s; cap 10k gives **95.4%** at only **2.3k q/s** | **Accepted.** **Conjunctive keys**: name-skeleton×house-number, name-skeleton×locality-token, house-number×street-token. **Measured: cap 2k, recall@10 96.8% at 11.2k q/s (India) and 97.5% at 15.0k q/s (US)**. Top-1 score also separates better: matched vs distractor p50 0.60 vs 0.36 (previously 0.59 vs 0.50) |
| R2-B7 | BLK | Major | Character 3-grams add spurious overlap | India recall@10: 3-grams **96.8%** vs 4-grams **97.6%**, @50 98.0% vs **98.7%**, at the same speed | **Accepted.** 4-grams |
| R2-B8 | BLK | Major | k=10 across 10M records gives 100M pairs, which costs heavily in features and in the candidate file | Keeping candidates with score ≥ 0.5×best keeps **37.6%** of candidates and loses **0.01%** of positives. An absolute cutoff of 0.15 loses 2.6% | **Accepted.** Ratio pruning ≥ 0.5 (rank-1 always kept). **Measured 1.88 (India) / 1.68 (US) candidates per record at k=5, losing 105 and 49 true pairs**; about 2 per record at k=10 |
| R2-B9 | BLK | Minor | Remaining misses are dominated by empty-address records | Misses@10: **US 86%**, **India 45%** have an empty query address | **Accepted as a known ceiling.** These are name-only records with many twins and are mostly unresolvable even with perfect recall. The matcher is taught to abstain (ambiguity features) instead of adding a special pass |
| R2-N5 | NRM | **Critical** | Rule-based romanization alone leaves Indic names weakly matchable | Held-out Indic pairs: token-set p50 is **77** | **Accepted.** A **transliteration dictionary learned from train pairs** (support ≥3, precision ≥0.6): **713 entries**. Held-out p10/p50 goes 58/77 → **100/100**, **90.3%** exact; **test token coverage 94.7%** |
| R2-N6 | NRM | Major | Script-specific romanization rules are missing: Tamil `ச`→s initially, intervocalic voicing, `ஃப`→f; Malayalam `റ്റ`→tt and chillu; final schwa kept after conjuncts | Failures such as `சன் பவர்`→`chan pavar` (Sun Power), `limirrad` | **Accepted.** Implemented in the probe and verified |
| R2-N7 | NRM | Major | Abbreviations for France and any unseen country need a *generic* mechanism, not a per-country list | — | **Accepted.** An **unsupervised abbreviation learner**: token substitutions seen in identical local contexts, kept only if abbreviation-shaped. On *unlabelled* France addresses it found `r→rue` (121,592 contexts), `av/ave→avenue`, `bd/blvd→boulevard`, `all→allee`, `imp→impasse`, `b→bis`, `t→ter`, `st→saint`, and typo fixes (`avnue→avenue`, `impase→impasse`). On US addresses it learned `tx→texas`, `oh→ohio`, `wi→wisconsin` **without any gazetteer** |
| R2-N8 | NRM | Major | Name decorations are mishandled | A probe **bug**: `(ID: primeadvantage.com 4355)` became an empty core name. Also `| www.x.com` suffixes, `@`/`#` handles, `F/K/A`, `M/s`/`Smt`/`Mr`, and leetspeak `5ervices` | **Accepted.** Parse decorations into name *variants* and score the max over variants. Leetspeak digit→letter repair inside alphabetic tokens. Honorific and noise lexicon (R2-F6) |
| R2-N9 | NRM | Minor | Dotted initialisms (`S.A.S`, `E.U.R.L.`, `P.A.`, `O.D.`) split into letters | Samples | **Accepted.** Collapse |
| R2-F5 | FEA | **Critical** | **Sibling distractors cause 86% of false positives.** They are perturbed copies differing in one distinctive name token and/or a house-number sub-part | India error analysis at t=0.7: 1,357 of 1,575 FPs are distractors, e.g. `2-16-133/6` vs `/2`, `D-904` vs `D-901`, `A25` vs `A20`, `Superior Kumar` vs `Superior Infra` | **Accepted.** Full compound house-number parsing and comparison (all parts, letter suffix, numeric distance), distinctive-token *substitution* features (IDF-weighted), address-twin counts |
| R2-F6 | FEA | Major | **Benign noise is over-penalized**: 65% of false negatives are below threshold because of appended generic words, honorifics, leetspeak and truncated numbers | India: 6,334 of 9,722 FNs are below threshold, e.g. `Laurel Trading Limited 5ervices`, `M/s Raas LLP Center`, `19-A-89` vs `919-A-89` | **Accepted.** An insertion-noise lexicon *learned from train pairs* (tokens frequently present in S2/S3 but absent in the matched S1), an "all distinctive tokens matched" feature, and number prefix/suffix containment |
| R2-F7 | FEA | Major | Train/serve skew: IDF, twin counts and retrieval scores depend on index size, and the test US index is half the size of train (663k vs 1.32M) | Profile | **Accepted.** Statistics are computed per partition from the processed split's S1. Relative and rank features are preferred |
| R2-D4 | DEC | Major | Argmax assignment alone adds almost nothing (context features already encode competition) | India: rule A **0.9525** vs argmax rule B **0.9530** | **Accepted, reframed.** Keep argmax as a *zero-cost hard safety*. Real gains must come from collective information: stage-2 measured **+0.48 pts** (0.9520 → **0.9568**), driven by `p1_margin` (26% of stage-2 gain) |
| R2-D5 | DEC | Major | Is per-S1 expected-F0.5 subset selection better than a global threshold? | Stage-1: 0.9511 vs 0.9520 (**worse**). Stage-2: 0.9576 vs 0.9568 (+0.08) | **Accepted as an optional variant.** Global threshold stays primary. Expected-F is kept only if it wins on the shifted validation set |
| R2-V3 | VAL | **Critical** | **Distribution shift:** test has 5.53–5.82 S2/S3 records per S1, train 4.67. If matches per S1 are unchanged, test distractors are about 40% against 26% | Profile (E6) | **Accepted.** Simulate by **dropping 19% of S1 from the index** so their matches become distractors. **Measured India: 0.9530 → 0.9479 (−0.5 pts); optimal t stays 0.7.** Thresholds are chosen on the shifted condition |
| R2-V4 | VAL | **Critical** | The size of the unseen-country penalty is unknown | — | **Accepted.** LOCO run. Result: **US→India 0.8507** (in-domain 0.9530), **India→US 0.9572** (in-domain 0.9740). Pooled training costs 0.2 pts in-domain. The penalty is large and asymmetric, which escalates to R3-V7 |
| R2-S3 | SCL | Major | Probe peak RSS reached 10.0 GB (India) and **15.4 GB** (US) because of Python feature lists and `hstack` copies | `/usr/bin/time -l` | **Accepted.** Featurize straight into numpy arrays per chunk, one country at a time, stream queries in chunks of 400k or fewer, no whole-matrix copies. Target ≤ 12 GB |
| R2-S4 | SCL | Minor | Is the throughput enough? | Measured 9.8k q/s (India, 4.13M queries in 7:00) and about 12k q/s (US) | **Accepted.** About 15–18 min for 10M test queries |
| R2-C3 | CMP | Major | Is it fair to use *unlabelled test* data (IDF, abbreviation learning, twin counts)? | The rules forbid **external** data; test files are provided data. No labels are involved | **Accepted and documented transparently** in the methodology. Without it France is impossible (it has no train data). Flagged for your review in §4.11 |
| R2-C4 | CMP | Minor | Learned resources must be reproducible artifacts, not hand-made files | C9 | **Accepted.** The dictionary, abbreviation rules and lexicons are generated by pipeline code on each run |

**Revision v2** changes from v1:
- Conjunctive keys, 4-grams, k=10 with ratio pruning.
- Learned transliteration dictionary, script-specific rules, unsupervised abbreviation learner, decoration and leetspeak handling.
- Sibling-aware number features and insertion-noise features.
- Stage-2 collective model (out-of-fold stage-1).
- Shift-simulated validation; LOCO; memory-lean chunking.

### 3.4 Round 3 — critiques of v2

| ID | Seat | Sev. | Critique | Evidence | Resolution |
|---|---|---|---|---|---|
| R3-F9 / V7 | FEA + VAL | **Critical** | **v2's matcher does not survive an unseen country.** A model that never saw a country's feature regime ranks that country's candidates badly, and re-thresholding cannot fix a ranking problem | LOCO: **US→India 0.8507** (in-domain 0.9530; singleton false-positive rate 23.1% vs 6.7%), **India→US 0.9572** (in-domain 0.9740). Best LOCO thresholds drift (0.8 / 0.6 vs 0.7) | **Accepted as Critical.** Remedies were tested in the LOCO harness. **Round A** (`loco2.py`, all measured; US→India / India→US):<br>(1) drop "fragile" features **0.8219 / 0.9132**, *worse*;<br>(2) + monotone constraints **0.8160 / 0.9123**, *worse*;<br>(3) + self-training on the target's unlabelled pairs **0.8167 / 0.9134**, *no gain*: pseudo-labels reinforce the model's own ranking errors.<br>The oracle-best threshold beats fixed 0.7 by only 0.001, so the gap is a **ranking** problem, not calibration. **All three rejected.** **Round B** (`loco3.py`, run with the learned dictionary):<br>• The **dictionary alone** lifts US→India from 0.8507 to **0.8992**. About half the penalty was a script problem, which France (Latin script) doesn't have.<br>• A **cardinality-matched threshold** needs no labels. It picks t so that predicted matches per S1 equals the model's in-domain value, which is justified because cardinality is country-invariant (3.462 vs 3.461). It gives US→India **0.8992→0.9131** (oracle-t 0.9183) and India→US **0.9539→0.9563** (oracle-t 0.9566), recovering **73% / 89%** of the oracle-threshold gain *in both directions*.<br>• Per-country percentile normalization (−1.3 / +0.4 pts) and full-feature self-training (−1.1 / +0.6) each help one direction and hurt the other, so both are **rejected as not robust**.<br>Residual after every label-free measure: −5.0 pts (US→India, a very different regime) and −1.8 pts (India→US). **Adopted:** dictionary + cardinality-matched thresholds + co-location features (R3-F11) + pooled training. |
| R3-F8 | FEA | **Critical** | Stage-2 leaks if its collective features use *in-sample* stage-1 scores: p1 on training pairs is over-confident, so stage-2 learns to trust it blindly and fails on test | Standard stacking failure. The probe's +0.48 pts was measured **with** out-of-fold p1 | **Accepted.** Strict 5-fold OOF grouped by S1. The validation fold is never seen by either stage. Test p1 is the fold-model average |
| R3-F10 | FEA | Major | Several v2 features are *implicit country detectors*: `q_indic` is always 0 outside India, and number counts and name lengths follow country address formats. The GBDT can split on them and extrapolate badly | Feature distributions per country; see R3-F9 | **Tested, then REJECTED.** Dropping them costs **−11.8 pts in-domain on India (0.9530→0.8348) and −4.5 pts on US (0.9740→0.9287)**, and makes transfer worse (R3-F9). They are core signal: absolute retrieval scores and twin counts carry information the relative features don't. Distribution shift in them is addressed by *normalizing* rather than deleting (R3-F9 round B) |
| R3-F11 | FEA + VAL | **Critical** | **Root cause of France over-merging: co-location.** An exact address match is near-unique evidence in US/India, so the model learned to trust it. In France it isn't | Exact normalized address shared with ≥1 other S1: **France 33.48%** (max group 114) vs **US 5.49–7.45%, India 6.61–6.78%** (max 11–21). Explains France's excess predicted matches (R3-V5) | **Accepted.** **Address-twin features**, all computed per partition without labels: (a) how many S1 entities share the candidate's address; (b) whether the candidate is the best *name* match among the query's candidates at that address; (c) how many of the query's candidates share an address. Training has 6–7% co-located entities to learn from. **Measured in-domain (`coloc_probe.py`):** co-located S1 score about 1 pt *below* unique-address S1 (India 0.9541 vs 0.9639; US 0.9643 vs 0.9747), so errors concentrate there. **With address-twin features: India co-located +0.65 pts (singleton false-positive rate 8.7%→3.8%), US co-located +0.50 pts; unique-address entities unchanged (±0.01).** In training this is a 7% slice (+0.02–0.05 overall). **In France it is 33% of entities.** Adopted |
| R3-B10 | BLK | Major | Training candidates must be generated under the **same conditions as test**: same k, cap and pruning, *and* the same distractor density. Otherwise stage-1 learns on an easier negative distribution than test | E6; shift probe (−0.5 pts) | **Accepted.** Training candidates come from a per-country S1-dropped index. The drop fraction is computed from data as dᶜ = 1 − rᶜ_train / rᶜ_test (records per S1), about 19%. Matches of dropped S1s become labelled distractors |
| R3-B11 | BLK | Minor | With two model stages, which set is "the last-stage candidate set" for `candidate_pairs.tsv`? A stage-2 that re-ranked a larger set would make the file wrong | C5 wording ("exact set of records you feed into your matching model") | **Accepted.** Pruning happens once, before stage-1. Both stages score exactly that set, and the file is written from it. `matched ⊆ candidates` is asserted in code |
| R3-B12 | BLK | Minor | Partition edge cases: an S2/S3 country absent from S1, an empty country label, or a tiny partition. Each would either crash the index build or silently drop records | C6 (open set); ingest checks | **Accepted.** Empty partitions short-circuit to "no candidates", with a logged count. The empty label is its own partition. Unit tests cover a 3-record partition and an S1-less partition |
| R3-S9 | SCL | Major | **multiprocessing on macOS uses `spawn`, which re-imports the entry module in every worker.** Module-level work (argv parsing, loading data) then runs 10× or crashes | **Hit twice while building the probes** (`probe_apply.py` recursive re-execution; `probe_block2.py` argv parsing at import) | **Accepted.** All entry points sit behind `if __name__ == "__main__"`. Worker functions are pure and top-level. No work happens at import time; a lint test asserts that importing each module is side-effect-free |
| R3-S7 | SCL | Major | **Hash collisions.** Conjunctive keys create roughly 50–60M distinct features per partition (about 64 per record × 0.88M), against 2²³ = 8.4M buckets in the probe, i.e. about 7 features per bucket. That inflates df, corrupts IDF, and adds spurious candidate hits | Arithmetic on the probe config | **Accepted.** 64-bit stable keys (two CRC32s), **compacted per partition with `np.unique`**, giving an exact vocabulary. The probe recall was measured *despite* collisions, so this can only help |
| R3-N10 | NRM | Major | Learned abbreviation rules have **observed errors**: `rue→route` and `no→nord` (France addresses); `sa→sarl`, `de→des`, `du→distribution` (France names); `fl→floor` where FL also means Florida (US) | Learner output (§3.3 R2-N7) | **Accepted.** Guards: the short form must be rarer than the long form; the pair must cover ≥5% of the short form's contexts; support ≥20. **Soft expansion** only (append, never replace), so an ambiguous `fl` keeps both readings. Stricter thresholds for names than for addresses |
| R3-N11 | NRM | Major | The transliteration dictionary could leak validation labels and could mis-map ambiguous tokens | — | **Accepted.** Learned only from **training-fold** entities. Rebuilt leakage-free: still **713 entries**. Applied only to tokens of non-Latin names, and the original token is kept as a variant |
| R3-N12 | NRM | Minor | Regional-script tokens inside *addresses* (state names such as `महाराष्ट्र`) are not covered by the name dictionary | Samples | **Accepted (cheap).** Same alignment learner over address tokens. Impact is low because state tokens have low IDF |
| R3-D6 | DEC | Major | A threshold tuned in-domain is not the optimum for an unseen country (LOCO optima 0.6–0.8) | R3-F9 evidence | **Accepted.** Choose t\* by the **worst-case** criterion across US-shift, India-shift and both LOCO directions. Then apply a label-free **cardinality-matched t_c per partition** (R3-F9 round B), which moved LOCO +1.4 / +0.24 pts |
| R3-D7 | DEC | Minor | Argmax ties make the output non-deterministic | — | **Accepted.** Break ties by retrieval score, then S1 ID |
| R3-V5 | VAL | Major | France needs *unsupervised* health checks, because labels are impossible | — | **Accepted and measured.** The pooled probe model on test France predicts **3.55 matches per S1 with 4.8% singletons** at t=0.7; the cardinality histogram tracks the train ground-truth prior (3.46, 5.6%) bin by bin. **But the same model predicts only 3.22 (test India) and 3.33 (test US)**, matching its train-validation values (3.19 and 3.28), so France is the outlier. Cardinality is country-invariant in train (3.462 vs 3.461) and in-domain recall is about 0.93, so **3.55 is only reachable with extra false merges: France precision ≲0.95 at t=0.7 vs 0.98 in-domain.** The unseen-country risk therefore shows up as **over-merging**, the error F0.5 punishes most, which motivates R3-F9 round B |
| R3-V6 | VAL | Major | Is the "more distractors in test" hypothesis (E6) right, or does test simply have more matches per S1? The answer changes where the threshold should sit | — | **Accepted and measured.** Same pooled model at t=0.7: predicted matches per S1 are **test India 3.22 vs train-val India 3.19** and **test US 3.33 vs train-val US 3.28**, i.e. unchanged even though test has 24% more records per S1. **The extra test records are distractors, not matches.** This confirms the shift simulation (R2-V3) and the use of the train cardinality prior |
| R3-S8 | SCL | Major | Stage-2 needs *all* stage-1 scores of a partition before any collective feature exists, which means two passes over about 20–25M pairs | — | **Accepted.** Store (record, S1, p1) as int32/int32/float32, about 0.3 GB, and compute aggregates with numpy `bincount`/`maximum.at` |
| R3-C5 | CMP | Major | If self-training on France's unlabelled pairs is adopted, the model is adapted using test inputs. There are no labels and no external data, but a strict reader could object | — | **Moot for self-training**: it was rejected on accuracy grounds (round B), so no model is ever trained on test inputs. The adopted *label-free* adaptations (unsupervised statistics, and the cardinality-matched threshold, which reads only the distribution of the model's own predictions) are documented and switchable (§5) |
| R3-C6 | CMP | Minor | A license table and pinned versions are required in the package | C9 | **Accepted** (§4.11) |
| R3-X3 | BLK | — | *Proposal:* scale the df-cap with index size | Cost and recall are governed by posting length, not by relative frequency. Small partitions are cheap anyway | **Rejected.** Keep 2,000 fixed, log the share of pruned postings per partition, and revisit only if a partition's recall@10 drops |
| R3-X4 | DEC | — | *Proposal:* graph clustering / transitive closure across S2↔S3 before deciding | Exclusivity plus S1-anchored output already give a star structure. Transitive chaining through sibling distractors would *add* false merges | **Rejected** in favour of stage-2 collective features |

**Revision v3.** Everything in v2, plus:
- Strict OOF stacking.
- Shift-matched training candidates.
- 64-bit hashing with compaction.
- Guarded, soft abbreviation expansion.
- Leakage-free dictionary (names and addresses).
- Worst-case threshold selection.
- France health checks.
- Unseen-country package: dictionary, co-location features (R3-F11), pooled training, and cardinality-matched per-partition thresholds. No monotone constraints, feature dropping or self-training (all measured harmful or inconsistent).

### 3.5 Round 4 — convergence check on v3

Every seat re-reviewed v3 against its checklist. **No seat raised a new Critical issue.** Remaining items are logged as non-blocking and assigned to implementation phases:

| ID | Seat | Sev. | Remaining concern | Disposition |
|---|---|---|---|---|
| R4-B1 | BLK | Minor | The residual recall ceiling (about 2% of pairs: empty-address twins, corrupted numbers) is accepted, not solved. Is k=20 worth it? | P2 ablation on k ∈ {10, 20}. Adopt only if validation F0.5 improves by ≥0.1 pts, since recall@50 is only +0.7–1.1 pts over @10 |
| R4-B2 | BLK | Minor | Conjunctive keys lean on house numbers. Records *without* a number fall back to name×locality keys, and their recall hasn't been broken out | P2 reports recall@10 sliced by "query has a house number". A dedicated pass is added only if that slice is ≥2 pts below the rest |
| R4-B3 | BLK | Minor | One global df-cap across partitions with very different vocabulary entropy (France's templated names vs US names) could prune very different shares of postings | Log pruned-posting share per partition. Alert if France's share is outside the US/India range |
| R4-N1 | NRM | Minor | Scripts absent from the data (Arabic, Cyrillic, CJK) go through the generic fallback, which is untested on real data | Unit tests on synthetic strings. Such scripts don't occur in the provided data, so there's no score risk |
| R4-N2 | NRM | Minor | Leetspeak repair could corrupt legitimate alphanumeric names (`3M`, `A1`, `7eleven`) | Repair only digits *surrounded by letters*, and only if the repaired token exists in the partition's S1 vocabulary. Unit tests on those examples |
| R4-N3 | NRM | Minor | French function-word and gender variation (`st`/`ste`, `du`/`de la`) isn't normalized | Left to IDF weighting (these tokens are high-frequency and low-weight) and soft expansions. No hand-written French grammar |
| R4-F1 | FEA | Minor | Some name-similarity features are redundant (ratio vs token_sort vs JW), which costs a little compute | P4: drop features with near-zero gain *and* no LOCO benefit |
| R4-F2 | FEA | Minor | The pseudo-word flag depends on the size of the partition's S1 vocabulary (France's is smaller) | Define it by relative token frequency (≤ 1 per 10⁶ tokens), not by absolute counts |
| R4-F3 | FEA | Minor | Robustness measures (monotone constraints or a pruned feature set) may cost some in-domain accuracy | Resolved by measurement (LOCO harness, in-domain REF row): dropping features costs 4.5–11.8 pts in-domain and monotone constraints give no transfer gain. Neither is used, so there is nothing to trade |
| R4-D1 | DEC | Minor | The expected-F variant is +0.03 to +0.08 pts, within noise | Adopt only if it wins on the shifted validation *and* in both LOCO directions |
| R4-D2 | DEC | Minor | The worst-case threshold rule can give up about 0.1 pts in-domain in exchange for robustness | An accepted trade, since France is 15% of the score. The in-domain-optimal and chosen t\* are both reported |
| R4-V1 | VAL | Minor | The France estimate is inferred, not measured. That is irreducible without labels | Mitigated by LOCO bounds plus unsupervised diagnostics. The residual uncertainty is stated in §4.9 |
| R4-V2 | VAL | Minor | LOCO gives only two transfer directions (n=2), so the France penalty bound is coarse | Stated as a range, not a point. The France diagnostics act as an independent sanity check |
| R4-V3 | VAL | Minor | Bootstrap CIs don't capture how the public and private leaderboards split test | Documented. We don't chase public-leaderboard deltas smaller than about ±0.003 |
| R4-S1 | SCL | Minor | The ≤12 GB peak is a target, not yet a measurement (the probes hit 10–15 GB) | A hard gate in P2 and P3, with psutil high-water logging and automatic chunk shrinking |
| R4-S2 | SCL | Minor | Parquet checkpoints need disk: normalized records ~3 GB, plus candidates and features ~4 GB per split | ~15 GB in total against 436 GB free. Checkpoints can be cleaned by `make clean` |
| R4-S3 | SCL | Minor | Numba JIT compile time on every run | `@njit(cache=True)`. It's warmed once, and the cost is a few seconds |
| R4-C1 | CMP | Minor | Two judgement calls need your confirmation: unlabelled-test statistics, and seed lexicons | Both are documented and switchable by config, with their accuracy cost measured (§5) |
| R4-C2 | CMP | Minor | `design_probes/` (throwaway code with hard-coded paths) must not end up in the submission zip | The zip builder uses an explicit allow-list: `output/`, `code/business_entity_resolution/`, `Documentation_template.md` |
| R4-C3 | CMP | Minor | The seed lexicons must be auditable | Listed verbatim in the methodology document's appendix |

**Convergence bar**

| Criterion | Where it is met |
|---|---|
| Every Phase-0 constraint is satisfied by design | §1.2 table: C1–C12 and E1–E6 each map to a mechanism, plus tests and gates (§4.14) |
| Recall ceiling and reduction ratio are reasoned explicitly | §4.4: measured recall@k by country and cap, oracle F0.5 ceiling, RR ≈ 0.999997 |
| Provably tractable at test scale | §4.10: every stage is measured or derived from full-country probes; ~45 min inference, ~1.5–2 h full; ≤12 GB target gated |
| Explicit unseen-country strategy, classical only | §4.8: mechanism stack with LOCO-measured effects |
| Explicit precision-over-recall strategy | §4.7: exclusivity, F0.5-tuned threshold under shift, worst-case selection, singleton policy |
| GBDT library/version named; license and size verified | §4.6: LightGBM 4.7.0, MIT, <5M parameters |
| Zero external services or data | §4.11 audit |
| Holdout + exact F0.5 + simulated unseen category | §4.9: S1-grouped folds, exact macro formula (unit-tested), shift simulation, LOCO |

### 3.6 Trade-off analyses (real alternatives)

**T1 — Candidate generation method**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **(a) df-capped inverted-index TF-IDF + conjunctive keys, numba top-k** | Exact top-k per record; cap tunes cost; conjunctive keys resolve twins; **measured 97.5–97.8% recall@10 at 10–15k q/s** | Custom kernel to maintain; needs a hashing and compaction layer | **Chosen** |
| (b) `sparse_dot_topn` sparse matrix product on character n-grams | Off-the-shelf, multithreaded C++ | Computes full products (no per-feature cap unless columns are pruned by hand); conjunctive keys explode its intermediate products; uncapped-equivalent traversal measured **457–926 q/s**, i.e. hours | Rejected |
| (c) MinHash-LSH on shingles (`datasketch`, MIT) | Sublinear and classical | Approximates unweighted Jaccard, so frequent tokens (`pvt`, `road`) dominate; band/row tuning trades recall blindly; hard to weight name vs address; Python-heavy at 10M scale | Rejected |
| (d) Classic key blocking (sorted neighbourhood, phonetic key + city) | Simplest, very fast | Brittle to reordering, typos and missing cities; twin blocks (546 `meridian`) are quadratic; needs many passes to reach recall | Rejected (conjunctive keys keep its best idea inside (a)) |

**T2 — Retrieval direction:** **reverse (S2/S3→S1) is chosen.** It is bounded at k × |records|, matches exclusivity, and measured 97–98% recall@5. Forward (S1→S2/S3) fans out on twins and needs K ≥ 11 per S1 for the largest groups. Using both adds cost for recall the matcher could not use (rejected, R1-X2).

**T3 — Candidate budget:**
- k=5: recall@5 96.95% / 98.22%; 20.7M / 30.9M train pairs.
- **k=10 with ratio ≥0.5 (chosen):** @10 97.5–97.8%, about 2 candidates per record after pruning (1.7–1.9 measured at k=5).
- k=20: +0.3–0.5 pts recall at about 2× cost (R4-B ablation).

**T4 — Cross-script handling**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **(a) Stdlib `unicodedata` romanizer + script rules + dictionary learned from train pairs** | No dependencies; covers every Brahmic script; data-adapted to informal "Hinglish" spellings; **held-out p50 77→100** | Rules are hand-written (small) | **Chosen** |
| (b) `indic-transliteration` (MIT) scheme tables (ITRANS/IAST) | Mature, precise tables | Scholarly schemes (`rāma`, diacritics) differ from informal business spellings; still needs the learned dictionary; extra dependency | Rejected |
| (c) `anyascii` (ISC) / `unidecode` (GPL) | Every script in one call | Generic, no schwa deletion; weak separability (p50 0.59 vs random 0.53); `unidecode` is GPL | Rejected |

**T5 — Phonetic encoding:** **a custom romanization-aware skeleton is chosen.** It merges the Indic spelling variants measured in the data (`praivet`/`private` → `prvt`). Double Metaphone is English-centric, so it mis-codes romanized Hindi and French; it's kept as an optional ablation. Soundex and NYSIIS use first letter plus 3 digits, with heavy collisions in twin-rich data, so they're rejected.

**T6 — GBDT library**

| Option | License | Pros | Cons | Verdict |
|---|---|---|---|---|
| **LightGBM 4.7.0** | MIT | Fastest CPU histogram GBDT at 10M+ rows; native NaN; monotone constraints; deterministic mode; measured 40 s for 5.8M×28×500 | Leaf-wise growth can overfit (controlled by `min_data_in_leaf`) | **Chosen** |
| XGBoost | Apache-2.0 | Equally accurate with `hist`; mature | Somewhat slower on this CPU; no advantage here | Alternative |
| CatBoost | Apache-2.0 | Ordered boosting is robust on small data | Much slower CPU training at 20M rows; larger models | Rejected |

**T7 — Decision rule:**
- Global threshold only: 0.9525.
- **Argmax + global threshold (chosen):** 0.9530, and a structural guarantee.
- Per-S1 expected F0.5: 0.9511 (stage-1) / +0.08 (stage-2); optional.
- Per-country thresholds: rejected, since France has no labels.

**T8 — Collective reasoning:**
- **Stage-2 GBDT on OOF collective features (chosen: +0.48 pts without the dictionary, +0.39 with it).**
- Graph clustering / transitive closure: rejected (R3-X4).
- Single stage: baseline.

**T9 — Unseen-country adaptation:** 

| Option | Measured effect (LOCO US→India / India→US, with dictionary unless noted) | Verdict |
|---|---|---|
| Drop country-fragile features | −2.9 / −4.4 pts (no dictionary); −4.5 / −11.8 pts in-domain | Rejected |
| Monotone constraints | No gain (−0.6 / −0.1 on top of the reduced set) | Rejected |
| Self-training on target pairs (reduced / full features) | No gain / −1.1 and +0.6 | Rejected |
| Per-country percentile normalization | −1.3 / +0.4 | Rejected (inconsistent) |
| **Learned transliteration dictionary** | US→India **+4.9** (0.8507→0.8992); India in-domain +1.0 | **Adopted** |
| **Cardinality-matched threshold (label-free)** | **+1.4 / +0.24** (73% / 89% of the oracle-threshold gain) | **Adopted** |
| **Co-location features** | +0.65 / +0.50 pts on co-located entities in-domain (France: 33.5% co-located) | **Adopted** |
| Pooled US+India training | −0.2 pts in-domain; covers both regimes | **Adopted** |

---

## 4. Phase 3 — Solution plan

### 4.1 Architecture overview

```
 test/train TSVs
      │
 [0] INGEST ── raw tab split → RFC-4180 unquote → clean → schema checks → run report
      │
      │   ┌──────────── for each country partition c (open set; identical code path; no country branching) ─────────────┐
      ▼   │                                                                                                             │
 [1] NORMALIZE ─ script-agnostic romanizer (stdlib unicodedata) + learned transliteration dictionary (train pairs)      │
      │          names → variants / core / spaceless / skeleton / initials / legal form / honorifics                    │
      │          addresses → compound house numbers / street / locality / region + learned abbreviations (unsupervised) │
      ▼                                                                                                                 │
 [2] BLOCK ──── S2/S3 record → top-k S1 via df-capped TF-IDF inverted index (numba)                                     │
      │          views: name 4-grams+tokens+skeletons | address tokens+numbers | conjunctive name×address keys          │
      │          k=10, keep score ≥ 0.5×best  ──────────────────────────────────────────► candidate_pairs.tsv           │
      ▼                                                                                                                 │
 [3] FEATURES ─ ~50 country-agnostic features: retrieval context, name, address/house-number, ambiguity, format         │
      ▼                                                                                                                 │
 [4] MATCH ──── LightGBM stage-1 (5-fold out-of-fold by S1) → collective features → LightGBM stage-2                    │
      ▼                                                                                                                 │
 [5] DECIDE ─── per record: argmax S1 only (exclusivity) · keep if p ≥ t_c                                              │
      │          t_c = t* (F0.5-tuned under test-like shift), cardinality-matched per partition (no labels)             │
      │                                                                                                                 │
      │   └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
      ▼
 [6] WRITE + VALIDATE ── matching_results.tsv (1 row per test S1) · validator --check-ids · subset assertion```

**Why this beat the alternatives** (details in §3.6):
- **Reverse, conjunctive-key inverted-index retrieval** is the only option measured to reach ≥97.5% recall@10 at ≥10k queries/s on a laptop. Name-only blocking tops out at 51–64% recall@10. Near-uncapped retrieval runs at only 457–926 q/s.
- **A rule-based romanizer plus a dictionary learned from train pairs** takes Indic-script names from no separability to 90% exact agreement, with no pretrained or GPL component.
- **A two-stage GBDT** measured +0.39–0.48 pts over one stage. **Argmax exclusivity** is a free safety guarantee backed by ground-truth structure.
- **For the unseen country, only measured winners were kept:** the dictionary (+4.9 pts LOCO), the label-free cardinality-matched threshold (+1.4 / +0.24) and co-location features (+0.5–0.65 on co-located entities). The textbook remedies (feature dropping, monotone constraints, self-training, percentile normalization) measured negative or inconsistent and were rejected.
- **No country branching:** everything country-specific is *learned from the partition's own data* (IDF, abbreviations, noise lexicons). The pipeline therefore behaves identically for any label, France included.

### 4.2 Stage 0 — Ingest & input robustness

- **Read:** `polars.read_csv(separator="\t", quote_char=None, infer_schema=False)`, so everything is read as text. Then, per text field, if it is wrapped in `"…"`, strip the outer quotes and replace `""` with `"`. This is correct for the E1 rows, e.g. `"""ehpad Club SAS"` → `"ehpad Club SAS`.
- **Checks, each logged with a count and never fatal unless the output would be wrong:**
  - Exact header.
  - 4 fields per row. Malformed rows are salvaged by merging surplus fields into the address.
  - ID prefix matches the file.
  - Duplicate IDs: keep the first and log.
  - Empty country: its own partition label `""`, processed normally.
  - S2/S3 countries absent from S1 produce no candidates.
- **Run report (JSON):** anomaly counts, per-partition sizes, timings and memory high-water marks. It feeds the Documentation template.

### 4.3 Stage 1 — Normalization

Every component below is rule-based or learned from provided data. Components marked **L** are learned artifacts that the pipeline regenerates on every run.

| Component | Detail | Evidence |
|---|---|---|
| Cleaning | Mojibake map (`Â\x80\x93`→`-`, `\x1a`→space…); control characters; NFKC; null tokens (`<NULL>`, `NULL`, `N/A`, `nan`, `none`); lowercase; diacritics stripped via NFKD (`Pàrents`→`parents`, `Çlub`→`club`); apostrophes *joined* (`de l'Alcazar`/`lAlcazar`→`lalcazar`); `&`→`and`; dotted initialisms collapsed (`S.A.S`→`sas`); leetspeak repair inside alphabetic tokens (`5ervices`→`services`) | E2, E3, France samples, error analysis |
| Romanizer | Brahmic scripts via **stdlib `unicodedata.name`**: consonant/vowel/virama/nukta/anusvara grammar, final-schwa deletion (kept after conjuncts), Tamil initial `ச`→s plus voicing plus `ஃப`→f, Malayalam chillu and `റ്റ`→tt. Any other script: letters derived from the Unicode name (e.g. `CYRILLIC … ZHE`→`zh`), otherwise raw characters, which still work in character n-grams | Verified on 9 scripts |
| **L** transliteration dictionary | Romanized token → Latin token, aligned on true train pairs (support ≥3, precision ≥0.6). Applied only to tokens from non-Latin names; the raw romanized token is kept as a variant | 713 entries; held-out token-set p50 **77→100**; test coverage **94.7%** |
| Names | Variants: split on `|`, `f/k/a`, `d/b/a`, `aka`, and `(ID: …)` with the domain extracted. Domain and handle flags. Honorifics (`mr`, `smt`, `m/s`, `shri`…; seed plus **L**). Legal form (seed EN/FR/IN plus **L** boundary-frequency detection). **Core** tokens with the legal form removed, a **spaceless** core for concatenations and domains, **skeleton** tokens, **initials** | R2-N8 |
| Addresses | Comma components. **Compound house numbers**: `2-16-133/6`→parts {2,16,133,6} plus the full token; `D-904`→(D, 904); `14 bis`→(14, bis); ranges `756-758`→{756, 758}; zero padding stripped; markers (`#`, `no`, `h.no`, `door no`, `plot no`, `n°`) removed. Street tokens, locality/city tokens (tail components), region tokens | R2-F5 |
| **L** abbreviation rules | Unsupervised, per partition, on the split being processed: substitutions in identical local contexts, kept if abbreviation-shaped (prefix, or subsequence anchored on the first letter), support ≥20, **the short form must be rarer than the long form**, and the pair must account for ≥5% of the short form's contexts. Applied as **soft expansion** (append the long form, never delete the original) | France: `r→rue`, `av→avenue`, `imp→impasse`, `b→bis`… US: `tx→texas`… Guards added after seeing `rue→route`, `no→nord`, `fl`→floor/Florida |
| **L** noise lexicons | (a) Insertion-noise tokens learned from train pairs (tokens often present in S2/S3 but absent from the true S1: `services`, `center`, `partners`, `smt`…). (b) Unsupervised per partition: tokens over-represented in S2/S3 names relative to S1 names (document-frequency ratio, ≥2k occurrences). **Measured with no labels:** on France it finds `participations` 1033×, `holding` 279×, `snc`, `dba`, `formerly`, `distribution`, `associes`, `developpement`, `groupe`; on US it finds `com`, `www`, `dba`, `aka`, `fka`, `formerly`, `lnc`, and the generator's injected `midtown`, `northside`, `eastgate`, `riverside`…. **Ordering requirement:** it must run *after* romanization and the dictionary. On India it otherwise flags every romanized token (`praivet` 15,687×), because S1 is all-Latin | R2-F6; `noise_probe.py` |
| Phonetic skeleton | Custom, rule-based, tuned for romanization variants: `ph→f`, `bh/b/w→v`, `sh→s`, `x→ks`, `z→j`, soft `c→s`, soft `g→j`, collapse doubled letters, drop non-initial vowels and h. Example: `praivet`/`private`→`prvt`, `phud`/`food`→`fd` | §1.3 |

### 4.4 Stage 2 — Candidate generation (blocking)

**Record representation.** Two L2-normalized sparse views, combined as a weighted sum (w = 0.5 each):
- **Name view:** 4-grams of the spaceless core, core tokens, skeleton tokens, and the conjunctive keys `skel(name token)×house-number` and `skel(name token)×locality token`.
- **Address view:** street and locality tokens, their skeletons, house numbers, and the conjunctive key `house-number×street token`.

Weighting is sublinear TF × IDF, with **IDF taken from the partition's S1**.

**Hashing.** A 64-bit stable key (two CRC32s) is **compacted per partition with `np.unique`**, which gives a collision-free vocabulary (R3-S7). Python's salted `hash()` is never used.

**Retrieval.**
- An inverted index over S1 with a **df-cap of 2,000**: stop-features are excluded from posting traversal but kept in the norms.
- The numba kernel is parallel over query blocks, uses a reusable dense accumulator plus a touched-list, and ends with a top-k partial sort.
- **k = 10.** A candidate is kept if its score is ≥ 0.5 × the query's best score (the best is always kept).

**Direction.** Only S2/S3 → S1, justified by exclusivity. Each S1's candidate list is the union over records that retrieved it.

**Measured recall and cost** (full-country train indexes, not samples):

| | recall@1 | @5 | @10 | @50 | queries/s |
|---|---|---|---|---|---|
| US (1.32M index; 3-grams, cap 2k) | 95.6% | 97.1% | 97.5% | 98.2% | 15.0k |
| India (883k index; 4-grams, cap 2k) | 94.5% | 97.0% | 97.6% | 98.7% | 10.3k |
| India, full 4.13M queries (probe) | — | 96.95% | — | — | 9.8k |
| US, full 6.19M queries (probe) | — | 98.22% | — | — | ~12k |
| Ratio pruning (score ≥ 0.5×best) | loses **0.01%** of positives, keeps 37.6% of candidates | | | | |

**Recall ceiling.**
- Pair-level: about **97.5–98.2%** at k=10. With the learned transliteration dictionary, India's full-country recall@5 rises from 96.95% to **97.79%**, and the India oracle rises from 0.9893 to **0.9929**.
- Entity-level oracle, i.e. the F0.5 a perfect matcher would get on our candidates: **0.9946 (US) / 0.9893 (India)** at k=5.
- The residual misses are dominated by **empty-address records whose names have many S1 twins**: US 86% and India 45% of misses@10. A perfect matcher could not assign these reliably either, so we accept this ceiling rather than widening k.

**Reduction ratio.** After pruning there are **1.88 (India) and 1.68 (US) candidates per S2/S3 record at k=5** (measured). At k=10 this is about 2, so roughly **20–25M test pairs**. The within-country cross product is about **6.7×10¹²** pairs (India 3.8×10¹², US 2.5×10¹², France 3.7×10¹¹). That gives **RR ≈ 0.999997**, or about 0.9999987 against the full 1.73M × 9.97M cross product.

**`candidate_pairs.tsv`** is written from exactly this pruned set: about 12–14 IDs per S1 row, about 0.3 GB. Stage-1 and stage-2 score *this* set and nothing else.

### 4.5 Stage 3 — Features (country-agnostic)

| Group | Features | Motivation (measured) |
|---|---|---|
| Retrieval context (7) | `ret_score`, `cos_name`, `cos_addr`, `ret_rank`, `ret_ratio` (score/best), `ret_gap` (best−score), `n_cands` | 74–78% of probe gain. Encodes competition among twins |
| Name similarity (10) | rapidfuzz `ratio`, `token_sort`, `token_set(core)`, `partial_ratio(spaceless)`, `ratio(spaceless)`, Jaro-Winkler(core), 4-gram Jaccard(core), skeleton Jaccard, IDF-weighted token cosine, **max over name variants** | Typos, reordering, domains, handles |
| Name structure (8) | Distinctive-token coverage (share of S1's IDF mass matched); **distinctive-token substitutions** (unmatched high-IDF tokens on both sides); **insertion-noise count**; honorific present; legal-form agreement (same / compatible / conflict / absent); initials match; domain/handle/concatenated flag; **pseudo-word flag** (core tokens absent from the partition's S1 vocabulary) | Sibling distractors (R2-F5), benign noise (R2-F6), trade names |
| Address & house number (12) | token_set, Jaccard, IDF-weighted overlap, street overlap, locality overlap; house number: full-token equal, all-parts equal, primary equal, any equal, prefix/suffix containment (`19-A-89`⊂`919-A-89`), **numeric distance** (`901` vs `904`), letter/sub-part agreement (`D1` vs `D12`, `/2` vs `/6`, `bis`); query-address-empty and S1-address-empty flags | 86% of FPs are number-nudged siblings |
| Ambiguity & co-location (7) | S1 name-twin count (log), twin candidates of the query, S1 rank-1 popularity, **address-twin count** (S1s sharing the candidate's exact address), **best name match among the query's co-located candidates**, **number of the query's candidates at the same address**, candidates with the same street+number | Twins make up 45–54% of S1. **Co-location is 33.5% in France vs 7% in train; +0.50–0.65 pts on co-located entities (R3-F11)** |
| Format (3) | Query source (S2/S3), query name all-uppercase, query name was non-Latin | Source-specific noise styles, which are not country-specific |
| Stage-2 collective (6) | `p1`, `p1_rank`, **`p1_margin`** vs the runner-up S1, `q_best_p1`, `s_support_others`, `s_nstrong_others` | +0.39–0.48 pts; `p1_margin` carries 26% of stage-2 gain |

**Explicitly excluded:** country, script *name* (only a generic "was non-Latin" flag), entity IDs and row order (no leakage, and never features).

**Engines.** `rapidfuzz.process.cpdist` runs the string metrics (C++, all cores). Numba kernels run set and number features over per-record token-id arrays. All features are float32, and the pipeline processes one country and one chunk at a time.

### 4.6 Stage 4 — Model

- **Library: LightGBM 4.7.0** (`lightgbm==4.7.0`, **MIT License**, Microsoft; verified in the installed package metadata), trained from scratch on the provided training data. No pretrained weights exist anywhere in the pipeline.
- **Stage-1:** binary objective, `num_leaves` 63–127, lr 0.05, up to 1,500 rounds with early stopping on the S1-grouped validation fold, `min_data_in_leaf` 200, feature and bagging fraction 0.8, `deterministic=True`, fixed seeds and thread count. 5-fold **out-of-fold** predictions are grouped by S1. Test predictions are the average of the fold models.
- **Stage-2:** the same library, with stage-1 features plus collective features computed from **out-of-fold** stage-1 scores. Without that, the collective features leak (R3-F8).
- **Size:** stage-1 is at most 5 × 1,500 trees × 127 leaves, about 0.95M leaf values plus a similar number of split thresholds. Stage-2 is comparable. The total is **<5M parameters, about 1/1,600 of the 8B cap**.
- **Robustness choices (measured, §3.4):**
  - All features are kept. Dropping "fragile" features cost 4.5–11.8 pts in-domain.
  - **No monotone constraints** (no transfer benefit).
  - **No self-training** (inconsistent across LOCO directions).
  - **Pooled US+India training.** It costs about 0.2 pts in-domain vs single-country training but covers both regimes.

### 4.7 Stage 5 — Decision strategy for F0.5

1. **Exclusivity (hard).** Each S2/S3 record may only be assigned to its **argmax** S1. Ties break on the retrieval score, then the S1 ID, so the result is deterministic.
2. **Threshold.**
   - **t\*** is chosen on the **shift-simulated validation** to maximize the exact macro F0.5. Among near-optimal values we pick the one that maximizes the **worst case** across scenarios (US-shift, India-shift and LOCO transfers). In every in-domain and shifted probe the curve is flat within about ±0.3 pts over t ∈ [0.6, 0.8], and t=0.7 was optimal.
   - **Then, per partition, a label-free cardinality-matched t_c** (§4.8), clamped to [t\*−0.15, t\*+0.25]. For test US and India it lands at about t\*, because their predicted cardinality already matches validation. For France it raises the threshold, countering the diagnosed over-merging. LOCO: +1.4 / +0.24 pts.
3. **Singletons are handled implicitly.** An S1 gets an empty list unless some record's best match clears its partition's t_c. Singleton loss was measured at **0.37 pts** (India), against 4.33 pts on non-singletons, so precision on multi-match entities is where the points are.
4. **Optional per-S1 expected-F0.5 subset selection** on isotonic-calibrated stage-2 scores. It measured +0.08 pts (no dictionary) and +0.03 pts (with the dictionary), i.e. within noise, so it is adopted only if it wins on the shifted validation *and* in both LOCO directions.
5. **Rejected:** an ambiguity-margin abstention rule (measured +0.00 to +0.01 pts, because `p1_margin` inside stage-2 already captures it). Per-country thresholds are also rejected: they need labels, and France has none.

### 4.8 Unseen-country strategy (France and anything else)

**Principle: no country branching.** The code never tests a country name. Every country-specific behaviour comes from one of three sources:
1. Generic, script-agnostic rules.
2. Statistics and lexicons learned **without labels from that partition's own records**.
3. Vocabulary-level resources learned from labelled train data.

The matcher is pooled across countries. A new label ("Germany", "", "XX") takes exactly the same path as France.

**Layer 1: input-side adaptation (unsupervised, per partition). Each item was measured on France, with no labels:**

| Mechanism | What it did on France |
|---|---|
| IDF from the partition's own S1 | Templated tokens (`sarl` 73k, `club` 23k, `amicale`…) are automatically down-weighted in blocking and features |
| Learned abbreviation rules | `r→rue` (121,592 contexts), `av/ave→avenue`, `imp→impasse`, `bd/blvd→boulevard`, `all→allee`, `b→bis`, `t→ter`, `st→saint`, plus typo repairs (`avnue→avenue`, `impase→impasse`) |
| Learned injected-noise lexicon | `participations` (1033×), `holding` (279×), `groupe`, `dba`, `formerly`, `distribution`, `associes`, `developpement` |
| Twin counts and pseudo-word vocabulary | Computed on France S1 (45.8% twins, largest group 493) |
| Blocking | Ran on France at 10.2k q/s; top-1 score quantiles 0.27/0.37/0.52/0.72/0.82, at least as high as US (0.23/0.35/0.49/0.70/0.82) |

**Layer 2: a matcher that transfers.**
- **Pooled US+India training.** The model sees two very different regimes, at a cost of only about 0.2 pts in-domain (LOCO table).
- **The learned dictionary removes the script gap.** India +1.0 pt (0.9530→0.9630). In LOCO it lifted US→India from 0.8507 to 0.8992. About half of the unseen-country penalty was a script problem, which France (Latin script) doesn't have.
- **Remedies that measurably *failed* are not used:** dropping features, monotone constraints, and self-training on the reduced set (R3-F9 round A).
- **Co-location features** (R3-F11) target the diagnosed France mechanism: +0.50 / +0.65 pts on co-located entities in-domain. In France, co-location affects **33.5%** of entities vs about 7% in train.
- Also measured and rejected: per-country percentile normalization and full-feature self-training. Each helps one LOCO direction and hurts the other.

**Layer 3: label-free guardrails at inference.** These are run for every partition, and are the France gate:
1. **Cardinality check.** Predicted matches per S1 and the singleton rate are compared with (a) the train prior (3.46 / 5.6%, identical across train countries) and (b) the other test partitions under the same model. Probe values: France 3.55 / 4.8%, test India 3.22 / 6.2%, test US 3.33 / 5.7%. **France over-predicts.** 
   **Response: a cardinality-matched threshold (adopted, label-free).**
   - *Rule:* for each partition, choose t_c so that predicted matches per S1 equals the reference value the same model produces on labelled validation at t\* (about 3.2–3.3).
   - *Justification:* cardinality is a property of the data generator and is measured to be country-invariant (3.462 vs 3.461 mean; 5.57% vs 5.59% singletons). Test India and US already reproduce their validation values (3.22 / 3.33 vs 3.19 / 3.28), so the rule leaves them at about t\*. France (3.55) gets a higher, more conservative threshold.
   - *Evidence:* in LOCO it recovers 73% / 89% of the oracle-threshold gain (US→India 0.8992→0.9131; India→US 0.9539→0.9563).
   - *Guard:* t_c is clamped to [t\*−0.15, t\*+0.25] and logged. It applies uniformly to every partition, with no country branching.
2. **Per-source caps.** Train maxima are ≤5 S2 and ≤6 S3 matches per S1. An S1 exceeding them is almost certainly over-merged, and its violations are counted and inspected.
3. **Manual spot-check** of 50 random France S1 decisions and of the 50 highest-scoring matches with low name similarity. These are the most suspicious "address-only" merges, given the multi-business addresses seen in France.

### 4.9 Validation methodology & honest F0.5 estimate

**Holdout design**

1. **Split by S1 entity, never by pair.** Train S1 entities are assigned to 5 folds by a stable hash of the ID. **Fold 0 (20%, about 441k entities) is the holdout.** Folds 1–4 train stage-1 with 4-fold OOF and stage-2 on the OOF collective features. Neither stage ever sees fold 0.
2. **Two index conditions:**
   - (i) *natural*: all train S1 in the index;
   - (ii) *shift-simulated*: per country, drop dᶜ ≈ 19% of S1 from the index so their matches become distractors (E6). The dropped set is disjoint from fold 0.
   Training candidates and the threshold use (ii). Both conditions are reported.
3. **Metric:** the exact macro F0.5 from the README, over holdout S1 entities including singletons. **Blocking misses count as false negatives** (true counts come from ground truth, not from candidates). The implementation is unit-tested against the README example (0.714, verified) and edge cases.
4. **Unseen-category simulation:** LOCO in both directions, US→India and India→US, run with the *full* final stack (remedies included). It is used to (a) choose features, constraints and adaptation, (b) choose t\* by the worst-case criterion, and (c) bound the France penalty.
5. **Uncertainty:** 1,000 bootstrap resamples over holdout S1 entities give 95% CIs (about ±0.0005 at 441k entities; ±0.002 at the probes' 30k).
6. **Error analysis:**
   - FP types: sibling distractor, other-S1 confusion, singleton FP.
   - FN types: blocking miss, below threshold, lost argmax.
   - Slices: script, empty address, domain/handle, pseudo-word, twin-group size, cardinality.
   These go into the Documentation template's "common false positives/negatives".
7. **France health checks (no labels):**
   - Predicted cardinality histogram vs the train prior.
   - Best-p distribution.
   - Share of assigned records.
   - Top-1 retrieval-score quantiles.
   Each is compared against test US and India under the same model. A hand spot-check of 50 random France S1 decisions follows. These are a **gate**, not a tuning signal.

**Honest estimate**

| Country (share of test S1) | Measured basis | Adjustments | Estimate |
|---|---|---|---|
| US (38.3%) | 0.9740 in-domain (stage-1, 28 probe features) | Stage-2 +0.3–0.4 (measured on India), distractor shift −0.5, planned features +0.1–0.3 | **0.972** (0.968–0.977) |
| India (46.8%) | **0.9663** in-domain (dictionary + stage-2) | Distractor shift −0.5, planned features (sibling-number, noise lexicons, co-location) +0.1–0.3 | **0.963** (0.957–0.968) |
| France (15.0%) | No labels. Transfer residual after all label-free remedies: −1.8 pts (similar regime) to −5.0 pts (dissimilar) below a US-like in-domain ~0.97 | Co-location features aimed at the diagnosed mechanism; cardinality-matched threshold corrects the over-merging seen in diagnostics | **0.93** (0.90–0.95) |
| **Overall** (weighted by test S1 share) | | | **≈ 0.96** (0.953–0.969) |

**How to read this estimate:**
- It is *conservative in-domain*: the probes used 28 of the ~53 planned features and trained on 60k of ~1.4M available training entities.
- It is *genuinely uncertain for France*: the France range alone moves the overall score by about −0.005 / +0.003. The rest of the overall range comes from US/India feature and shift uncertainty.
- In Phase 4 the US/India numbers are re-measured on the full holdout, and France's entry is replaced by the LOCO-with-final-stack residual plus the diagnostics gate.

### 4.10 Scalability plan

| Stage | Work (test) | Measured basis | Est. time (test / train) | Peak memory |
|---|---|---|---|---|
| Ingest + normalize | 11.7M records (+12.5M train) | 110–200k records/s featurization with 10 processes | ~2 min / ~2 min | < 3 GB (polars/Arrow) |
| Index build | per country, ≤0.81M S1 | ~10 s per country | < 1 min | ~1.5 GB per country |
| Retrieval | 9.97M records | 9.8–15k q/s (full-country probes) | **~15–18 min / ~16 min** (+~13 min for the shifted train index) | ~2 GB (queries streamed in chunks of 400k) |
| Pair features | ~20–25M pairs | probe: 8.65M pairs × 28 features in ~110 s; ~50 features ≈ 2× cost | **~8–12 min / ~12 min** (train subsampled by S1) | Chunks of ≤5M pairs × 50 × 4 B ≈ 1 GB, spilled to parquet |
| LightGBM stage-1 (5-fold) + stage-2 | ~15–20M training rows | probe: 5.8M rows × 28 features × 500 rounds in ~40 s | **~20–30 min** (train only) | ~5 GB (binned) |
| Predict + decide + write | ~20–25M pairs | ~1M rows/s | ~2 min | < 3 GB |
| **Total** | | | **~45 min inference-only · ~1.5–2 h full train+test** | **target ≤ 12 GB** (the probes hit 10–15 GB because of Python lists; fixed by numpy-only featurization and no whole-matrix copies) |

Other safeguards:
- **Nothing is O(n·m).** The largest operation is inverted-index traversal, capped per feature.
- **Determinism:** stable hashing, fixed seeds, `deterministic=True`, sorted outputs. Two runs produce byte-identical TSVs, and this is checked in the final phase.
- **Resumability:** each stage writes parquet checkpoints (normalized records, candidates, features, predictions), so a failure costs only that stage.

### 4.11 Compliance & fair-play audit (CMP gate)

| Check | Status |
|---|---|
| External data, APIs, geocoding, downloads at runtime | **None.** Inputs are only the 7 provided TSVs. No network access in code. Romanization uses Python's stdlib `unicodedata` tables. No gazetteers: state codes, street abbreviations and transliterations are *learned from the provided files* |
| Model license and size | LightGBM 4.7.0, **MIT**; <5M parameters (cap 8B) |
| Classical-only brief | No embeddings (including no LSA/SVD dense vectors), no transformers, no pretrained networks, no neural networks at all. Sparse TF-IDF, hashing, string metrics, rule-based phonetics and transliteration, GBDT |
| Dependency licenses (verified from installed package metadata) | lightgbm 4.7.0 **MIT** · polars 1.44.2 MIT · pyarrow 25.0.1 Apache-2.0 · numpy 2.5.3 BSD-3 (+0BSD/MIT/Zlib parts) · scipy 1.18.1 BSD · scikit-learn 1.9.1 BSD-3 (isotonic calibration only) · rapidfuzz 3.14.6 MIT · numba 0.67.0 BSD · llvmlite 0.49.0 BSD-2 + Apache-2.0 with LLVM exception · psutil 7.2.2 BSD-3. **No GPL or LGPL.** `unidecode` and `python-Levenshtein` (GPL) are deliberately excluded; pandas and jellyfish are not needed |
| Hand-written resources | Small seed lexicons only: legal forms, honorifics and common English/French abbreviations, about 150 tokens of general language knowledge and not business-identity data. Everything else is learned. Seed-lexicon ablation is reported (§4.14 P4) |
| ⚠ Use of *unlabelled test* inputs | IDF, abbreviation rules, noise lexicons, twin counts and co-location counts, plus the cardinality-matched threshold (which reads only the distribution of the model's own predictions), are computed from the test files being resolved. No labels exist or are used. This is provided data, not external data, and indexing the test corpus is inherent to blocking. **My reading is that this is allowed. It is flagged for your confirmation (§5).** |
| Reproducibility | `requirements.txt` pins exact versions (Python 3.12). The README gives the exact commands. Every learned artifact is regenerated by the code. A clean-room rerun in a fresh venv is the final gate |

### 4.12 Risk register

| # | Risk | Likelihood / impact | Mitigation | Residual |
|---|---|---|---|---|
| K1 | **The matcher transfers poorly to France.** The diagnosed mechanism is **co-location** (33.5% vs 7%), which causes over-merging (predicted 3.55 matches per S1 vs about 3.3 expected) | High / High (15% of score) | Address-twin features (+0.5–0.65 pts on co-located entities in-domain); pooled training; §4.8 guardrails and gate; the dictionary; the cardinality-matched threshold (LOCO: recovers 73–89% of the oracle-threshold gain) | Largest uncertainty; quantified in §4.9 |
| K2 | The test distractor ratio exceeds train (5.5–5.8 vs 4.67 records per S1) | High / Low–Medium | Shift-simulated training and threshold choice (measured −0.5 pts; t\* unchanged) | Low |
| K3 | Sibling distractors (number-nudged, token-swapped) | Certain / Medium (86% of FPs) | Compound house-number and distinctive-substitution features; stage-2 margin | Some irreducible ambiguity |
| K4 | Empty-address records with twin names | Certain / Low | Ambiguity features → abstain; accepted recall ceiling | ~1–2% of pairs |
| K5 | Wrong learned rules (abbreviations or dictionary) | Medium / Low | Frequency, support and shape guards; soft expansion; the dictionary only touches non-Latin tokens; ablations | Low |
| K6 | Memory spike on the 24 GB machine | Medium / High (crash) | Numpy-only featurization, per-country and chunked streaming, checkpoints, psutil high-water logging, a guard to shrink chunk size | Low |
| K7 | Train/serve skew (index size, statistics) | Medium / Medium | Per-partition statistics from the processed split; relative features; train candidates generated under the same shift conditions | Low |
| K8 | Stage-2 leakage | Low with OOF / High | Strict 5-fold OOF by S1; validation fold never seen by either stage | Low |
| K9 | Public/private leaderboard variance | Certain / Low | Don't tune on the public leaderboard; bootstrap CIs internally (±0.002 at 30k entities) | Low |
| K10 | Unknown scripts or labels beyond France | Low / Medium | Script-agnostic romanizer fallback; country as an open partition; empty or unknown country handled | Low |
| K11 | Format rejection | Low / Fatal | A custom writer, unit tests (no `\r`, no quotes, row counts), the validator with `--check-ids`, and a subset assertion | Very low |
| K12 | Compliance misread (unlabelled-test statistics, seed lexicons) | Low / Fatal | Transparent documentation; your confirmation requested (§5); both can be switched off by config (with a measured accuracy cost) | Depends on your call |

### 4.13 Mapping to required deliverables

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        ← [5]+[6]; 1,732,544 rows; validator --check-ids PASS
│   └── candidate_pairs.tsv         ← exact pruned pair set scored by stage-1/2 (~20–25M pairs)
├── code/business_entity_resolution/
│   ├── src/ber/
│   │   ├── io.py            ingest, unquote, schema checks, TSV writers
│   │   ├── text/            clean.py · romanize.py · phonetic.py · names.py · addresses.py · lexicons.py (seed lists)
│   │   ├── learn/           translit_dict.py · abbreviations.py · noise_lexicon.py   (generate learned artifacts)
│   │   ├── blocking/        featurize.py · index.py (numba kernel) · retrieve.py
│   │   ├── features/        pairwise.py · numbers.py · context.py · collective.py
│   │   ├── model/           train.py (OOF, stage-1/2) · predict.py
│   │   ├── decide.py        exclusivity + threshold (+ optional expected-F)
│   │   ├── evaluate.py      exact macro-F0.5, bootstrap CIs, LOCO, shift simulation, diagnostics, error analysis
│   │   └── pipeline.py      CLI: `python -m ber.pipeline {train,predict,all,validate}`
│   ├── tests/               metric (README example = 0.714), unquote, romanizer, parsers, writer contract
│   ├── README.md            exact reproduction steps (data → blocking → matching → output) + runtime/memory
│   └── requirements.txt     pinned (Python 3.12.13)
└── Documentation_template.md       filled with real validation numbers, ablations, error analysis
```

### 4.14 Phased implementation checklist (to be followed after approval)

Each phase ends with a **gate**: its tests pass and its measured numbers match or beat the probe baseline. If a gate fails, I stop and report instead of pressing on.

- [ ] **P0 — Scaffold & contracts.**
  - Package layout, pinned `requirements.txt`, config (YAML: k, cap, weights, thresholds), logging, memory high-water logging.
  - `evaluate.py` metric with unit tests: the README example must give 0.714, plus singleton and empty edge cases.
  - Loader with unquote tests using the E1 rows; writer contract tests; validator hook.
  - *Gate:* tests green; a dummy all-empty submission passes the validator.
- [ ] **P1 — Normalization & learned resources.**
  - Cleaning, romanizer, skeleton, name and address parsers.
  - Learners: transliteration dictionary (train pairs only), abbreviations (per partition, guarded), noise lexicons.
  - Unit tests from the profiling hard cases (§1.3 examples).
  - *Gate:* held-out Indic token-set p50 ≥ 95; learned-rule precision spot-check (for US, compare with known state and street codes).
- [ ] **P2 — Blocking.**
  - 64-bit hashing with compaction, index, numba search, pruning.
  - Run on train (natural and shifted) and on test.
  - *Gate:* recall@10 ≥ 97.5% (US) and ≥ 97.5% (India) on held-out entities; throughput ≥ 8k q/s; peak RSS ≤ 12 GB.
- [ ] **P3 — Features.**
  - All groups from §4.5, via cpdist and numba, chunked to parquet.
  - *Gate:* feature parity tests; stage-1 on probe-equivalent features reproduces probe F0.5 within ±0.002.
- [ ] **P4 — Models & robustness.**
  - Stage-1 5-fold OOF, then stage-2.
  - Ablations: seed lexicons on/off, learned abbreviations on/off, co-location features on/off (co-located slice), cardinality-matched vs fixed threshold (LOCO), k ∈ {10, 20}, the expected-F variant, optional Double-Metaphone features.
  - LOCO both directions; shift simulation.
  - *Gate:* holdout F0.5 under shift ≥ **0.958 (India)** and ≥ **0.968 (US)**, i.e. no worse than the shift-adjusted probe results. LOCO residual no worse than round B (US→India ≥ 0.913, India→US ≥ 0.956).
- [ ] **P5 — Decision & validation report.**
  - Choose t\* with the robust criterion; evaluate the expected-F variant.
  - Bootstrap CIs; FP/FN error analysis by category.
  - *Gate:* a report with numbers for §4.9.
- [ ] **P6 — Test inference.**
  - Run all three countries.
  - France diagnostics: predicted cardinality vs prior, score distributions, unassigned-record rate, the cardinality-matched t_c per partition (logged with the clamp), and per-source cap violations.
  - Write outputs; validator `--check-ids`; subset assertion; spot-check 50 France matches by hand.
  - *Gate:* validator PASS; diagnostics within tolerance.
- [ ] **P7 — Package & reproduce.**
  - README, filled `Documentation_template.md` (real numbers, ablations, error analysis), zip.
  - **Clean-room rerun** in a fresh venv from raw data, with byte-identical outputs.
  - *Gate:* identical hashes; runtime and memory recorded.

---

## 5. Decisions I made that you may want to override

The provided documents don't settle these explicitly. I picked a default for each rather than stopping. Every one is a **config switch**, and its accuracy cost will be measured in P4/P5, so changing your mind later is cheap.

1. **Using unlabelled *test* inputs for unsupervised statistics** (IDF, twin counts, learned abbreviation and noise lexicons), plus the **cardinality-matched threshold** (label-free; it reads only the model's own predicted-match counts per partition). Self-training on test was tested and *rejected*, so no model is ever trained on test inputs.
   - *Default: ON.*
   - The rules ban *external* data. The test files are provided data, no labels exist or are used, and blocking must index the test corpus anyway.
   - Without this, France cannot be normalized at all, because it has no training data.
   - **Please confirm you're comfortable with this reading.**
2. **Small hand-written seed lexicons** (about 150 tokens: English/French legal forms, honorifics, common street abbreviations).
   - *Default: ON, with an ablation reported.*
   - This is general language knowledge, not business-identity data, and the learned rules cover most of it anyway.
3. **"No embeddings" read strictly.** It also excludes LSA/SVD dense vectors, not only neural embeddings. *Default: strict.*
4. **Tooling:** Python 3.12.13 in a project-local uv venv (`.venv/`), pinned requirements. The code lives in `code/business_entity_resolution/` inside this project, and a script builds the zip.
5. **Compute budget:** about 1.5–2 h for a full train+test run on this laptop, and about 45 min for test-only inference once models exist.

---

## Appendix A — Probe evidence index

These are throwaway measurement scripts, not the pipeline. They are copied to `design_probes/` for inspection. The implementation will be written cleanly per §4.13, and none of this code ships in the submission.

| Probe | Question it answered | Key result |
|---|---|---|
| `raw_check.py` | Are the TSVs structurally sound? | 4 fields in every row; 813 CSV-quoted test rows; ~730 rows with control characters; no CR, no invalid UTF-8 |
| `profile1.py` | Sizes, nulls, scripts, patterns per file | §1.3 file table; 9 Indic scripts; PIN ≤0.8% |
| `profile2.py` | Ground-truth structure and leakage | Mean 3.46 matches, max 11, 5.58% singletons, **exclusivity 0 violations**, 26% distractors, no leakage |
| `profile3.py`, `profile5.py` | What do match groups and zero-overlap pairs look like? | Noise taxonomy (§1.3); 8.6% zero name overlap; vocabulary coverage 99.6% |
| `profile4.py` | Name ambiguity and distractor similarity | Twins 45–54%; 20.6% of distractors share an S1 name |
| `translit.py`, `probe_lib.py` | Can stdlib-only romanization work for 9 scripts? | Yes, with script-specific fixes (R2-N6) |
| `norm_probe.py` | Name separability by script | Indic p50 0.59 vs random p99 0.53 (before the dictionary) |
| `dict_probe.py`, `dict_build.py` | A dictionary learned from train pairs | 713 entries; held-out p50 77→100; test coverage 94.7%; leakage-free rebuild identical in size |
| `noise_probe.py` | Can injected noise tokens be found without labels? | Yes: France `participations`, `holding`, `groupe`…; US `dba`, `aka`, `midtown`…; must run after the dictionary for Indic |
| `abbr_probe.py` | Unsupervised abbreviation learning on any country | France `r→rue` (121k)…; US state codes; failure modes → guards |
| `probe_block.py`, `probe_block2.py` | Blocking recall vs cost per representation | Name-only 51–64% @10; joint 88–95%; **conjunctive keys 96.8–97.6% @10 at 10–15k q/s** |
| `probe_model.py` | End-to-end F0.5 per country, shift, dictionary | US 0.9737; India 0.9530; India-shift 0.9479; **India + dictionary 0.9630** (recall@5 97.79%, oracle 0.9929) |
| `err_analysis.py` | Where are the FPs and FNs? | 86% of FPs are sibling distractors; 65% of FNs are below threshold (benign noise) |
| `stack_probe.py` | Stage-2 and expected-F? | Stage-2 +0.48 pts (no dictionary) / **+0.39 pts (with dictionary: 0.9624→0.9663)**; expected-F +0.08 / +0.03, within noise |
| `loco.py`, `loco2.py` | Unseen-country penalty and remedies | US→India 0.8507, India→US 0.9572; round A remedies (drop features / monotone / self-training) all hurt: 0.816–0.822 / 0.912–0.913 |
| `loco3.py` | Round B remedies, with dictionary | US→India 0.8992 → **0.9131** with the cardinality-matched threshold; India→US 0.9539 → **0.9563**; percentile normalization and self-training inconsistent |
| `coloc_probe.py` | Why does France over-merge? Does a feature fix it? | France co-location 33.5% vs 5.5–7.5%; co-located entities score about 1 pt lower; address-twin features give +0.50/+0.65 pts on them |
| `probe_apply.py` | Behaviour on unlabelled test (France) | Predicted matches per S1: France **3.55**, test India 3.22, test US 3.33 (train-val 3.19 and 3.28). Extra test records are distractors; France over-merges |
