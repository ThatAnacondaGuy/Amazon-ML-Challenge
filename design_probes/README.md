# Design probes (throwaway measurement scripts — NOT the pipeline)

These scripts produced every measured number in `../SOLUTION_PLAN.md` (see its Appendix A for the question each one
answered and the result). They are intentionally quick-and-dirty: hard-coded paths, no tests, Python-object-heavy
memory use (10–15 GB peaks). None of this code ships in the submission; the pipeline is re-implemented cleanly
under `code/business_entity_resolution/` after the plan is approved.

Run with the project venv, e.g.:

    .venv/bin/python design_probes/profile2.py
    NG=4 CONJ=1 WA=0.5 CAPS=2000,5000 .venv/bin/python design_probes/probe_block2.py India 60000 20000
    DROP=0 .venv/bin/python design_probes/probe_model.py India 60000 30000 5

Note: `load.py` hard-codes the dataset path (`BASE`), and several scripts write intermediate `.npz`/`.log`
files next to themselves.
