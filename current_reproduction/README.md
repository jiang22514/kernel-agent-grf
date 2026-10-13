# Current paper: unified reproduction guide

All current materials are available from the repository's root URL:

https://github.com/jiang22514/kernel-agent-grf

The dated directories preserve published records while this page provides one navigation point for the current paper. The engineering application is the public **Leendert de Boerspolder 29-CPT section**, not the earlier private CPT development case.

## Start with the archived results

From the repository root:

```text
python -B current_reproduction/verify_current.py
```

Python 3.10+ and its standard library are sufficient. The command checks the new public-case file manifest and all original data/MATLAB package hashes, reproduces the six public-site search histories, verifies the existing synthetic release, and replays both the paired-feedback experiment and random-continuation comparison. It writes a combined report to `recomputed/current_reproduction/summary.json`. No API, MATLAB, GP refitting or random-field simulation is invoked.

## Follow the paper

| Paper component | Data, code and records |
|---|---|
| Agent method and complete prompts | [Public-site prompts](../public_case_2026_10/public_search/prompts/), [prompt construction and acceptance functions](../public_case_2026_10/public_search/source/), [synthetic prompts and optional live-search implementation](../research_release_2026_10/README.md) |
| Six independent synthetic cases H01–H06 | [Observations, generating mechanisms, fitted candidate scores and archived searches](../research_release_2026_10/README.md) |
| Paired numerical-feedback experiment | [Full archived requests, replies and offline replay](../feedback_followup_2026_10_10/feedback_ablation/records/README.md) |
| Random-continuation comparison | [Score tables, accepted access sequences and standard-library reproduction](../feedback_followup_2026_10_10/continuation_public/README.md) |
| Public Dutch CPT source, selection and preprocessing | [Portable reconstruction and exact frozen subsets](../public_case_2026_10/data_public_staging/data_README.md) |
| Public-site agent comparison | [Six archived searches and all 376 training-candidate scores](../public_case_2026_10/public_search/README.md) |
| Prediction and conditional simulation | [Saved models, MATLAB entry points and exact GPML dependencies](../public_case_2026_10/matlab_public_package/README.md) |
| Engineering plots, training density and low-resistance area summaries | [Numerical source CSVs and figure PDFs](../public_case_2026_10/figures/) |

The two preassigned first public-site searches recover the same champion. Repetitions, synthetic tests and the paired-feedback comparison remain separately identified in their records. The complete candidate score table is released to reproduce search performance; archived requests show that agents saw only previously accessed scores during each search.

## Optional numerical work

The public-case guide explains data reconstruction from the CC0 source archive. The MATLAB package provides saved-model predictions, exact conditional field generation, and explicitly requested selected-structure refits. Those operations require the listed numerical dependencies; the quick command above verifies the archived evidence without repeating them. New paid searches are not required for reproduction and are never started by the verification command.

The current engineering figure CSVs are in `public_case_2026_10/figures/`; drawing examples retained in the older releases can reflect earlier figure styling. The published numerical values and archived trajectories are unchanged.
