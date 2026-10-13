# Kernel-agent geotechnical random fields

Code, data and reproducibility materials for **An LLM agent framework for composite covariance discovery and exact conditional simulation of geotechnical random fields**.

**Start here: [current reproduction guide](current_reproduction/README.md).** This repository is the single entry point for the current paper: the public Dutch engineering case, synthetic tests, complete prompt templates, archived agent searches, feedback experiments, fitted models, conditional-simulation code and numerical figure data.

## What the framework does

An LLM agent selects explicit covariance–mean combinations using data summaries and numerical AIC feedback, evaluating 20 of 376 candidates. A coordinate-union sampler then carries the selected covariance into joint conditional random fields through independent component sampling and a joint Matheron update.

- **Synthetic tests:** on six independent cases, Astra structured search and Fable competing-structure search each reached near-optimal models in all six cases. A paired feedback experiment reached 6/6 for both combinations, compared with 2/6 and 3/6 when scores were hidden.
- **Public engineering case:** the Leendert de Boerspolder dataset provides 29 closely spaced crest CPTs. The main model uses 696 observations from 12 training CPTs, and predictive assessment uses 638 observations from 11 unseen CPTs. All six searches reached within two AIC units of the fitted-library minimum; both preassigned first runs recovered the champion, `(MA5*PER)+MA1` with a quadratic mean.
- **Spatial assessment:** the selected composite improves AIC by 33.3 over the best single kernel and achieves 89.5% coverage for nominal 90% prediction intervals. Its conditional ensembles quantify low-resistance extent. A matched 500-field benchmark at 11,615 query locations gives a 13.23-fold simulation speedup over dense posterior simulation, including preparation and conditioning.

## Reproduction materials

| Material | Start here |
|---|---|
| Public CPT source data, fixed training/test split and portable preparation | [Public case](public_case_2026_10/README.md) |
| All six public-case agent searches, prompts, visited candidates and offline replay | [Search records](public_case_2026_10/public_search/README.md) |
| Saved models, MATLAB/GPML prediction, conditional simulation and selected-structure refitting | [Numerical package](public_case_2026_10/matlab_public_package/README.md) |
| Numerical CSVs and plots for the public engineering case | [Figure source data](public_case_2026_10/figures/) |
| Six independent synthetic cases, generators, complete prompts and search evidence | [Synthetic research release](research_release_2026_10/README.md) |
| Paired score-feedback experiment and random-continuation comparison | [Feedback and continuation materials](feedback_followup_2026_10_10/README.md) |

## Check the archived results without new model calls

From the repository root, using Python 3.10 or later:

```text
python -B current_reproduction/verify_current.py
```

This standard-library check verifies the public-case package hashes, replays the six public-site searches, and checks the existing synthetic and feedback evidence. It makes no network calls, fits no GP and does not require MATLAB. Outputs are saved under ignored `recomputed/current_reproduction/`. The guide distinguishes these quick checks from optional data reconstruction, model refitting and new field generation.

## Data and licenses

The public CPT dataset is by Tom de Gast, Michael A. Hicks and Philip Vardon (2020): [Cone Penetration Test (CPT) dataset to study soil heterogeneity](https://doi.org/10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6), distributed by 4TU.ResearchData under CC0 1.0. The unchanged source archive, attribution and fixed study subsets are included. The site is Leendert de Boerspolder near Leiden; “Delft” in program names identifies the source institution.

Project code uses the existing [MIT license](LICENSE); bundled GPML dependency files retain their separate FreeBSD license and notices. No MATLAB runtime or license is included. The earlier private CPT observations are not distributed. Their aggregate development records remain in the historical release; the current engineering case uses the public Dutch data above.

## Earlier versions

The dated releases and root-level `selection/`, `agent/`, `simulation/`, `diagnostics/` and `results/` directories remain available so older references continue to resolve. The [earlier README](README_EAAI_legacy.md) describes the original submission. Use the [current guide](current_reproduction/README.md) to identify which data and experiments support the current paper.
