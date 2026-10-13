# Public engineering case: Leendert de Boerspolder

This directory contains the current paper's public Dutch CPT application: raw public data and fixed subsets, six complete agent searches, fitted models, portable numerical code, and figure source data.

## Study and data

The source is Tom de Gast, Michael A. Hicks and Philip Vardon (2020), [Cone Penetration Test (CPT) dataset to study soil heterogeneity](https://doi.org/10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6), 4TU.ResearchData, CC0 1.0. Its unchanged 32.9 MB source archive and license metadata are included in `data_public_staging/source/`.

The two-dimensional section uses 29 crest CPTs spanning approximately 50 m. The main fit uses 696 observations from 12 training CPTs; 638 observations from 11 entire unseen CPTs assess prediction. Nested 6/12/18-CPT training subsets and a finer vertical-sampling case share the fixed test identities. The fitted response is `ln(qc / 1 MPa)`; physical cone resistance remains available.

## Reproduction map

1. **Data preparation:** [data README](data_public_staging/data_README.md). Rebuild all four frozen cases from the source archive and verify the numerical arrays with NumPy and SciPy.
2. **Agent searches:** [search README](public_search/README.md). Replay all six searches with standard-library Python. Complete fixed prompts, actual visible requests/replies, model settings, accepted candidates, training scores and usage records are included. All six searches reached an AIC gap at most two; both preassigned first runs found the fitted-library champion.
3. **Prediction and conditional fields:** [MATLAB README](matlab_public_package/README.md). Load the selected saved parameters, predict at unseen CPTs, generate conditional latent fields, or explicitly refit selected structures. The package includes the required GPML runtime files under their own license. It does not automatically repeat all 376 optimizations.
4. **Figure source data:** [figures](figures/). CSVs provide search curves, CPT geometry and profiles, held-out metrics, training-density results, field-map quantities and realization-level low-resistance area fractions; PDFs show the corresponding engineering plots.

From the repository root, `python -B current_reproduction/verify_current.py` checks this package together with the synthetic and feedback evidence. For only the six public-site searches, run `python -B public_case_2026_10/public_search/verify_public_search.py`.

`PUBLICATION_MANIFEST.json` identifies every supplied public-case file by SHA-256. The original subpackage manifests and the prior relocated-MATLAB validation record are retained unchanged. The latter checked saved-model predictions plus three conditional draws per model on a small grid; it is not a fresh rerun of the complete ensemble or timing experiments.

Project code follows the repository MIT license; GPML keeps its separate license. CC0 applies to the source dataset. Private development CPT observations, credentials, hidden provider reasoning and the authors' local experimental caches are excluded.
