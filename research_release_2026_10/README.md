# Compositional covariance search and exact conditional random fields

Public research release, 8 October 2026. This directory is self-contained and can be moved or placed under `research_release/` in another repository. It contains current LIN-v2 numerical methods, both frozen search policies, synthetic observations and search-score evidence. It does not include the manuscript or private CPT observations.

## Reproduce the reported search results without an API

From this directory:

```text
python -B tools/verify_release.py
```

This uses only the Python standard library. It verifies file hashes, all visited candidate scores, unique-candidate budgets, terminal states, success counts and actual costs. It neither fits a GP nor calls an API. `verification.json` contains recomputed results by model, method and phase.

The original **structured panel** contains 80 planned searches (75 completed), including the 10-model/50-case panel, the repeated flagship panel and the common-initial-candidate comparison. Its complete safe visits, planned model settings, initial candidates, static prompts and mechanical/random reference summaries are in `results/structured_panel/` and `prompts/`. The **development** experiment contains 50 additional searches on EX1–EX4 and CPT: screening (three methods) followed by new calls on the same five cases (control and the selected revised strategy). The screening cases helped choose the method and are not an independent test set. The **holdout** experiment contains 18 searches on six newly generated cases H01–H06: Astra control and revised strategy on each case, plus Fable 5.1 revised strategy. The latter has no Fable control arm.

Each search can access at most 20 different candidates in the common library of 94 kernel expressions × four mean functions. The control uses four batches of five; the revised strategy uses five batches of four and retains three distinct promising kernel structures. An exact hit has final AIC gap ≤10⁻⁹; a near-optimal result has gap ≤2 and includes exact hits. Failed or unfinished searches must remain in the planned denominator.

## Synthetic observations and generators

Install `numpy`, `scipy` and `matplotlib` in a project environment to use the optional scientific tools:

```text
python -m pip install -r requirements.txt
python -B tools/reproduce_synthetic.py
python -B tools/plot_search_results.py
```

`reproduce_synthetic.py` regenerates H01–H06 from their fixed seeds and compares every coordinate and observation against released CSVs. `--write` writes regenerated files under `recomputed/`. The six generating mechanisms and parameters are in `data/holdout/generation_truth.json`. They were not sent to search models. H01–H04 are mechanisms represented by the library; H05–H06 contain warping/amplitude modulation or spatially varying mixture structure outside the library. The best finite-sample AIC model need not equal the generating model.

EX1–EX4 original observations are in `data/development/EX*.csv`. Their MATLAB generator is `code/matlab/experiments/generate_development_cases.m` with `rng(1)`. EX4 is three-dimensional; the other development synthetic cases are two-dimensional. Their released point-prediction comparisons are synthetic-only files in `results/synthetic_prediction/`.

## Numerical kernel, fitting and simulation code

`code/matlab/selection/lin_offset_v2/` contains the corrected LIN covariance with a fitted horizontal offset, likelihood/AIC evaluation, geometry transformation, nested mean initialization, polishing, prediction and conditional simulation. `code/matlab/simulation/` contains the supporting expansion, covariance and positive-semidefinite square-root operations. `code/matlab/selection/kernels_v1/` contains one-dimensional Matérn/RQ factors used by the product construction; these are not radial multivariate Matérn/RQ functions.

Install **GPML 4.2 (2018-06-11)** in `third_party/gpml/` so that `third_party/gpml/gp.m` exists. GPML is distributed separately; its license and dependency notes are supplied here. In MATLAB:

```matlab
addpath(fullfile(pwd,'code','matlab'));
setup_revision_paths;
test_lin_offset_v2;
```

To refit the independent synthetic test scores, explicitly invoke:

```matlab
run_holdout_case('H01'); % repeat for H02–H06 when desired
```

This is a substantial optional computation. It uses the frozen 15 starts, 200 objective evaluations per start, up to 500 polishing evaluations, same seed rule and nested-mean safeguard. Output goes to `recomputed/holdout/`, never over the published score tables. Its paths have been adapted for this release; scientific evaluation code is unchanged. Saved checkpoints from the author's original absolute directory are intentionally not reused. MATLAB/GPML numerical versions can affect new fits; the supplied frozen tables make the search experiment reproducible independently of refitting.

The old development score tables arose from the documented LIN revision of an existing experiment; do not describe a fresh 15-start H-series refit as a byte-for-byte reproduction of that historical optimization path. The current scientific definitions, synthetic observations and frozen candidate scores are provided.

## Optional new model calls

The complete static prompts are in `prompts/control_system.txt` and `prompts/strategy_system.txt`. Dynamic feedback and legal-neighbor construction are in `code/python/base/core.py` and `code/python/search.py`. Their scientific function bodies were extracted without changes from the frozen experiments. The unranked catalogue is `prompts/kernel_catalog.json`.

The default entry point makes **no network call**:

```text
python -B tools/run_search.py --case H01 --method strategy
```

A new paid run requires all of: `--allow-paid-api`, an explicit `--max-usd` cap, and the `OPENROUTER_API_KEY` environment variable. Never save a key in this repository. Example after setting a key in your own shell:

```text
python -B tools/run_search.py --case H01 --method strategy --model openai/gpt-6-astra --run-id my_H01_astra --allow-paid-api --max-usd 2
```

The pricing snapshot is dated to the experiment; verify current model availability and pricing before enabling new calls. Costs are reserved before each request and billed requests are not resent after an interrupted local save. New requests and responses are written only below ignored `recomputed/api/`; do not publish that directory. Live LLM outputs are stochastic and provider/model revisions can change future results. The safe archived visits reproduce the reported figures without new calls.

## CPT data boundary

The raw CPT coordinates and observations may be requested from the authors, subject to the applicable data permissions. No CPT training vector, pointwise target/prediction table, posterior coefficients, gridded posterior field arrays or raw data profile is included here. `results/cpt_aggregate/` contains the candidate AIC table, the top five models' full-precision covariance/mean/noise hyperparameters and coordinate preprocessing summaries, and model-level or probability-bin aggregate results. CPT scoring and simulation require the original observations as well as those parameters. The algorithm itself can be run on the public synthetic data or on a reader's own site data.

Original provider transcripts, hidden reasoning, full requests, account identifiers, local account paths and credentials are excluded. `provenance/build_sources.json` records the source hashes and publication transformations; `MANIFEST.json` records hashes of the released files. All publication files were copied into a separate tree; the working research repository and its history were not reset or rewritten.

## File guide

|Path|Contents|
|---|---|
|`code/matlab/`|Current numerical kernel, fitting, selection and simulation source|
|`code/python/`|Frozen search logic and synthetic generators|
|`prompts/`|Full static prompts, legal catalogue and model/pricing snapshot|
|`data/development/`|EX1–EX4 observations/profiles and frozen synthetic scores|
|`data/holdout/`|H01–H06 observations, generating mechanisms and frozen scores|
|`results/structured_panel/`, `results/development/`, `results/holdout/`|Safe 80+50+18 visited-candidate records, actual costs, settings and baselines|
|`results/cpt_aggregate/`|CPT score and aggregate outputs only|
|`tools/`|Offline audit, seed reproduction, plotting and explicitly gated new API entry|

Code licensing follows the included MIT license. Third-party GPML has its own FreeBSD license. No CPT raw-data license is granted by this package.
