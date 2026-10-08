# Kernel-agent geotechnical random fields

Code and reproducibility materials for **Compositional covariance discovery for geotechnical random fields: LLM-guided search and exact conditional simulation**.

## Current research release — 8 October 2026

Start with [research_release_2026_10](research_release_2026_10/README.md). This release contains the corrected linear covariance with a fitted horizontal offset, a 376-candidate kernel–mean library, structured LLM search, and exact conditional simulation on a coordinate-union grid.

The two scientific components are:

- **Data-guided covariance discovery:** an LLM proposes explicit kernel–mean candidates and receives their numerical likelihood/AIC evaluations. Structured search uses four batches of five; a competing-structure variant uses five batches of four. Both use the same 20-candidate budget.
- **Faithful conditional random fields:** the selected covariance is expanded into separable terms, sampled independently on the observation–query coordinate union, and conditioned jointly through a Matheron update. It preserves the fitted covariance without inducing-point interpolation or a shared eigenbasis across covariance terms.

On six new synthetic cases generated after fixing the search procedures, Astra structured search and Fable 5.1 with the competing-structure variant each recovered five library optima and reached within two AIC units in all six cases, evaluating 20 of 376 candidates. Uniform random selection has a mean near-optimal probability of 15.60%. Astra with the variant reached four of six; the results support evaluating each model–procedure combination rather than assuming a universally better prompt.

For the 744-observation CPT case, the best composite improves AIC by 49.8 over the best single kernel. A matched same-model benchmark generated 500 fields at 24,321 locations in 6.47 seconds versus 44.35 seconds for dense conditional simulation, including preparation (6.85×). Mean weak-zone proportions were 10.77%, 16.10% and 12.73% for SE, RQ and the selected composite. The model-selection advantage does not imply superior probabilistic prediction: those comparisons are reported separately.

## Reproduce without new model calls

```text
cd research_release_2026_10
python -B tools/verify_release.py
```

The standard-library verifier checks the released file hashes, candidate-access records, outcomes and costs. No API key or GP refitting is required. The release README gives optional synthetic-data regeneration, numerical fitting and explicitly enabled live-API instructions.

Included are synthetic observations and generators, prompt templates, model settings, candidate catalogues, sanitized search records, numerical code, fitted-model results and CPT aggregate outcomes. **Original CPT coordinates and observations are not public; they may be requested from the authors.** CPT reruns require access to those observations. Keys, private provider transcripts and posterior field arrays are excluded from the current release.

## Earlier versions

The root-level `selection/`, `agent/`, `simulation/`, `diagnostics/` and `results/` directories are retained as historical material for the earlier submission. They use earlier kernel definitions and experiments and should not be mixed with the current release. The [earlier README](README_EAAI_legacy.md) documents that version. Use the self-contained release directory for the current paper.

Code is available under the existing [MIT license](LICENSE); third-party GPML is obtained separately under its own license. See the current release for dependency details.
