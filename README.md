# kernel-agent-grf

**Kernel Agent for Geotechnical Random Fields** — a closed-loop LLM agent for
compositional kernel discovery.

Code, results, and complete LLM transcripts accompanying the paper:

> *A closed-loop LLM agent for compositional kernel discovery and fast simulation of geotechnical random fields*

The framework replaces the conventional fixed kernel library of Gaussian-process
random field modeling with a bounded compositional grammar (7 primitives, sum and
product operators, 156 kernel/mean candidates), and uses a closed-loop LLM agent --
Perceive, Plan, Act, Reflect -- to discover the best composite structure at a
fraction of the exhaustive cost. Two architectural firewalls (grammar validation
and exact-likelihood AIC arbitration) guarantee that LLM errors can waste search
budget but never corrupt the final selection. A sum-of-Kronecker extension of the
grid inducing-point engine keeps every discovered composite kernel on the fast
simulation path.

## Requirements

- MATLAB (tested on R2023b+; no special toolboxes required)
- [GPML toolbox v4.2](http://gaussianprocess.org/gpml/code/matlab/doc/) --
  download separately and add to the MATLAB path (`addpath(genpath(...))`).
- For agent scripts only: an [OpenRouter](https://openrouter.ai) API key in the
  `OPENROUTER_API_KEY` environment variable. Everything except the live agent
  runs (enumeration, simulation, figures) works without any API key.

All scripts assume the repository folders are on the MATLAB path:

```matlab
addpath(genpath('path/to/this/repo'));
addpath(genpath('path/to/gpml-matlab-v4.2'));
```

## Repository layout

| Folder | Contents |
|---|---|
| `selection/` | Compositional grammar, composite-kernel GP fitting and AIC/BIC evaluation (exact marginal likelihood), exhaustive ground-truth enumeration, held-out validation |
| `agent/` | Closed-loop LLM agent: data profiling (Perceive), proposal parsing with grammar validation, the agent loop, the multi-model study driver, and figure generation |
| `simulation/` | Sum-of-Kronecker simulation engine: structure routing, separable-term expansion (incl. PER/RQ realizations), Matheron posterior sampler, validation tests, borehole simulation figures |
| `diagnostics/` | One-off probes used during development (SKI accuracy vs. grid density, mean-function pipeline checks) |
| `results/` | Saved `.mat` results of every experiment in the paper |
| `results/transcripts/` | Complete verbatim LLM transcripts of all 40 agent runs (8 LLMs x 5 cases): prompts, proposals, per-proposal reasons, per-round analyses |
| `results/figures/` | Paper figures produced by the scripts |

## Key entry points

| Script | What it does | Paper section |
|---|---|---|
| `selection/enumerate_grammar_aic.m` | Exhaustive 156-candidate ground truth, synthetic cases | 4.3 |
| `selection/enumerate_borehole_fulln.m` | Exhaustive ground truth, borehole at full n=744 | 5.2 |
| `selection/holdout_borehole.m` | Held-out predictive validation (10 splits, RMSE/NLPD) | 5.2 |
| `agent/run_agent_study.m` | Full closed-loop agent study (8 LLMs x 5 cases) | 4.4, 5.3 |
| `agent/make_agent_figures.m` | Convergence and open-loop-vs-closed-loop figures | 4.4 |
| `simulation/test_grid_sim_terms.m` | Validates the K>=2 sampler against the exact GP posterior | 2.3.3 |
| `simulation/test_reform.m` | Validates PER/RQ simulation-stage realizations | 2.3.3 |
| `simulation/make_borehole_sim_figures.m` | Three-model borehole field + variogram figures | 5.4 |

Test scripts (`test_*.m`) are self-contained checks; each prints a PASS/FAIL verdict.

## Reproducibility

- The arbitration layer (data, GP fitting, AIC, exhaustive ground truth) is fully
  deterministic under the fixed seeds baked into the scripts.
- The LLM layer is queried at temperature 0.01 -- near-deterministic but, as with
  all commercial LLM APIs, not bit-reproducible. The complete transcripts in
  `results/transcripts/` document the exact runs reported in the paper, and all
  saved `.mat` results allow every table and figure to be regenerated without
  re-querying any LLM.

## Data

The synthetic cases (EX1-EX4) are generated inside the scripts with fixed seeds.
The Wuhan CPT borehole dataset (`samedata.mat`) is not redistributed in this
repository; it is available from the corresponding author on reasonable request.

## License

MIT (see `LICENSE`). The GPML toolbox is distributed separately under its own
(FreeBSD-style) license.
