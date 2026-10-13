# Public-site covariance-search records: Leendert de Boerspolder

This package provides the actual six completed searches on the public Dutch CPT case: three GPT-6-Astra structured searches and three Claude Fable 5.1 competing-structure searches. All searches use the same 12-CPT training subset (696 observations), the same 94-kernel catalogue, four mean functions, and the response ln(qc / 1 MPa). The 11 held-out CPTs are reserved for local predictive assessment; their observations and summaries do not enter these searches.

Each search has a budget of **20 distinct candidate-score accesses**, not 20 new online GP fits. The 376 kernel–mean candidates were fitted once to the frozen training set using the common numerical protocol. Search agents request candidates from that table; they receive only scores they have already requested. Releasing the complete score table after the searches allows independent calculation of each search's distance from the library minimum. It does not mean that unseen scores were available in the prompts.

Astra uses four rounds of five accepted candidates, and Fable uses five rounds of four. The ordered replies include two reserves per round. The original validity and duplicate checks determine which primary/reserve proposals are accepted. The two procedures retain their respective fixed prompts, own short decision summaries, and score-free lists of legal neighbors derived from visited candidates. These are comparisons between model–procedure combinations, not an isolated comparison of model providers.

## Offline reproduction

Run with Python 3.9 or later; no third-party packages, API key, internet connection, MATLAB installation or GP fitting are needed:

    python -B verify_public_search.py

The verifier checks the package manifest, reconstructs every request from the training profile and preceding visits, parses the archived visible responses through unchanged acceptance functions, and reproduces the 20 accesses in order. It also verifies the short summary passed to the next round, candidate AIC values, each round's best-so-far value, final AIC gaps, champion/near-optimal counts, token usage and billed costs.

The following files are written to replay_output:

- summary.json: all six searches, separate three-search summaries for each model–procedure, request count and total experiment cost.
- runs.csv: final selected covariance, mean, AIC and distance from the library minimum.
- round_best_so_far.csv: cumulative accesses and best AIC after every round.
- access_history.csv: every accepted access and the running best AIC.
- usage.csv: prompt, completion and total token counts, plus cost for each request.

Lower AIC gap is better. A champion has a gap <= 1e-9; near-optimal means a gap <= 2, evaluated without rounding. Repetitions measure search variability on one fixed site/training set; they are not six independent engineering sites. The application representatives were fixed as the first Astra and first Fable searches, rather than the best of three repetitions.

## Contents and disclosure

assets.json contains the training-only numerical profile, unranked kernel catalogue and the full public-case table of 376 fitted AIC values and parameter counts. prompts contains both complete fixed system prompts. Each runs subdirectory contains its 20-access history and all actual request messages and visible response contents. The response summary is retained as a short decision summary; provider-hidden reasoning and reasoning details are excluded.

plan.json records the run ordering, model identifiers and settings. model_catalog.json retains relevant public model/pricing metadata from the experiment. ledger.json reports necessary model/provider, usage-cost and request-status fields without account identifiers. The request's reasoning.effort field specifies an API setting; it is not hidden model reasoning.

source contains only the pure parsing, proposal-acceptance, neighborhood and prompt-construction functions extracted unchanged from the frozen research code. source_provenance.json identifies their original research-source hashes and individual definition hashes. No API controller or runnable paid-experiment entry point is included.

Credentials, HTTP headers, provider-hidden reasoning, private filesystem paths, the private original CPT dataset, and the MathWorks runtime are not included. The full public Dutch data and numerical fitting workflow belong to the separate engineering reproduction package. This package verifies the recorded searches against their frozen score table; it does not independently rerun the 376 GP optimizations.

Fresh API calls can produce different outputs because model sampling and provider routing are stochastic and hosted models may change. Archived replay is deterministic and makes no network calls. No package file initiates a paid request.
