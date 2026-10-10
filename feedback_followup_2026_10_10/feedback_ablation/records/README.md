# Numerical-feedback paired follow-up: six synthetic cases

This package replays an additional mechanism experiment on the six previously used synthetic test cases H01–H06. It is not a new blind test. Report six paired cases separately for Astra structured search and Fable competing-hypothesis search; do not treat the two model groups as twelve independent datasets.

Each pair shares its archived score-free first call and continues afresh with numerical feedback or withheld scores. Both branches have the same system prompt, candidate-visit order, output rules, own-summary memory, and no score-selected neighborhood recommendations. The feedback branch receives AIC and its difference from the best visited score, along with parameter count and fit status. The blind branch receives only candidate identity, parameter count, and fit status. A branch's own guessed score is retained as part of its own summary, not replaced with a hidden score. The 24 continuations contain 84 new paid requests; 12 original first calls are supplied separately and are not counted in the new cost.

Run with Python 3.9+ (standard library only), without an API key:

```shell
python -B verify_public_ablation.py
```

The verifier checks file hashes, parses visible model outputs through the actual frozen acceptance functions, applies primary/reserve proposals in order, reconstructs all 20 distinct visits, checks every local AIC, rebuilds round prompts, checks blind-score invariance, and recomputes all 24 final scores and both six-case paired comparisons. Results appear in `replay_output/`. It disables HTTP explicitly. Lower AIC regret is better; near-optimal means regret <= 2 without rounding, and champion means regret <= 1e-9. Positive `gap_reduction = blind - feedback` favors feedback.

`assets.json` contains only the six synthetic-case numerical profiles, 94 legal kernel names and 376 local candidate scores per case. `common_first_calls/` contains the reused initial visible requests/answers; `runs/` contains 24 final results and 84 continuation requests/visible answers. `ledger.json` contains the necessary returned model/provider and cost fields. Provider hidden reasoning, reasoning details, request headers, credentials, account identifiers and private absolute paths are excluded. Visible response content is intentionally retained for audit. `source_audit/` contains unchanged research source snapshots for inspection; these are not a newly packaged paid-execution engine. The pure parsing and prompt functions are used by the offline verifier. Original source hashes and relative research locations are recorded in `plan.json`.

For a fresh paid reproduction, use the repository's existing runner/controller infrastructure with the included profiles, prompts, model settings and protocol. Supply your own key privately and explicitly enable paid execution; do not run audit copies as a turnkey API command. API sampling and provider routing are not deterministic, so new outputs can differ from these archived outputs. GP fitting is not repeated by this offline package. Synthetic generation and scoring code are provided with the repository's independent synthetic-test release. Repository: https://github.com/jiang22514/kernel-agent-grf .
