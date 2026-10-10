# Independent test search figure

Run `python plot_independent_gaps.py` with Python, NumPy and Matplotlib. Outputs are written to `regenerated/` as PDF, SVG and 300/600 dpi PNG files.

The CSV includes all 24 outcomes: six cases, three agent model/procedure combinations, and the width-two greedy reference. Values retain the recorded precision. Agent records are from `output/agent_holdout_test_2026-10-08/analysis/run_summary.csv` (`regret`); greedy records are from that experiment's `baselines/mechanical.json` (`results[].gap`). Both measure best accessed AIC minus the case's fitted library minimum. `control` means structured search and `strategy` means competing-structure search. No rows are excluded. Each procedure has one search per independent case at a common budget of 20 candidates. These are individual outcomes; no confidence intervals or significance tests are implied.

H05's Astra competing-structure value is 2.000131133047205, so it is outside the near-optimal threshold of 2. It is displayed to five decimals; other nonzero labels are rounded to two decimals. The manuscript supplementary table provides five-decimal values for all outcomes.
