# Paired numerical-feedback results

All 24 continuations and 84 paid requests completed. The six datasets are shared by the two models; each model is summarized separately over six pairs. AIC gap is the best visited score minus the fitted library minimum (smaller is better). Near-optimality uses the unrounded gap <= 2; exact recovery uses <= 1e-9.

`paired_case_results.csv` contains all 12 model–case pairs. `gap_reduction = blind_gap - feedback_gap`, so positive values favor feedback. `regret_feedback_minus_blind` uses the opposite sign and is explicitly named. `runs.csv` retains all 24 continuations; `trajectories.csv` records every visited candidate; `request_costs.csv` records the 84 request charges. The original independent-test outcomes are separate.

From this directory, recreate the figure with:

```text
python -B plot_feedback_ablation.py --input paired_case_results.csv --output regenerated
```

Requires Python, NumPy and Matplotlib. Both panels use the same scale, linear up to AIC gap 2 and logarithmic thereafter. For full offline reconstruction from visible requests, responses and fixed scores, run `python -B ../records/verify_public_ablation.py`. That replay requires no API calls or GP refitting.
