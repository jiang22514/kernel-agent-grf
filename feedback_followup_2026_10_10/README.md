# Prompt presentation and numerical-feedback follow-up — 10 October 2026

- `figure_source/agent_framework/`: run `python draw_agent_framework.py`
  (Matplotlib, PyMuPDF, Pillow) to reproduce conceptual Fig. 1; no API or fitting.
- `figure_source/independent_case_gaps/`: standalone plotting script and all 24
  full-precision results; run `python plot_independent_gaps.py` (NumPy, Matplotlib).
- `continuation_public/`: standalone standard-library script, all six synthetic
  score tables, all 18 access sequences, reference results and verification report.
  Run `python reproduce_continuations.py` inside this directory: no original project
  directory, model API, fitting or extra package is required. Every CSV field and
  summary value is checked against the verified references. This comparison with
  random continuation is separate from the no-feedback LLM ablation.
- `continuation_analysis_audit/`: unchanged original project analysis script and a
  note directing readers to the standalone reproduction above.
- `feedback_ablation/`: only explicitly approved analysis files and optional sanitized
  summaries/prompts/runs listed in PUBLIC_FILES.json. No provider raw replies, internal
  reasoning, private CPT observations or credentials are included.

Packaging executes no analysis. Existing research materials:
https://github.com/jiang22514/kernel-agent-grf/tree/main/research_release_2026_10
This supplement adds explicit prompt examples, the expanded agent-loop diagram, exact random-continuation comparisons and a paired numerical-feedback ablation. The original independent-test outcomes remain in the 8 October release.

On the six shared synthetic cases, score-informed continuations reached near-optimality in 6/6 cases for each model; score-masked continuations reached 2/6 for Astra and 3/6 for Fable. Each model had five lower final AIC gaps and one tie with feedback. All 24 continuations completed 20 distinct candidate accesses, including their common initial batches; 84 fresh requests cost USD 5.35221. This is a mechanism follow-up on the existing six cases, reported separately by model.

`prompt_templates/` supplies the complete main-method system prompts and literal synthetic second-round examples. Rebuild the paired-feedback results with `python -B feedback_ablation/records/verify_public_ablation.py`; the script uses no API calls or GP refitting. The corresponding paired plot is in `feedback_ablation/analysis/`.
