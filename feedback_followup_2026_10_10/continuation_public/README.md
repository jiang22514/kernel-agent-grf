# Offline reproduction of the continuation analysis

Run from this directory or any working directory:

```console
python reproduce_continuations.py
```

Python 3.8 or newer is sufficient. Only the Python standard library is used;
no packages, model API, network, original project directory, GP fitting or random
simulation are required. Results are written to `regenerated/`. Optional arguments
`--input-dir` and `--output-dir` select other directories.

## Included evidence

- `score_tables.json`: all 376 fitted candidate AIC scores for each of six public
  synthetic cases H01–H06 (2256 scores), retaining only kernel, mean ID, AIC and fit
  status. No data profiles, original filesystem paths or fitted parameter arrays.
- `candidate_accesses.json`: all 20 candidate visits from each of 18 searches;
  round numbers identify the exact first batch. Each run retains its original
  `run_sha256` in `source_provenance`. No model replies, reasoning, prompts,
  request headers, costs or credentials are included.
- `reference/`: the previously verified complete CSV and three-group summary.
- `regenerated/`: recalculated outputs and the verification report supplied with
  this release; rerunning the script regenerates these files.

The run groups are Astra structured (`astra/control`), Astra competing-structure
(`astra/strategy`), and Fable competing-structure (`fable51/strategy`), each covering
H01–H06. The first batch contains five candidates for structured search and four
for competing-structure search. The total budget is 20 candidates in both cases.

## What is calculated

The recorded first batch is held fixed. The observed subsequent proposals are
compared with drawing the same remaining number of candidates uniformly without
replacement from the unvisited library. This isolates the recorded continuation
from a random continuation reference; it is not the no-feedback LLM ablation.

Let N be the number of unvisited candidates, b the remaining accesses, and h the
number of unvisited candidates below a specified AIC-gap threshold. If the first
batch has already reached that threshold, the final hit probability is one.
Otherwise the exact probability is `1 - C(N-h,b)/C(N,b)`. For ordered remaining
AIC gaps, the probability that position j (zero based) is the lowest selected
position is `C(N-j-1,b-1)/C(N,b)`. Weighting the better of that gap and the fixed
first-batch gap gives the exact expected final AIC gap. No Monte Carlo approximation
is used. Near-optimal means gap <= 2; champion means gap <= 1e-9. The probability
of matching an observed agent result uses its final gap plus 1e-9, matching the
original verified analysis.

Every CSV field and every summary value is checked against `reference/` using
exact equality. The delivered re-run reproduced both reference files byte for
byte. The output CSV retains the original `run_sha256` column for compatibility;
its values come from the explicit source_provenance fields, not hashes of the
smaller sanitized JSON records.

These are stored numerical evaluations on synthetic cases. This package contains
no private CPT observations. It does not fit a new model or change any reported
experimental result.
