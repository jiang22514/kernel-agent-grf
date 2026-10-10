"""Reproduce the exact finite-library continuation comparison, offline.

Standard library only. Reads six synthetic score tables and all 18 recorded
candidate-access sequences. No model API, numerical fitting, random sampling,
private CPT observations, or original project-directory access is required.
"""
from pathlib import Path
from math import comb, isfinite
import argparse, csv, hashlib, json

HERE = Path(__file__).resolve().parent
GROUPS = [('astra', 'control'), ('astra', 'strategy'), ('fable51', 'strategy')]
CASES = [f'H{i:02}' for i in range(1, 7)]


def calculate(input_dir):
    assets = json.loads((input_dir/'score_tables.json').read_text(encoding='utf-8'))
    recorded = json.loads((input_dir/'candidate_accesses.json').read_text(encoding='utf-8'))['runs']
    assert len(recorded) == 18 and set(assets['cases']) == set(CASES)
    runs = {(r['case'], r['model'], r['procedure']): r for r in recorded}
    assert len(runs) == 18
    rows = []
    for model, method in GROUPS:
        for case in CASES:
            run = runs[(case, model, method)]
            data, visited = assets['cases'][case]['rows'], run['visited']
            assert len(data) == 376 and all(r['ok'] and isfinite(r['aic']) for r in data)
            scores = {(r['kernel'], r['mean_id']): r['aic'] for r in data}
            assert len(scores) == 376
            optimum = min(scores.values())
            assert len(visited) == len({(r['kernel'], r['mean_id']) for r in visited}) == 20
            for r in visited:
                assert r['aic'] == scores[(r['kernel'], r['mean_id'])]
            initial = [r for r in visited if r['round'] == 1]
            first_keys = {(r['kernel'], r['mean_id']) for r in initial}
            assert len(first_keys) == len(initial) == (5 if method == 'control' else 4)
            first_gap = min(r['aic'] for r in initial)-optimum
            final_gap = min(r['aic'] for r in visited)-optimum
            remaining = sorted(v-optimum for k, v in scores.items() if k not in first_keys)
            N, b = len(remaining), 20-len(initial)
            denominator = comb(N, b)

            def hit_probability(threshold):
                if first_gap <= threshold:
                    return 1.0
                h = sum(v <= threshold for v in remaining)
                return 1-comb(N-h, b)/denominator if N-h >= b else 1.0

            # If remaining candidates are ordered by AIC gap, the event that
            # position j is the first selected position has this exact weight.
            weights = [comb(N-j-1, b-1)/denominator for j in range(N-b+1)]
            assert abs(sum(weights)-1) < 1e-12
            expected_gap = sum(p*min(first_gap, gap) for p, gap in zip(weights, remaining))
            rows.append({'case': case, 'model': model, 'procedure': method,
                'first_evals': len(initial), 'first_gap': first_gap, 'final_gap': final_gap,
                'gap_reduction': first_gap-final_gap,
                'first_near': first_gap <= 2, 'final_near': final_gap <= 2,
                'first_champion': first_gap <= 1e-9, 'final_champion': final_gap <= 1e-9,
                'random_near_probability': hit_probability(2),
                'random_champion_probability': hit_probability(1e-9),
                'random_expected_final_gap': expected_gap,
                'random_probability_matching_agent': hit_probability(final_gap+1e-9),
                'run_sha256': run['source_provenance']['run_sha256']})
    summary = {}
    for model, method in GROUPS:
        sub = [r for r in rows if (r['model'], r['procedure']) == (model, method)]
        summary[model+'_'+method] = {'cases': len(sub),
            **{k: sum(r[k] for r in sub) for k in ['first_near', 'final_near', 'first_champion', 'final_champion']},
            **{k: sum(r[k] for r in sub)/len(sub) for k in ['first_gap', 'final_gap', 'gap_reduction',
                'random_near_probability', 'random_champion_probability', 'random_expected_final_gap']}}
    return rows, summary


def verify(rows, summary, reference_dir):
    with (reference_dir/'continuation_case_results.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = list(reader)
        assert reader.fieldnames == list(rows[0])
    assert len(expected) == len(rows)
    numeric_checks = 0
    for calculated, saved in zip(rows, expected):
        for key, value in calculated.items():
            # Exact float equality, not rounded display or tolerance comparison.
            if isinstance(value, bool):
                assert str(value) == saved[key], (calculated['case'], key)
            elif isinstance(value, (int, float)):
                assert value == float(saved[key]), (calculated['case'], key, value, saved[key])
                numeric_checks += 1
            else:
                assert value == saved[key], (calculated['case'], key)
    saved_summary = json.loads((reference_dir/'continuation_summary.json').read_text(encoding='utf-8'))
    assert summary == saved_summary, 'Summary differs from verified reference'
    return {'case_rows': len(rows), 'numeric_csv_values': numeric_checks,
            'csv_all_fields_exact': True, 'summary_all_values_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path, default=HERE/'regenerated')
    args = parser.parse_args()
    rows, summary = calculate(args.input_dir)
    qa = verify(rows, summary, args.input_dir/'reference')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'continuation_case_results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir/'continuation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    qa['input_sha256'] = {name: hashlib.sha256((args.input_dir/name).read_bytes()).hexdigest()
        for name in ['score_tables.json', 'candidate_accesses.json']}
    qa['byte_identical_reference_outputs'] = {name:
        (args.output_dir/name).read_bytes() == (args.input_dir/'reference'/name).read_bytes()
        for name in ['continuation_case_results.csv', 'continuation_summary.json']}
    (args.output_dir/'verification.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
