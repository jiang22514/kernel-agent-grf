"""Offline reproduction of scores, search metrics, costs and release-field audit."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in
                  ['.git', '__pycache__', 'recomputed']) and p.name not in ['MANIFEST.json', 'verification.json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Create/refresh checksums for this local release tree.')
    args = parser.parse_args()
    issues, result, visit_count = [], [], 0
    cpt = read(ROOT / 'results/cpt_aggregate/candidate_scores.json')
    assert set(cpt) == {'case', 'kernel_definition', 'rows', 'privacy'}
    for row in cpt['rows']:
        assert set(row) == {'kernel', 'mean_id', 'mean', 'aic', 'k', 'ok'}
    for dataset in ['structured_panel', 'development', 'holdout']:
        assets = read(ROOT / 'data' / ('development' if dataset == 'structured_panel' else dataset) / 'assets.json')
        assert 'Borehole' not in assets['cases']
        cases = dict(assets['cases'])
        if dataset in ['structured_panel', 'development']:
            cases['Borehole'] = {'rows': cpt['rows']}
        for case, data in cases.items():
            assert len(data['rows']) == 376
            assert len({(r['kernel'], r['mean_id']) for r in data['rows']}) == 376
        runs = read(ROOT / 'results' / dataset / 'search_visits.json')['runs']
        billing = read(ROOT / 'results' / dataset / 'billing.json')
        by_run = defaultdict(list)
        for entry in billing:
            by_run[entry['run_id']].append(entry)
        groups = defaultdict(lambda: {'n': 0, 'completed': 0, 'champion': 0, 'near': 0, 'visits': 0, 'cost_usd': 0.0})
        for run in runs:
            score = {(r['kernel'], r['mean_id']): r for r in cases[run['case']]['rows']}
            visited = run['visited']; keys = [(r['kernel'], r['mean_id']) for r in visited]
            assert len(visited) <= 20 and len(set(keys)) == len(keys)
            assert len(visited) == run['n_evals']
            for row, key in zip(visited, keys):
                assert row['aic'] == score[key]['aic'] and row['k'] == score[key]['k']
            optimum = min(r['aic'] for r in score.values())
            gap = min((r['aic'] for r in visited), default=math.inf) - optimum
            if visited:
                assert abs(gap - run['regret']) < 1e-9
            costs = sum(e['charged_or_reserved_usd'] for e in by_run[run['run_id']] if not e['billing_unknown'])
            assert abs(costs - run['cost_usd']) < 1e-9
            complete = run['state'] == 'complete'
            hit, near = complete and gap <= 1e-9, complete and gap <= 2
            g = groups[(run.get('stage', dataset), run['model'], run['method'])]
            g['n'] += 1; g['completed'] += complete; g['champion'] += hit; g['near'] += near
            g['visits'] += len(visited); g['cost_usd'] += costs; visit_count += len(visited)
        result.append({'dataset': dataset, 'runs': len(runs), 'groups': [dict(stage=k[0], model=k[1], method=k[2], **v)
                       for k, v in sorted(groups.items())],
                       'cost_usd': sum(e['charged_or_reserved_usd'] for e in billing if not e['billing_unknown']),
                       'billing_unknown': sum(bool(e['billing_unknown']) for e in billing)})
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        if any(token in rel.lower() for token in ['transcript', 'llm_api_config', '.response.', '.request.', 'pointwise_predictions', 'point_predictions']):
            issues.append('PROHIBITED_FILE:' + rel)
        if path.suffix.lower() in ['.py', '.m', '.json', '.csv', '.txt', '.md', '.svg']:
            data = path.read_text(encoding='utf-8-sig')
            if re.search(r'sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}', data):
                issues.append('SECRET_PATTERN:' + rel)
            if re.search(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]', data) or re.search(r'[/\\]Users[/\\][^/\\]+', data):
                issues.append('AUTHOR_ABSOLUTE_PATH:' + rel)
        if path.suffix.lower() == '.mat' and not rel.startswith(('data/holdout/', 'results/synthetic_prediction/')):
            issues.append('UNAPPROVED_BINARY_MAT:' + rel)
    manifest_path = ROOT / 'MANIFEST.json'
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in files()}
    if args.write_manifest:
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
    elif manifest_path.exists():
        expected = read(manifest_path)
        if current != expected:
            issues.append('MANIFEST_FILE_SET_OR_HASH_MISMATCH')
    else:
        issues.append('MANIFEST_MISSING')
    report = {'status': 'PASS' if not issues else 'FAIL', 'issues': issues, 'files_checked': len(current),
              'candidate_visits_verified': visit_count, 'experiments': result,
              'CPT_raw_observations_released': False, 'provider_content_released': False,
              'API_calls': 0, 'GP_fits': 0}
    (ROOT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not issues else 1)


if __name__ == '__main__':
    main()
