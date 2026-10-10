"""Offline replay of public numerical-feedback records. Python standard library only."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def close(left, right):
    assert abs(left - right) <= 1e-9, (left, right)


def replay(root):
    # These imports only expose pure parser/prompt/statistics functions. No API
    # controller is instantiated and no paid-execution entry point is called.
    core = load('public_core', root / 'source_audit/core.py')
    dev = load('public_development', root / 'source_audit/experiment.py')
    loop = load('public_loop', root / 'source_audit/ablation.py')
    stats = load('public_stats', root / 'source_audit/analyze_ablation.py')
    plan, assets = read(root / 'plan.json'), read(root / 'assets.json')
    starts, ledger = read(root / 'common_starts.json'), read(root / 'ledger.json')
    assert len(plan['runs']) == 24 and set(assets['cases']) == {f'H{i:02d}' for i in range(1, 7)}
    assert len(assets['kernels']) == len(set(assets['kernels'])) == 94
    assert len(ledger) == 84 and all(not e['billing_unknown'] for e in ledger)
    assert len({e['request_file'] for e in ledger}) == 84
    runs, curves, requests = [], [], 0
    for run in plan['runs']:
        result = read(root / 'runs' / run['run_id'] / 'result.json')
        assert result['state'] == 'complete'
        rows = assets['cases'][run['case']]['rows']
        assert len(rows) == 376
        score = {(r['kernel'], r['mean_id']): r for r in rows}
        optimum = min(r['aic'] for r in rows)
        profile = assets['cases'][run['case']]['profile']
        shared = starts[f"{run['case']}__{run['model_label']}"]
        first = read(root / 'common_first_calls' / (f"{run['case']}__{run['model_label']}" + '.json'))
        initial_user = (core.structured_prompt(profile, [], 1, assets['kernels'], '') if run['method'] == 'control'
                        else dev.strategy_prompt(core, profile, [], 1, assets['kernels'], ''))
        original_system = (root / 'original_prompts' / (run['method'] + '.txt')).read_text(encoding='utf-8-sig')
        assert first['request']['messages'] == [{'role': 'system', 'content': original_system},
                                                {'role': 'user', 'content': initial_user}]
        first_obj = core.parse(first['response']['message']['content'])
        if run['method'] == 'control':
            accepted, _ = core.accepted(first_obj, 'structured', assets['kernels'], set())
        else:
            accepted, _ = dev.accepted(first_obj, assets['kernels'], set(), run['batch_size'])
        assert accepted == [(r['kernel'], r['mean_id']) for r in shared['first_round']]
        assert str(first_obj.get('summary', ''))[:1200] == shared['summary']
        visited = [dict(r) for r in shared['first_round']]
        seen = {(r['kernel'], r['mean_id']) for r in visited}
        summary = shared['summary']
        system = (root / 'prompts' / (run['method'] + '.txt')).read_text(encoding='utf-8')
        for rd in range(2, run['total_rounds'] + 1):
            record = read(root / 'runs' / run['run_id'] / f'round_{rd}.json')
            request = record['request']
            assert request['model'] == run['model'] and request['max_tokens'] == 8192
            assert request['reasoning'] == {'effort': 'medium'}
            expected_messages = [{'role': 'system', 'content': system}, {'role': 'user',
                'content': loop.user_prompt(profile, assets['kernels'], visited, summary, run, rd)}]
            assert request['messages'] == expected_messages, ('PROMPT_MISMATCH', run['run_id'], rd)
            if run['arm'] == 'blind':
                changed = [dict(r, aic=999999.125 - i) for i, r in enumerate(visited)]
                assert loop.user_prompt(profile, assets['kernels'], changed, summary, run, rd) == expected_messages[1]['content']
            obj = core.parse(record['response']['message']['content'])
            if run['method'] == 'control':
                pairs, rejected = core.accepted(obj, 'structured', assets['kernels'], seen)
            else:
                pairs, rejected = dev.accepted(obj, assets['kernels'], seen, run['batch_size'])
            assert len(pairs) <= run['batch_size']
            for key in pairs:
                assert key not in seen and key in score
                seen.add(key)
                visited.append({**score[key], 'round': rd})
            stored_round = next(r for r in result['rounds'] if r['round'] == rd)
            assert stored_round['new_evals'] == len(pairs) and stored_round['rejected'] == rejected
            summary, note = loop.safe_summary(obj.get('summary', ''), run['arm'])
            assert stored_round['summary'] == summary and stored_round.get('summary_note') == note
            entry = next(e for e in ledger if e['request_file'] == f"runs/{run['run_id']}/round_{rd}.json")
            close(record['response']['usage']['cost'], entry['charged_or_reserved_usd'])
            requests += 1
        assert len(visited) == len(seen) == 20
        assert len(result['visited']) == len(visited)
        for observed, archived in zip(visited, result['visited']):
            assert (observed['kernel'], observed['mean_id'], observed['round']) == (archived['kernel'], archived['mean_id'], archived['round'])
            close(observed['aic'], archived['aic'])
            assert observed['k'] == archived['k']
        best = min(visited, key=lambda r: r['aic'])
        gap = best['aic'] - optimum
        first_gap = min(r['aic'] for r in shared['first_round']) - optimum
        close(gap, result['regret'])
        assert result['n_evals'] == 20 and result['success_champion'] == (gap <= 1e-9)
        assert result['success_near'] == (gap <= 2)
        trajectory = [min(r['aic'] for r in visited[:i]) - optimum for i in range(1, 21)]
        assert len(result['regret_trajectory']) == 20
        for left, right in zip(trajectory, result['regret_trajectory']):
            close(left, right)
        entries = [e for e in ledger if e['run_id'] == run['run_id']]
        cost = sum(e['charged_or_reserved_usd'] for e in entries)
        close(cost, result['known_cost_usd'])
        runs.append({**run, 'state': 'complete', 'n_evals': 20, 'regret': gap, 'first_gap': first_gap,
            'improvement_from_shared_first': first_gap - gap,
            'champion_success': gap <= 1e-9, 'near_success': gap <= 2, 'best_kernel': best['kernel'],
            'best_mean': best['mean'], 'known_cost_usd': cost, 'api_requests': len(entries)})
        curves.extend({'run_id': run['run_id'], 'case': run['case'], 'model': run['model_label'],
                       'arm': run['arm'], 'visit': i + 1, 'regret': v} for i, v in enumerate(trajectory))
    assert requests == 84
    pairs = []
    for model in ['astra', 'fable51']:
        for case in plan['case_order']:
            row = {r['arm']: r for r in runs if r['model_label'] == model and r['case'] == case}
            f, b = row['feedback'], row['blind']
            close(f['first_gap'], b['first_gap'])
            pairs.append({'model_label': model, 'case': case, 'pair_complete': True,
                          'feedback_gap': f['regret'], 'blind_gap': b['regret'],
                          'regret_feedback_minus_blind': f['regret'] - b['regret'],
                          'gap_reduction': b['regret'] - f['regret'], 'first_gap': f['first_gap'],
                          'feedback_cost_usd': f['known_cost_usd'], 'blind_cost_usd': b['known_cost_usd']})
    groups, paired = stats.aggregate(runs, pairs)
    cost = sum(e['charged_or_reserved_usd'] for e in ledger)
    report = {'status': 'PASS', 'all_24_complete': True, 'verified_runs': 24, 'verified_new_requests': requests,
              'known_cost_usd': cost, 'prior_cost_usd': plan['prior_experiment_cost_usd'],
              'project_cost_usd': cost + plan['prior_experiment_cost_usd'],
              'model_arm_summary': groups, 'paired_summary': paired, 'paired_cases': pairs,
              'returned_models': dict(Counter(e.get('returned_model') or 'unknown' for e in ledger)),
              'providers': dict(Counter(e.get('provider') or 'unknown' for e in ledger)),
              'http_calls': 0, 'note': 'Six reused test cases analyzed separately per model; not 12 independent datasets or a new blind test.'}
    output = root / 'replay_output'
    output.mkdir(exist_ok=True)
    stats.dump(output / 'summary.json', report)
    stats.csv_write(output / 'runs.csv', runs)
    stats.csv_write(output / 'paired_case_results.csv', pairs)
    stats.csv_write(output / 'trajectories.csv', curves)
    return {k: report[k] for k in ['status', 'all_24_complete', 'verified_runs', 'verified_new_requests', 'known_cost_usd', 'http_calls']}


def verify(root):
    from unittest.mock import patch
    manifest = read(root / 'manifest.json')
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, ('HASH_MISMATCH', name)
    with patch('urllib.request.urlopen', side_effect=AssertionError('NETWORK_DISABLED_IN_REPLAY')):
        return replay(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))
