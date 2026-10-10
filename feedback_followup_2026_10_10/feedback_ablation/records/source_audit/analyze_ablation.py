"""Read-only analysis of the paired numerical-feedback experiment; no API imports."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def aggregate(runs, pairs):
    groups = []
    models = list(dict.fromkeys(r['model_label'] for r in runs))
    for model in models:
        for arm in ('feedback', 'blind'):
            rows = [r for r in runs if r['model_label'] == model and r['arm'] == arm]
            complete = [r for r in rows if r['state'] == 'complete']
            gaps = [r['regret'] for r in complete]
            groups.append({'model_label': model, 'arm': arm, 'planned_cases': len(rows),
                'completed_cases': len(complete), 'champion_count': sum(r['champion_success'] for r in rows),
                'near_count': sum(r['near_success'] for r in rows),
                'full20_completed': sum(r['n_evals'] == 20 for r in complete),
                'mean_regret_completed': statistics.mean(gaps) if gaps else None,
                'median_regret_completed': statistics.median(gaps) if gaps else None,
                'known_cost_usd': sum(r['known_cost_usd'] for r in rows),
                'requests': sum(r['api_requests'] for r in rows)})
    paired = []
    for model in models:
        rows = [p for p in pairs if p['model_label'] == model]
        complete = [p for p in rows if p['pair_complete']]
        ds = [p['regret_feedback_minus_blind'] for p in complete]
        paired.append({'model_label': model, 'planned_pairs': len(rows), 'complete_pairs': len(complete),
            'feedback_lower_regret': sum(d < -1e-9 for d in ds),
            'ties': sum(abs(d) <= 1e-9 for d in ds), 'feedback_higher_regret': sum(d > 1e-9 for d in ds),
            'mean_paired_regret_difference': statistics.mean(ds) if ds else None,
            'median_paired_regret_difference': statistics.median(ds) if ds else None})
    return groups, paired


def analyze(experiment, output):
    loaded = {}

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        loaded[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8-sig'))

    plan = read(experiment / 'plan.json')
    plan_hash = loaded[str((experiment / 'plan.json').resolve())]
    expected_hash = (experiment / 'plan.sha256').read_text().strip()
    if plan_hash != expected_hash:
        raise ValueError('PLAN_HASH_MISMATCH')
    assets, common = read(plan['assets_path']), read(experiment / 'common_starts.json')
    live = Path(plan['live_directory'])
    ledger = read(live / 'ledger.json') if (live / 'ledger.json').exists() else []
    ids = {r['run_id'] for r in plan['runs']}
    if len(ids) != len(plan['runs']) or any(e['run_id'] not in ids for e in ledger):
        raise ValueError('UNPLANNED_OR_DUPLICATE_RUN')
    runs, trajectories, receipts = [], [], []
    for e in ledger:
        receipts.append({k: e.get(k) for k in ('run_id', 'request_file', 'model', 'returned_model', 'provider',
            'state', 'started_unix', 'elapsed_seconds', 'finish_reason', 'reserved_usd',
            'charged_or_reserved_usd', 'billing_unknown', 'http_status')})
    for run in plan['runs']:
        folder = live / 'runs' / run['run_id']
        terminal = read(folder / 'terminal.json') if (folder / 'terminal.json').exists() else None
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else None
        entries = [e for e in ledger if e['run_id'] == run['run_id']]
        state = terminal['state'] if terminal else ('running' if result or entries else 'not_started')
        rows = assets['cases'][run['case']]['rows']
        score = {(r['kernel'], r['mean_id']): r for r in rows}
        optimum = min(r['aic'] for r in rows)
        visited = result.get('visited', []) if result else []
        seen, best, first_gap = set(), None, None
        start = common[f"{run['case']}__{run['model_label']}"]
        for i, row in enumerate(visited, 1):
            key = row['kernel'], row['mean_id']
            if key in seen or key not in score or i > 20:
                raise ValueError('INVALID_VISITED_CANDIDATE:' + run['run_id'])
            seen.add(key)
            expected = score[key]
            if not math.isfinite(row['aic']) or abs(row['aic'] - expected['aic']) > 1e-9 or row['k'] != expected['k']:
                raise ValueError('VISITED_SCORE_MISMATCH:' + run['run_id'])
            if best is None or row['aic'] < best['aic']:
                best = row
            gap = best['aic'] - optimum
            trajectories.append({'run_id': run['run_id'], 'case': run['case'], 'model_label': run['model_label'],
                'arm': run['arm'], 'run_state': state, 'visit': i, 'round': row['round'],
                'kernel': row['kernel'], 'mean_id': row['mean_id'], 'candidate_aic': row['aic'],
                'best_aic_so_far': best['aic'], 'regret': gap,
                'shared_archived_first_round': row['round'] == 1})
            if i == run['batch_size']:
                first_gap = gap
        if visited:
            initial = visited[:run['batch_size']]
            expected = start['first_round']
            if [(r['kernel'], r['mean_id'], r['aic']) for r in initial] != [(r['kernel'], r['mean_id'], r['aic']) for r in expected]:
                raise ValueError('SHARED_START_MISMATCH:' + run['run_id'])
        gap = best['aic'] - optimum if best else None
        if terminal:
            if not result or result['state'] != state or result['plan_sha256'] != plan_hash or terminal['plan_sha256'] != plan_hash:
                raise ValueError('TERMINAL_RESULT_IDENTITY_MISMATCH:' + run['run_id'])
            if result['n_evals'] != len(visited) or abs(result['regret'] - gap) > 1e-9:
                raise ValueError('RESULT_METRIC_MISMATCH:' + run['run_id'])
        known = sum(e['charged_or_reserved_usd'] for e in entries if not e.get('billing_unknown', True))
        runs.append({**run, 'state': state, 'n_evals': len(visited), 'full20': len(visited) == 20,
            'initial_regret': first_gap, 'regret': gap, 'best_aic': best['aic'] if best else None,
            'optimum_aic': optimum, 'best_kernel': best['kernel'] if best else None,
            'best_mean_id': best['mean_id'] if best else None, 'best_mean': best['mean'] if best else None,
            'improvement_from_shared_first': first_gap - gap if gap is not None and first_gap is not None else None,
            'champion_success': state == 'complete' and gap is not None and gap <= 1e-9,
            'near_success': state == 'complete' and gap is not None and gap <= 2,
            'api_requests': len(entries), 'known_cost_usd': known,
            'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in entries),
            'billing_unknown_requests': sum(e.get('billing_unknown', True) for e in entries),
            'returned_models': sorted({e['returned_model'] for e in entries if e.get('returned_model')}),
            'providers': sorted({e['provider'] for e in entries if e.get('provider')}),
            'missing_returned_model_records': sum(not e.get('returned_model') for e in entries),
            'missing_provider_records': sum(not e.get('provider') for e in entries),
            'blind_own_score_claim_notes': sum(bool(r.get('summary_note')) for r in (result or {}).get('rounds', [])),
            'error_type': (result or {}).get('error_type')})
    pairs = []
    for model in dict.fromkeys(r['model_label'] for r in runs):
        for case in plan['case_order']:
            pair = {r['arm']: r for r in runs if r['case'] == case and r['model_label'] == model}
            if set(pair) != {'feedback', 'blind'}:
                raise ValueError('UNPAIRED_PLAN')
            f, b = pair['feedback'], pair['blind']
            complete = f['state'] == b['state'] == 'complete'
            pairs.append({'model_label': model, 'model': model, 'case': case, 'pair_complete': complete,
                'feedback_state': f['state'], 'blind_state': b['state'],
                'feedback_n_evals': f['n_evals'], 'blind_n_evals': b['n_evals'],
                'shared_initial_regret': f['initial_regret'] if f['initial_regret'] is not None else b['initial_regret'],
                'first_gap': f['initial_regret'] if f['initial_regret'] is not None else b['initial_regret'],
                'feedback_gap': f['regret'], 'blind_gap': b['regret'],
                'gap_reduction': b['regret'] - f['regret'] if complete else None,
                'feedback_cost_usd': f['known_cost_usd'], 'blind_cost_usd': b['known_cost_usd'],
                'feedback_regret': f['regret'], 'blind_regret': b['regret'],
                'regret_feedback_minus_blind': f['regret'] - b['regret'] if complete else None,
                'feedback_champion': f['champion_success'], 'blind_champion': b['champion_success'],
                'feedback_near': f['near_success'], 'blind_near': b['near_success'],
                'feedback_best_kernel': f['best_kernel'], 'feedback_best_mean': f['best_mean'],
                'blind_best_kernel': b['best_kernel'], 'blind_best_mean': b['best_mean']})
    group, pair_group = aggregate(runs, pairs)
    complete_count = sum(r['state'] == 'complete' for r in runs)
    charged = sum(e['charged_or_reserved_usd'] for e in ledger)
    report = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan_hash,
        'interpretation': 'Paired mechanism follow-up reusing six previous test cases; not a new blind test. Report six pairs separately for each model; do not pool them as 12 independent datasets.',
        'planned_runs': len(runs), 'completed_runs': complete_count,
        'all_24_complete': len(runs) == 24 and complete_count == 24,
        'states': dict(Counter(r['state'] for r in runs)), 'model_arm_summary': group,
        'paired_summary': pair_group, 'paired_cases': pairs,
        'api_requests': len(ledger),
        'known_cost_usd': sum(e['charged_or_reserved_usd'] for e in ledger if not e.get('billing_unknown', True)),
        'charged_or_reserved_usd': charged,
        'billing_unknown_requests': sum(e.get('billing_unknown', True) for e in ledger),
        'prior_cost_usd': plan['prior_experiment_cost_usd'],
        'project_charged_or_reserved_usd': plan['prior_experiment_cost_usd'] + charged,
        'new_cap_usd': plan['new_paid_cap_usd'], 'total_cap_usd': plan['total_user_authorization_usd'],
        'actual_models': dict(Counter(e.get('returned_model') or 'unknown' for e in ledger)),
        'actual_providers': dict(Counter(e.get('provider') or 'unknown' for e in ledger)),
        'validation': 'Visited scores/parameters, uniqueness, candidate cap, common starts, and completed terminal metrics verified against local assets.',
        'snapshot_note': 'Files are read without locking the live experiment; rerun after all terminals exist for the final report.',
        'input_sha256': loaded}
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / 'runs.csv', runs)
    csv_write(output / 'paired_cases.csv', pairs)
    csv_write(output / 'paired_case_results.csv', pairs)
    csv_write(output / 'trajectories.csv', trajectories)
    csv_write(output / 'request_costs.csv', receipts)
    dump(output / 'summary.json', report)
    def fmt(x):
        return '—' if x is None else f'{x:.6f}'
    lines = ['# 数值反馈配对补充实验', '',
        f"完成 {complete_count}/{len(runs)} 条续跑；{'全部24条完整' if report['all_24_complete'] else '当前为未完成快照，不能当作最终结果'}。",
        '这是复用 H01–H06 的机制补充，每个模型分别有 6 对；不是新增盲测，也不是 12 个独立数据集。', '',
        'AIC差值越小越好；冠军阈值为差值 ≤ 1e−9，近优为 ≤ 2。成功数仅计入完成的续跑，分母保留预定案例数。', '',
        '|模型|分支|已完成/预定|冠军|近优|平均差值（已完成）|中位差值（已完成）|',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in group:
        lines.append(f"|{g['model_label']}|{g['arm']}|{g['completed_cases']}/{g['planned_cases']}|{g['champion_count']}/{g['planned_cases']}|{g['near_count']}/{g['planned_cases']}|{fmt(g['mean_regret_completed'])}|{fmt(g['median_regret_completed'])}|")
    lines += ['', '配对差 = 有反馈差值 − 屏蔽评分差值，负数表示有反馈更好；仅两支均完成时计算。', '',
        '|模型|案例|有反馈状态|屏蔽状态|有反馈差值|屏蔽差值|配对差|', '|---|---|---|---|---:|---:|---:|']
    for p in pairs:
        lines.append(f"|{p['model_label']}|{p['case']}|{p['feedback_state']}|{p['blind_state']}|{fmt(p['feedback_regret'])}|{fmt(p['blind_regret'])}|{fmt(p['regret_feedback_minus_blind'])}|")
    lines.append('')
    for g in pair_group:
        lines.append(f"{g['model_label']}：完成 {g['complete_pairs']}/{g['planned_pairs']} 对；有反馈更好/相同/更差 = {g['feedback_lower_regret']}/{g['ties']}/{g['feedback_higher_regret']}；平均配对差 {fmt(g['mean_paired_regret_difference'])}。")
    lines += ['', f"真实请求记录 {len(ledger)} 次；已知费用 {report['known_cost_usd']:.8f} 美元，含未结算预留 {charged:.8f} 美元；未知账单 {report['billing_unknown_requests']} 条。项目累计已知或预留 {report['project_charged_or_reserved_usd']:.8f} 美元。",
        f"返回模型：{json.dumps(report['actual_models'], ensure_ascii=False)}；服务提供者：{json.dumps(report['actual_providers'], ensure_ascii=False)}。缺失字段标为 unknown。", '',
        '逐次访问轨迹见 trajectories.csv；逐条费用见 request_costs.csv；各条最优组合、实际访问数和未完成状态见 runs.csv。本脚本不读取原始模型回答，不重新拟合，不发起 API。', '']
    (output / '汇总.md').write_text('\n'.join(lines), encoding='utf-8')
    return {k: report[k] for k in ('planned_runs', 'completed_runs', 'all_24_complete', 'states',
                                  'api_requests', 'known_cost_usd', 'billing_unknown_requests')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, default=HERE / 'ablation')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'ablation_analysis')
    args = parser.parse_args()
    print(json.dumps(analyze(args.experiment_dir, args.output_dir), ensure_ascii=False))
