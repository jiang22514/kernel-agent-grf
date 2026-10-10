"""Isolate numerical score feedback; prepare/check never send HTTP requests.

All GP scores and the common first batch are read from completed archives.
Only `execute --allow-api --authorized-total-usd 60` can call the API.
"""
from pathlib import Path
from collections import Counter
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HOLDOUT = ROOT / 'output/agent_holdout_test_2026-10-08'
DEV = ROOT / 'output/agent_search_development_2026-10-08'
BASE = ROOT / 'output/lin_offset_revision_2026-10-06/lin_v2_agent/live'
CASES = [f'H{i:02d}' for i in range(1, 7)]
GROUPS = {'astra': ('openai/gpt-6-astra', 'control', 5, 4),
          'fable51': ('anthropic/claude-fable-5.1', 'strategy', 4, 5)}
PRIOR, CAP, TOTAL = 48.15048253902, 8.0, 60.0
ORDER_SEED = 2026101001
SCORE_CLAIM = re.compile(r'\b(?:AIC|dAIC|delta|regret|gap)\s*(?:[=:<>≈~]|is|of|about|approximately)?\s*[+−-]?\d', re.I)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def model_catalog(path):
    catalog = read(path)
    return {row['id']: row for row in catalog['data']} if isinstance(catalog.get('data'), list) else catalog


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod


def engines(live):
    runner = module('feedback_ablation_frozen_runner', BASE / 'base/runner.py')
    runner.OUT = Path(live)
    runner.dump, runner.load = dump, read
    development = module('feedback_ablation_frozen_development', DEV / 'experiment.py')
    return runner, development


def planned_runs():
    # Construct this ordering without inspecting responses, scores or outcomes.
    rng = random.Random(ORDER_SEED)
    blocks = [(case, label) for case in CASES for label in GROUPS]
    rng.shuffle(blocks)
    rows = []
    for case, label in blocks:
        arms = ['feedback', 'blind']
        rng.shuffle(arms)
        model, method, batch, rounds = GROUPS[label]
        for arm in arms:
            rows.append({'run_id': f'{case}__{label}__{arm}', 'case': case,
                         'model_label': label, 'model': model, 'method': method,
                         'arm': arm, 'batch_size': batch, 'total_rounds': rounds,
                         'reasoning_effort': 'medium'})
    return rows


def adapted_system(original, method):
    if method == 'control':
        old = ('A numerical\nevaluator supplies the AIC only for candidates you request.')
        new = ('A numerical\nevaluator scores only candidates you request; whether scores are revealed is specified below.')
        assert original.count(old) == 1
        text = original.replace(old, new)
        old = ('   The supplied neighborhood IDs are legal alternatives differing by one mean,\n'
               '   primitive, or added/deleted component. They contain NO unseen scores.\n')
        assert text.count(old) == 1
        text = text.replace(old, '')
        old = ('   previous round will be provided along with ALL evaluated scores.')
        assert text.count(old) == 1
        text = text.replace(old, '   previous round will be provided along with the permitted candidate history.')
    else:
        old = 'numerical evaluator fits parameters and reveals AIC only for requested pairs.'
        assert original.count(old) == 1
        text = original.replace(old, 'numerical evaluator scores only requested pairs; score visibility is specified below.')
        old = ('  The supplied neighbors are suggestions, not a restriction on the catalogue.\n')
        assert text.count(old) == 1
        text = text.replace(old, '')
    return text + ('\n\nNUMERICAL-FEEDBACK ABLATION (applies equally to both arms):\n'
        'History is listed in candidate-visit order, never in score order. No automatically\n'
        'chosen neighborhood or best-candidate recommendation is supplied. If numerical\n'
        'scores are provided, use them to revise your comparisons. If scores are withheld,\n'
        'rely on the data profile, the candidate identities, and your own modeling hypotheses.\n'
        'Do not infer performance from history order or invent numerical scores. Your short\n'
        'summary is carried forward only within this continuation. Do not present\n'
        'guessed scores as numerical observations.\n'
        'The locally scored best accessed candidate is selected after the search ends.\n')


def visible_rows(visited, arm):
    """Whitelisting makes the blind constructor independent of all score values."""
    result = [{k: row[k] for k in ('kernel', 'mean_id', 'mean', 'k')} | {'fit': 'OK'}
              for row in visited]
    if arm == 'feedback':
        best = min((row['aic'] for row in visited), default=0)
        for out, row in zip(result, visited):
            out.update(aic=row['aic'], delta=row['aic'] - best)
    elif arm != 'blind':
        raise ValueError('UNKNOWN_ARM')
    return result


def safe_summary(summary, arm):
    text = str(summary)[:1200]
    # Both arms retain their own reasoning memory identically. A blind numerical
    # guess is not a leak from the local score table; record it without censoring.
    if arm == 'blind' and SCORE_CLAIM.search(text):
        return text, 'self_generated_numerical_score_claim_retained'
    return text, None


def user_prompt(profile, names, visited, summary, run, rd):
    history = visible_rows(visited, run['arm'])
    summary, _ = safe_summary(summary, run['arm'])
    lines = []
    for i, row in enumerate(history, 1):
        line = (f"Visit {i}: K{names.index(row['kernel']) + 1}/M{row['mean_id']} "
                f"{row['kernel']} {row['mean']}: k={row['k']}, fit={row['fit']}")
        if run['arm'] == 'feedback':
            line += f", AIC={row['aic']:.6f}, delta={row['delta']:.6f}"
        lines.append(line)
    mode = ('Provided for accessed candidates only.' if run['arm'] == 'feedback'
            else 'Withheld: no scores, score differences, rankings, or best-candidate markers are supplied.')
    catalog = '\n'.join(f'{i + 1}: {name}' for i, name in enumerate(names))
    batch = run['batch_size']
    return (profile + '\n\nUNRANKED KERNEL CATALOG:\n' + catalog +
            f"\n\nROUND {rd}/{run['total_rounds']}. Evaluated {len(visited)}/20.\n" +
            'NUMERICAL SCORE VISIBILITY: ' + mode + '\nCANDIDATE HISTORY (visit order):\n' +
            ('\n'.join(lines) or 'None yet.') + '\n\nYour previous short decision summary:\n' +
            (summary or 'None.') + f'\n\nReturn {batch + 2} ordered distinct unvisited proposals: '
            f'{batch} primary and two reserves. Only the first {batch} valid unvisited pairs are evaluated.')


def prior_cost():
    plan = read(HOLDOUT / 'api/plan.json')
    ledger = read(HOLDOUT / 'api/live/ledger.json')
    if any(e.get('billing_unknown', True) for e in ledger):
        raise ValueError('PRIOR_BILLING_UNKNOWN')
    total = plan['prior_experiment_cost_usd'] + sum(e['charged_or_reserved_usd'] for e in ledger)
    if abs(total - PRIOR) > 1e-9:
        raise ValueError('PRIOR_PROJECT_COST_CHANGED')
    return total


def prepare(refresh=False):
    if (HERE / 'plan.json').exists():
        if not refresh:
            return check()
        # Refreezing is for pre-execution corrections only; preserve old plans.
        live = HERE / 'live'
        if ((live / 'ledger.json').exists() and read(live / 'ledger.json')) or list(live.rglob('*.request.json')):
            raise ValueError('CANNOT_REFREEZE_AFTER_ANY_REAL_REQUEST')
        archive = HERE / 'preexecution_plans' / str(time.time_ns())
        archive.mkdir(parents=True)
        shutil.copy2(HERE / 'plan.json', archive / 'plan.json')
        shutil.copy2(HERE / 'plan.sha256', archive / 'plan.sha256')
    runs = planned_runs()
    (HERE / 'source_prompts').mkdir(exist_ok=True)
    (HERE / 'prompts').mkdir(exist_ok=True)
    (HERE / 'live').mkdir(exist_ok=True)
    runner, development = engines(HERE / 'live')
    sources = [Path(__file__).resolve(), HERE / 'test_offline.py',
               BASE / 'base/runner.py', BASE / 'base/core.py', DEV / 'experiment.py',
               HOLDOUT / 'assets.json', HERE / 'catalog_current.json',
               HOLDOUT / 'api/plan.json', HOLDOUT / 'api/live/ledger.json']
    original_prompts = {'control': BASE / 'prompt_structured.txt', 'strategy': DEV / 'prompts/strategy.txt'}
    systems = {}
    for method, src in original_prompts.items():
        copied = HERE / 'source_prompts' / (method + '.txt')
        shutil.copy2(src, copied)
        assert src.read_bytes() == copied.read_bytes()
        target = HERE / 'prompts' / (method + '.txt')
        target.write_text(adapted_system(copied.read_text(encoding='utf-8-sig'), method), encoding='utf-8')
        systems[method] = str(target)
        sources.extend([src, copied, target])
    assets = read(HOLDOUT / 'assets.json')
    common, equality = {}, []
    for case in CASES:
        rows = assets['cases'][case]['rows']
        assert len(rows) == 376
        scores = {(r['kernel'], r['mean_id']): r for r in rows}
        for label, (_, method, batch, _) in GROUPS.items():
            source_id = f'holdout__{case}__{label}__{method}'
            directory = HOLDOUT / 'api/live/runs' / source_id
            result_path, request_path = directory / 'result.json', directory / 'round_1.request.json'
            result, request = read(result_path), read(request_path)
            assert result['state'] == 'complete'
            first = [r for r in result['visited'] if r['round'] == 1]
            assert len(first) == batch and len({(r['kernel'], r['mean_id']) for r in first}) == batch
            for row in first:
                expected = scores[(row['kernel'], row['mean_id'])]
                assert abs(row['aic'] - expected['aic']) <= 1e-10 and row['k'] == expected['k']
            # Verify the archived first decision was elicited before any scores.
            profile, names = assets['cases'][case]['profile'], assets['kernels']
            initial_user = (runner.core.structured_prompt(profile, [], 1, names, '') if method == 'control'
                            else development.strategy_prompt(runner.core, profile, [], 1, names, ''))
            expected_messages = [{'role': 'system', 'content': original_prompts[method].read_text(encoding='utf-8-sig')},
                                 {'role': 'user', 'content': initial_user}]
            assert request['messages'] == expected_messages, 'FIRST_REQUEST_NOT_SCORE_FREE_OR_CHANGED'
            summary = str(runner.core.parse(result['rounds'][0]['reply']).get('summary', ''))[:1200]
            assert not SCORE_CLAIM.search(summary), 'COMMON_SUMMARY_CONTAINS_SCORE_CLAIM'
            key = f'{case}__{label}'
            common[key] = {'source_run_id': source_id, 'first_round': first, 'summary': summary,
                           'first_request_sha256': sha(request_path), 'first_input_had_no_scores': True}
            equality.append({'case': case, 'model_label': label, 'initial_candidates': batch,
                             'common_input_sha256': digest({'profile': profile, 'catalog': names,
                                 'first_round': first, 'summary': summary, 'system': Path(systems[method]).read_text(encoding='utf-8')}),
                             'same_profile_catalog_system_first_candidates_summary': True})
            sources.extend([result_path, request_path])
    dump(HERE / 'common_starts.json', common)
    dump(HERE / 'run_order.json', {'seed': ORDER_SEED, 'runs': runs})
    dump(HERE / 'input_equality.json', equality)
    sources.extend([HERE / 'common_starts.json', HERE / 'run_order.json', HERE / 'input_equality.json'])
    plan = {'schema_version': 'numeric_feedback_ablation_v1', 'purpose': 'Paired supplementary mechanism analysis; not a replacement for the original independent test.',
            'assets_path': str(HOLDOUT / 'assets.json'), 'catalog_path': str(HERE / 'catalog_current.json'),
            'live_directory': str(HERE / 'live'), 'prompts': systems, 'runs': runs,
            'case_order': CASES, 'ordering_seed': ORDER_SEED, 'candidate_cap': 20,
            'max_requests': 84, 'max_workers': 1, 'max_output_tokens_per_request': 8192,
            'new_paid_cap_usd': CAP, 'prior_experiment_cost_usd': prior_cost(),
            'total_user_authorization_usd': TOTAL,
            'authorization_gate': 'Execute requires --allow-api --authorized-total-usd 60; preparation is not authorization.',
            'summary_policy': 'Reuse only the score-free first-round summary; thereafter each branch carries its own summary, identically truncated to 1200 characters. Record but retain any self-generated numerical score claims in the blind branch; hidden scores never enter its inputs.',
            'source_hashes': {str(p.resolve()): sha(p) for p in sources}}
    dump(HERE / 'plan.json', plan)
    (HERE / 'plan.sha256').write_text(sha(HERE / 'plan.json') + '\n', encoding='utf-8')
    return check()


def check():
    plan = read(HERE / 'plan.json')
    if sha(HERE / 'plan.json') != (HERE / 'plan.sha256').read_text().strip():
        raise ValueError('PLAN_HASH_MISMATCH')
    for path, expected in plan['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('SOURCE_HASH_MISMATCH:' + path)
    if (plan['runs'] != planned_runs() or plan['case_order'] != CASES or
        plan['new_paid_cap_usd'] != CAP or plan['total_user_authorization_usd'] != TOTAL or
        abs(plan['prior_experiment_cost_usd'] - PRIOR) > 1e-9 or
        plan['max_requests'] != 84 or plan['candidate_cap'] != 20 or
        plan['max_workers'] != 1 or plan['max_output_tokens_per_request'] != 8192):
        raise ValueError('EXPERIMENT_DESIGN_MISMATCH')
    prior_cost()
    return {'status': 'READY_OFFLINE_ONLY', 'runs': len(plan['runs']), 'paired_cases_per_model': 6,
            'max_new_requests': sum(r['total_rounds'] - 1 for r in plan['runs']),
            'new_cap_usd': CAP, 'prior_cost_usd': PRIOR, 'total_cap_usd': TOTAL,
            'api_calls_in_preparation': 0}


def run_one(plan, run, runner, development, assets, starts, controller, plan_hash):
    directory = Path(plan['live_directory']) / 'runs' / run['run_id']
    directory.mkdir(parents=True, exist_ok=True)
    terminal = directory / 'terminal.json'
    if terminal.exists():
        saved = read(terminal)
        if saved['plan_sha256'] != plan_hash:
            raise ValueError('TERMINAL_IDENTITY_CHANGED')
        return saved
    common = starts[f"{run['case']}__{run['model_label']}"]
    score = {(r['kernel'], r['mean_id']): r for r in assets['cases'][run['case']]['rows']}
    visited = [dict(r) for r in common['first_round']]
    seen = {(r['kernel'], r['mean_id']) for r in visited}
    summary = common['summary']
    result = {**run, 'plan_sha256': plan_hash, 'state': 'running', 'visited': visited,
              'common_start_sha256': digest(common), 'rounds': [
                  {'round': 1, 'source': 'shared_archived_score_free_first_decision', 'new_evals': len(visited), 'new_cost_usd': 0}]}
    try:
        for rd in range(2, run['total_rounds'] + 1):
            user = user_prompt(assets['cases'][run['case']]['profile'], assets['kernels'], visited, summary, run, rd)
            messages = [{'role': 'system', 'content': Path(plan['prompts'][run['method']]).read_text(encoding='utf-8')},
                        {'role': 'user', 'content': user}]
            reply, receipt, replayed = development.request_or_replay(controller, run, messages,
                                                                    directory / f'round_{rd}.request.json')
            obj = runner.core.parse(reply)
            if not isinstance(obj, dict) or not isinstance(obj.get('proposals'), list):
                raise ValueError('INVALID_SEARCH_RESPONSE_OBJECT')
            if run['method'] == 'control':
                pairs, rejected = runner.core.accepted(obj, 'structured', assets['kernels'], seen)
            else:
                pairs, rejected = development.accepted(obj, assets['kernels'], seen, run['batch_size'])
            if len(pairs) > run['batch_size']:
                raise ValueError('ROUND_BUDGET_EXCEEDED')
            runner.evaluate_pairs(pairs, score, seen, visited, rd)
            summary, summary_note = safe_summary(obj.get('summary', ''), run['arm'])
            result['rounds'].append({'round': rd, 'source': 'fresh_continuation', 'new_evals': len(pairs),
                                     'rejected': rejected, 'summary': summary, 'summary_note': summary_note,
                                     'cost_usd': receipt['charged_or_reserved_usd'], 'replayed_locally': replayed})
            dump(directory / 'result.json', result)
        result['state'] = 'complete'
    except PermissionError:
        raise
    except Exception as exc:
        result.update(state='stopped' if isinstance(exc, runner.StopRequests) else 'failed',
                      error=str(exc), error_type=type(exc).__name__)
    runner.grade_after_termination(result, score)
    entries = [e for e in controller.ledger if e['run_id'] == run['run_id']]
    result.update(full_20=len(visited) == 20,
                  success_champion=result['state'] == 'complete' and result['hit'],
                  success_near=result['state'] == 'complete' and result['near_optimal'],
                  api_requests=len(entries),
                  known_cost_usd=sum(e['charged_or_reserved_usd'] for e in entries if not e.get('billing_unknown', True)),
                  charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                  billing_unknown_requests=sum(e.get('billing_unknown', True) for e in entries))
    dump(directory / 'result.json', result)
    compact = {k: v for k, v in result.items() if k not in {'visited', 'rounds', 'regret_trajectory', 'best'}}
    dump(terminal, compact)
    return compact


@contextlib.contextmanager
def exclusive(live):
    lock = Path(live) / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'started_unix': time.time()}, f)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def status(plan=None):
    plan = plan or read(HERE / 'plan.json')
    live = Path(plan['live_directory'])
    ledger = read(live / 'ledger.json') if (live / 'ledger.json').exists() else []
    runs = []
    for r in plan['runs']:
        p = live / 'runs' / r['run_id'] / 'terminal.json'
        runs.append(read(p) if p.exists() else {**r, 'state': 'not_started'})
    charged = sum(e['charged_or_reserved_usd'] for e in ledger)
    return {'states': dict(Counter(r['state'] for r in runs)), 'requests': len(ledger),
            'known_cost_usd': sum(e['charged_or_reserved_usd'] for e in ledger if not e.get('billing_unknown', True)),
            'charged_or_reserved_usd': charged, 'billing_unknown_requests': sum(e.get('billing_unknown', True) for e in ledger),
            'project_charged_or_reserved_usd': PRIOR + charged,
            'remaining_new_cap_usd': max(0, CAP - charged), 'runs': runs}


def execute(allow_api=False, authorized_total=None):
    if not allow_api or authorized_total != TOTAL:
        raise PermissionError('REAL_API_DISABLED: explicit --allow-api --authorized-total-usd 60 required after human authorization')
    check()
    plan = read(HERE / 'plan.json')
    live = Path(plan['live_directory'])
    runner, development = engines(live)
    controller = runner.Controller(plan, model_catalog(plan['catalog_path']))
    if controller.stop_reason:
        raise RuntimeError(controller.stop_reason + ': no requests resubmitted')
    assets, starts = read(plan['assets_path']), read(HERE / 'common_starts.json')
    with exclusive(live):
        for run in plan['runs']:
            if controller.stop_reason:
                break
            result = run_one(plan, run, runner, development, assets, starts, controller, sha(HERE / 'plan.json'))
            report = status(plan)
            dump(live / 'status.json', report)
            print(json.dumps({k: result.get(k) for k in ('run_id', 'state', 'n_evals', 'regret', 'known_cost_usd')}, ensure_ascii=False), flush=True)
    return status(plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'check', 'status', 'execute'])
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--authorized-total-usd', type=float)
    parser.add_argument('--refresh-plan', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.refresh_plan)
    elif args.command == 'check': result = check()
    elif args.command == 'status': result = status()
    else: result = execute(args.allow_api, args.authorized_total_usd)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
