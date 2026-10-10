"""Independent prompt/search development. Reuses the frozen scorer and API budget.

No GP fitting. Only visited candidate scores enter messages. Existing paid
requests are replayed locally when complete evidence exists; never re-sent.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
CASES = ['EX1', 'EX2', 'EX3', 'EX4', 'Borehole']
BATCHES = {'control': [5] * 4, 'strategy': [4] * 5, 'enriched': [4] * 5}


def read(path):
    for attempt in range(6):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    """Unique temporary file: retry local sharing locks, never HTTP requests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    for attempt in range(6):
        try:
            temporary.write_text(payload, encoding='utf-8')
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def engine(plan):
    source = Path(plan['base_runner_path'])
    spec = importlib.util.spec_from_file_location('development_frozen_runner', source)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    module.OUT = Path(plan['live_directory'])
    module.dump = dump
    module.load = read
    return module


def validate(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    if plan_path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != sha(plan_path):
        raise ValueError('PLAN_HASH_MISMATCH')
    required = [Path(__file__).resolve(), Path(plan['base_runner_path']),
                Path(plan['base_runner_path']).with_name('core.py'),
                Path(plan['assets_path']), Path(plan['catalog_path']),
                Path(plan['enriched_profiles_path'])]
    required += [Path(p) for p in plan['prompts'].values()]
    frozen = {str(Path(p).resolve()): value for p, value in plan['source_hashes'].items()}
    for path in required:
        if not path.is_absolute() or str(path.resolve()) not in frozen:
            raise ValueError('UNFROZEN_SOURCE:' + str(path))
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError('SOURCE_HASH_MISMATCH:' + path)
    for key in ('new_paid_cap_usd', 'prior_experiment_cost_usd', 'total_user_authorization_usd'):
        if type(plan[key]) not in (float, int) or not math.isfinite(plan[key]) or plan[key] < 0:
            raise ValueError('INVALID_BUDGET:' + key)
    if (plan['new_paid_cap_usd'] > 20 or
        abs(plan['prior_experiment_cost_usd'] - 29.08819053902) > 1e-10 or
        plan['total_user_authorization_usd'] > 49.08819053902 or
        plan['max_output_tokens_per_request'] != 8192 or plan['max_requests'] != 230 or
        plan['candidate_cap'] != 20 or plan['batches'] != BATCHES or
        plan['max_workers'] not in (1, 2)):
        raise ValueError('EXPERIMENT_LIMITS_MISMATCH')
    assets, catalog_doc = read(plan['assets_path']), read(plan['catalog_path'])
    catalog = {row['id']: row for row in catalog_doc['data']}
    profiles = read(plan['enriched_profiles_path'])['profiles']
    names = assets['kernels']
    if len(names) != 94 or len(set(names)) != 94 or set(assets['cases']) != set(CASES):
        raise ValueError('ASSET_SHAPE_MISMATCH')
    if assets.get('kernel_definition') != 'lin_first_coordinate_offset_v2':
        raise ValueError('WRONG_KERNEL_DEFINITION')
    for case in CASES:
        rows = assets['cases'][case]['rows']
        if (len(rows) != 376 or {(r['kernel'], r['mean_id']) for r in rows} !=
            {(k, m) for k in names for m in range(1, 5)} or
            not all(r['ok'] and math.isfinite(r['aic']) for r in rows)):
            raise ValueError('INCOMPLETE_SCORE_TABLE:' + case)
        if not isinstance(profiles.get(case), str) or not 0 < len(profiles[case]) <= 9000:
            raise ValueError('INVALID_AGGREGATE_PROFILE:' + case)
    runs = plan['runs']
    models = {r['model'] for r in runs}
    if len(models) != 2 or len({r['run_id'] for r in runs}) != 50:
        raise ValueError('RUN_PLAN_SHAPE')
    expected = Counter((stage, model, case, method)
                       for stage, methods in [('screen', ['control', 'strategy', 'enriched']),
                                              ('confirmation', ['control', 'selected'])]
                       for model in models for case in CASES for method in methods)
    if Counter((r['stage'], r['model'], r['case'], r['method']) for r in runs) != expected:
        raise ValueError('RUN_PLAN_DESIGN_MISMATCH')
    for run in runs:
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', run['run_id']) or
            run.get('initial_pairs') != [] or run['reasoning_effort'] != 'medium'):
            raise ValueError('INVALID_RUN:' + run['run_id'])
        meta = catalog[run['model']]
        if 'reasoning' not in meta.get('supported_parameters', []):
            raise ValueError('REASONING_UNSUPPORTED')
        efforts = (meta.get('reasoning') or {}).get('supported_efforts')
        if efforts and 'medium' not in efforts:
            raise ValueError('MEDIUM_UNSUPPORTED')
        if any(not math.isfinite(float(meta['pricing'][f])) or float(meta['pricing'][f]) < 0
               for f in ('prompt', 'completion')):
            raise ValueError('INVALID_CATALOG_PRICE')
    return plan, assets, catalog, profiles


def accepted(obj, names, seen, limit):
    """Same legal/duplicate rules as core.accepted, with a four-pair cap."""
    pairs, rejected = [], []
    for proposal in obj.get('proposals', []):
        try:
            kid, mid = proposal['kernel_id'], proposal['mean_id']
            if type(kid) is not int or type(mid) is not int or not (1 <= kid <= len(names) and 1 <= mid <= 4):
                raise ValueError()
            key = (names[kid - 1], mid)
            if key in seen or key in pairs:
                rejected.append({'proposal': proposal, 'reason': 'duplicate'})
            elif len(pairs) < limit:
                pairs.append(key)
        except (ValueError, TypeError, KeyError, IndexError):
            rejected.append({'proposal': proposal, 'reason': 'invalid'})
    return pairs, rejected


def strategy_prompt(core, profile, visited, rd, names, summary, extra=''):
    """Takes visited rows only: no score-table argument is available here."""
    catalog = '\n'.join(f'{i + 1}: {name}' for i, name in enumerate(names))
    seen = {(r['kernel'], r['mean_id']) for r in visited}
    best = min((r['aic'] for r in visited), default=0)
    rows, anchors, kernels = [], [], set()
    for row in sorted(visited, key=lambda r: r['aic']):
        rows.append(f"K{names.index(row['kernel']) + 1}/M{row['mean_id']} {row['kernel']} {row['mean']}: "
                    f"AIC={row['aic']:.6f}, delta={row['aic'] - best:.6f}, k={row['k']}, fit=OK")
        if row['kernel'] not in kernels and len(anchors) < 3:
            kernels.add(row['kernel'])
            anchors.append(row)
    neighbors = []
    for row in anchors:
        key = (row['kernel'], row['mean_id'])
        ids = [f'K{i + 1}/M{m}' for i, kernel in enumerate(names) for m in range(1, 5)
               if (kernel, m) not in seen and core.neighbor(key, (kernel, m))]
        neighbors.append(f"Neighbors of K{names.index(row['kernel']) + 1}/M{row['mean_id']}: " + ', '.join(ids))
    aggregate = '\n\nADDITIONAL TRAINING-DATA AGGREGATES:\n' + extra if extra else ''
    return (profile + aggregate + '\n\nUNRANKED KERNEL CATALOG:\n' + catalog +
            f'\n\nROUND {rd}/5. Evaluated {len(visited)}/20.\nEVALUATED RESULTS:\n' +
            ('\n'.join(rows) or 'None yet.') + '\n\nLEGAL ONE-CHANGE NEIGHBORS (no scores):\n' +
            ('\n'.join(neighbors) or 'No evaluated anchors yet.') +
            '\n\nYour previous short decision summary:\n' + (summary or 'None.') +
            '\n\nReturn six ordered distinct unvisited proposals: four primary and two reserves. '
            'Only the first four valid unvisited pairs are evaluated.')


def request_or_replay(controller, run, messages, path):
    """A previously written request is never submitted to HTTP again."""
    if not path.exists():
        return (*controller.call(run, messages, path), False)
    body = read(path)
    if (body['model'] != run['model'] or body['messages'] != messages or
        body['max_tokens'] != 8192 or body.get('reasoning') != {'effort': 'medium'}):
        raise ValueError('REPLAY_REQUEST_MISMATCH')
    entries = [e for e in controller.ledger if
               e['run_id'] == run['run_id'] and Path(e['request_file']).name == path.name]
    if len(entries) != 1 or entries[0].get('billing_unknown', True) or entries[0]['state'] != 'returned':
        raise RuntimeError('EXISTING_REQUEST_NEEDS_MANUAL_RECONCILIATION_NO_RESEND')
    data = read(path.with_suffix('.response.json'))
    entry = entries[0]
    if data.get('id') != entry.get('request_id') or data.get('usage', {}).get('cost') != entry['charged_or_reserved_usd']:
        raise ValueError('REPLAY_RECEIPT_MISMATCH')
    content = (data.get('choices') or [{}])[0].get('message', {}).get('content') or ''
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('EMPTY_MODEL_OUTPUT')
    return content, entry, True


def run_one(runner, assets, profiles, systems, run, controller, plan_hash, selection_hash=None):
    directory = runner.OUT / 'runs' / run['run_id']
    terminal = directory / 'terminal.json'
    if terminal.exists():
        saved = read(terminal)
        if saved['plan_sha256'] != plan_hash or saved.get('selection_sha256') != selection_hash:
            raise ValueError('TERMINAL_IDENTITY_CHANGED')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    old = read(result_path) if result_path.exists() else {}
    if old and (old['plan_sha256'] != plan_hash or old.get('selection_sha256') != selection_hash):
        raise ValueError('RUN_IDENTITY_CHANGED')
    started = time.time()
    result = {**run, 'plan_sha256': plan_hash, 'selection_sha256': selection_hash,
              'state': 'running', 'started_unix': old.get('started_unix', started),
              'visited': [], 'rounds': []}
    names, data = assets['kernels'], assets['cases'][run['case']]
    score = {(r['kernel'], r['mean_id']): r for r in data['rows']}
    seen, summary = set(), ''
    try:
        for rd, limit in enumerate(BATCHES[run['method']], 1):
            if run['method'] == 'control':
                user = runner.core.structured_prompt(data['profile'], result['visited'], rd, names, summary)
                system = systems['control']
            else:
                user = strategy_prompt(runner.core, data['profile'], result['visited'], rd, names, summary,
                                       profiles[run['case']] if run['method'] == 'enriched' else '')
                system = systems['strategy']
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
            reply, receipt, replayed = request_or_replay(controller, run, messages,
                                                       directory / f'round_{rd}.request.json')
            obj = runner.core.parse(reply)
            if not isinstance(obj, dict):
                raise ValueError('INVALID_SEARCH_RESPONSE_OBJECT')
            if run['method'] == 'control':
                if 'proposals' not in obj and obj.get('stop') is not True:
                    raise ValueError('INVALID_SEARCH_RESPONSE_OBJECT')
                if not isinstance(obj.get('proposals', []), list):
                    raise ValueError('INVALID_PROPOSAL_LIST')
                pairs, rejected = runner.core.accepted(obj, 'structured', names, seen)
            else:
                if not isinstance(obj.get('proposals'), list):
                    raise ValueError('INVALID_PROPOSAL_LIST')
                pairs, rejected = accepted(obj, names, seen, limit)
            runner.evaluate_pairs(pairs, score, seen, result['visited'], rd)
            summary = str(obj.get('summary', ''))[:1200]
            result['rounds'].append({'round': rd, 'source': 'llm', 'reply': reply,
                                     'new_evals': len(pairs), 'rejected': rejected,
                                     'cost_usd': receipt['charged_or_reserved_usd'],
                                     'elapsed_seconds': receipt['elapsed_seconds'], 'replayed_locally': replayed})
            dump(result_path, result)
        result['state'] = 'complete'
    except PermissionError:
        # A persistent file lock is an interrupted run, not a model failure.
        raise
    except Exception as exc:
        result.update(state='stopped' if isinstance(exc, runner.StopRequests) else 'failed',
                      error=str(exc), error_type=type(exc).__name__)
    runner.grade_after_termination(result, score)
    with controller.lock:
        entries = [e for e in controller.ledger if e['run_id'] == run['run_id']]
        result.update(api_requests=len(entries),
                      cost_usd=sum(e['charged_or_reserved_usd'] for e in entries if not e['billing_unknown']),
                      charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                      billing_unknown=any(e['billing_unknown'] for e in entries))
    active = time.time() - started
    result.update(finished_unix=time.time(), active_seconds_this_execution=active,
                  elapsed_seconds=old.get('elapsed_seconds', 0) + active)
    dump(result_path, result)
    fields = ('run_id', 'stage', 'case', 'model', 'method', 'state', 'error', 'error_type',
              'n_evals', 'regret', 'hit', 'near_optimal', 'cost_usd', 'api_requests',
              'billing_unknown', 'plan_sha256', 'selection_sha256')
    compact = {k: result.get(k) for k in fields}
    dump(terminal, compact)
    return compact


def screen_digest(plan):
    paths = {}
    for run in plan['runs']:
        if run['stage'] != 'screen':
            continue
        directory = Path(plan['live_directory']) / 'runs' / run['run_id']
        terminal = read(directory / 'terminal.json')
        if terminal['state'] not in ('complete', 'failed', 'stopped'):
            raise ValueError('SCREEN_NOT_TERMINATED')
        paths[run['run_id']] = sha(directory / 'result.json')
    digest = hashlib.sha256(json.dumps(paths, sort_keys=True).encode()).hexdigest()
    return digest


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / 'RUNNING.lock').open('a+b')
    try:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.write(b'1')
        handle.flush()
        yield
    finally:
        handle.close()


def prepare(plan_path):
    plan, assets, _, profiles = validate(plan_path)
    runner = engine(plan)
    live = Path(plan['live_directory'])
    receipt = {'status': 'PREPARED', 'plan_sha256': sha(plan_path),
               'sources': plan['source_hashes'], 'created_unix': time.time()}
    if (live / 'prepared.json').exists():
        if read(live / 'prepared.json')['plan_sha256'] != receipt['plan_sha256']:
            raise ValueError('LIVE_BOUND_TO_ANOTHER_PLAN')
        return check(plan_path)
    samples = {}
    for method in BATCHES:
        profile = assets['cases']['EX1']['profile']
        samples[method] = (runner.core.structured_prompt(profile, [], 1, assets['kernels'], '')
                           if method == 'control' else strategy_prompt(runner.core, profile, [], 1,
                           assets['kernels'], '', profiles['EX1'] if method == 'enriched' else ''))
    dump(live / 'prompt_examples.json', samples)
    dump(live / 'prepared.json', receipt)
    return {'status': 'PREPARED', 'plan_sha256': receipt['plan_sha256'], 'runs': 50, 'max_requests': 230}


def check(plan_path):
    plan, _, _, _ = validate(plan_path)
    if read(Path(plan['live_directory']) / 'prepared.json')['plan_sha256'] != sha(plan_path):
        raise ValueError('PREPARED_PLAN_MISMATCH')
    return {'status': 'CHECK_PASS', 'plan_sha256': sha(plan_path), 'runs': 50, 'max_requests': 230}


def execute(plan_path, stage):
    check(plan_path)
    plan, assets, catalog, profiles = validate(plan_path)
    runner = engine(plan)
    systems = {k: Path(p).read_text(encoding='utf-8-sig') for k, p in plan['prompts'].items()}
    plan_hash, selection_hash = sha(plan_path), None
    runs = [dict(r) for r in plan['runs'] if r['stage'] == stage]
    if stage == 'confirmation':
        selection = read(plan['selection_path'])
        if (selection['selected_method'] not in ('strategy', 'enriched') or
            selection['screen_plan_sha256'] != plan_hash or
            selection['screen_results_sha256'] != screen_digest(plan)):
            raise ValueError('INVALID_OR_STALE_SELECTION')
        selection_hash = sha(plan['selection_path'])
        for run in runs:
            if run['method'] == 'selected':
                run['planned_method'] = 'selected'
                run['method'] = selection['selected_method']
    live = runner.OUT
    with exclusive(live):
        controller = runner.Controller(plan, catalog)
        state = {'status': 'RUNNING', 'stage': stage, 'pid': os.getpid(), 'started_unix': time.time(),
                 'plan_sha256': plan_hash, 'selection_sha256': selection_hash, 'completed': []}
        dump(live / 'status.json', state)
        try:
            def worker(run):
                if controller.stop_reason:
                    return {'run_id': run['run_id'], 'state': 'not_started', 'reason': controller.stop_reason}
                return run_one(runner, assets, profiles, systems, run, controller, plan_hash, selection_hash)
            with ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
                futures = [pool.submit(worker, run) for run in runs]
                for future in as_completed(futures):
                    compact = future.result()
                    state['completed'].append(compact)
                    state['budget'] = controller.snapshot()
                    dump(live / 'status.json', state)
                    print(json.dumps(compact, ensure_ascii=False), flush=True)
            state.update(status='STOPPED' if controller.stop_reason else 'TERMINATED', finished_unix=time.time(),
                         budget=controller.snapshot())
            if stage == 'screen' and all(r['state'] != 'not_started' for r in state['completed']):
                state['screen_results_sha256'] = screen_digest(plan)
            dump(live / 'status.json', state)
        except Exception as exc:
            state.update(status='INTERRUPTED', error_type=type(exc).__name__, error=str(exc),
                         finished_unix=time.time(), budget=controller.snapshot())
            dump(live / 'status.json', state)
            raise
    return state


def status(plan_path):
    plan = read(plan_path)
    live = Path(plan['live_directory'])
    state = read(live / 'status.json') if (live / 'status.json').exists() else {'status': 'NOT_STARTED'}
    # Never inspect a running run's result or provider response.
    ledger = read(live / 'ledger.json') if (live / 'ledger.json').exists() else []
    result = {k: v for k, v in state.items() if k != 'completed'}
    result['counts'] = dict(Counter(r['state'] for r in state.get('completed', [])))
    result['requests'] = len(ledger)
    result['known_new_cost_usd'] = sum(e['charged_or_reserved_usd'] for e in ledger if not e['billing_unknown'])
    result['charged_or_reserved_usd'] = sum(e['charged_or_reserved_usd'] for e in ledger)
    result['prior_cost_usd'] = plan['prior_experiment_cost_usd']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'run', 'status'])
    parser.add_argument('--plan', type=Path, default=HERE / 'plan.json')
    parser.add_argument('--stage', choices=['screen', 'confirmation'], default='screen')
    args = parser.parse_args()
    if args.command == 'run':
        value = execute(args.plan.resolve(), args.stage)
        value = {k: v for k, v in value.items() if k != 'completed'}
    else:
        value = globals()[args.command](args.plan.resolve())
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
