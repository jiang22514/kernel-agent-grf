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
BATCHES = {'control': [5]*4, 'strategy': [4]*5, 'enriched': [4]*5}
