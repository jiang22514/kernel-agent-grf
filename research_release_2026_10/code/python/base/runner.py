from pathlib import Path

import argparse

import concurrent.futures

import hashlib

import importlib.util

import json

import math

import os

import re

import sys

import threading

import time

import urllib.error

import urllib.request

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

class StopRequests(RuntimeError):
    pass

class Controller:
    def __init__(self, plan, catalog):
        self.plan, self.catalog = plan, catalog
        self.lock = threading.Lock()
        self.ledger_path = OUT / 'ledger.json'
        self.ledger = load(self.ledger_path) if self.ledger_path.exists() else []
        self.stop_reason = None
        if any(entry.get('billing_unknown', True) for entry in self.ledger):
            self.stop_reason = 'PRIOR_BILLING_UNKNOWN_OR_INTERRUPTED_REQUEST'

    def reserve(self, run, path, body):
        meta = self.catalog[run['model']]
        pricing = meta['pricing']
        prompt_price = float(pricing['prompt'])
        completion_price = float(pricing['completion'])
        # Conservatively use the largest advertised tier. Profiles are short,
        # but this also avoids accidentally under-reserving tiered models.
        for override in pricing.get('overrides', []):
            prompt_price = max(prompt_price, float(override.get('prompt', prompt_price)))
            completion_price = max(completion_price, float(override.get('completion', completion_price)))
        nbytes = sum(len(item['content'].encode('utf-8')) for item in body['messages']) + 256
        reserve = nbytes * prompt_price + 8192 * completion_price + float(pricing.get('request', 0) or 0)
        with self.lock:
            if self.stop_reason:
                raise StopRequests(self.stop_reason)
            charged = sum(entry['charged_or_reserved_usd'] for entry in self.ledger)
            if len(self.ledger) >= self.plan['max_requests']:
                self.stop_reason = 'REQUEST_CAP_REACHED'
                raise StopRequests(self.stop_reason)
            if (charged + reserve > self.plan['new_paid_cap_usd'] + 1e-12 or
                    charged + reserve + self.plan['prior_experiment_cost_usd'] > self.plan['total_user_authorization_usd'] + 1e-12):
                self.stop_reason = 'BUDGET_RESERVATION_LIMIT_REACHED'
                raise StopRequests(self.stop_reason)
            if path.exists():
                raise StopRequests('REQUEST_FILE_ALREADY_EXISTS')
            dump(path, body)
            entry = {'run_id': run['run_id'], 'request_file': str(path.relative_to(OUT)),
                     'model': run['model'], 'started_unix': time.time(),
                     'reserved_usd': reserve, 'charged_or_reserved_usd': reserve,
                     'billing_unknown': True, 'state': 'in_flight'}
            self.ledger.append(entry)
            dump(self.ledger_path, self.ledger)
            return entry

    def failure(self, entry, exc, elapsed):
        with self.lock:
            entry.update(state='request_failed', error_type=type(exc).__name__, elapsed_seconds=elapsed)
            if isinstance(exc, urllib.error.HTTPError):
                entry['http_status'] = exc.code
            self.stop_reason = 'REQUEST_FAILED_BILLING_UNKNOWN'
            dump(self.ledger_path, self.ledger)

    def returned(self, entry, data, elapsed):
        usage = data.get('usage', {})
        cost = usage.get('cost') if isinstance(usage, dict) else None
        known = (not isinstance(cost, bool) and isinstance(cost, (int, float)) and
                 math.isfinite(cost) and cost >= 0)
        choices = data.get('choices') or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        with self.lock:
            entry.update(state='returned', request_id=data.get('id'), returned_model=data.get('model'),
                         provider=data.get('provider'), usage=usage, elapsed_seconds=elapsed,
                         finish_reason=choice.get('finish_reason'))
            if known:
                entry.update(charged_or_reserved_usd=cost, billing_unknown=False)
            else:
                self.stop_reason = 'RETURNED_BILLING_UNKNOWN'
            dump(self.ledger_path, self.ledger)
        if not known:
            raise StopRequests('RETURNED_BILLING_UNKNOWN')
        content = choice.get('message', {}).get('content') or ''
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('EMPTY_MODEL_OUTPUT')
        return content, dict(entry)

    def call(self, run, messages, path):
        key = os.environ.get('OPENROUTER_API_KEY', '')
        if not key:
            raise StopRequests('ENVIRONMENT_CREDENTIAL_MISSING')
        body = {'model': run['model'], 'messages': messages, 'max_tokens': 8192,
                'provider': {'require_parameters': True}}
        if run['reasoning_effort'] is not None:
            body['reasoning'] = {'effort': run['reasoning_effort']}
        entry = self.reserve(run, path, body)
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                     'X-Title': 'GP-Kernel-Flagship-Development'}, method='POST')
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
            # Preserve received evidence even if it cannot be parsed as JSON.
            path.with_suffix('.response.json').write_bytes(raw)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('NONOBJECT_API_RESPONSE')
        except Exception as exc:
            self.failure(entry, exc, time.time() - started)
            raise StopRequests('REQUEST_FAILED_SEE_LEDGER') from None
        return self.returned(entry, data, time.time() - started)

    def snapshot(self):
        with self.lock:
            return {'requests': len(self.ledger),
                    'cost_known_usd': sum(x['charged_or_reserved_usd'] for x in self.ledger if not x['billing_unknown']),
                    'charged_or_reserved_usd': sum(x['charged_or_reserved_usd'] for x in self.ledger),
                    'billing_unknown_requests': sum(bool(x['billing_unknown']) for x in self.ledger),
                    'stop_reason': self.stop_reason}

def evaluate_pairs(pairs, score, seen, visited, rd):
    if len(pairs) > 5 or len(visited) + len(pairs) > 20:
        raise ValueError('EVALUATION_BUDGET_EXCEEDED')
    for key in pairs:
        if key in seen:
            raise ValueError('DUPLICATE_EVALUATION')
        seen.add(key)
        visited.append({**score[key], 'round': rd})

def grade_after_termination(result, score):
    visited = result['visited']
    optimum = min(row['aic'] for row in score.values())
    best = min(visited, key=lambda row: row['aic']) if visited else None
    gap = best['aic'] - optimum if best else None
    result.update(n_evals=len(visited), best=best, regret=gap,
                  hit=gap is not None and gap <= 1e-9,
                  near_optimal=gap is not None and gap <= 2,
                  regret_trajectory=[min(row['aic'] for row in visited[:i]) - optimum
                                     for i in range(1, len(visited) + 1)])

def run_one(assets, run, control, plan_hash):
    rdout = OUT / 'runs' / run['run_id']
    terminal = rdout / 'terminal.json'
    if terminal.exists():
        return load(rdout / 'result.json')
    # A paid attempt interrupted without a terminal is never silently replayed.
    if rdout.exists():
        return {'run_id': run['run_id'], 'state': 'interrupted_existing_directory'}
    rdout.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = {**run, 'plan_sha256': plan_hash, 'state': 'running',
              'visited': [], 'rounds': [], 'started_unix': started}
    visited, rounds, seen = result['visited'], result['rounds'], set()
    names = assets['kernels']
    data = assets['cases'][run['case']]
    score = {(row['kernel'], row['mean_id']): row for row in data['rows']}
    summary = ''
    dump(rdout / 'result.json', result)
    try:
        first_round = 1
        if run['initial_pairs']:
            pairs = [(p['kernel'], p['mean_id']) for p in run['initial_pairs']]
            evaluate_pairs(pairs, score, seen, visited, 1)
            rounds.append({'round': 1, 'source': 'predeclared_initial_pairs',
                           'new_evals': 5, 'cost_usd': 0, 'elapsed_seconds': 0})
            first_round = 2
            dump(rdout / 'result.json', result)
        for rd in range(first_round, 5):
            system = assets['original_system'] if run['method'] == 'original' else core.NEW_SYSTEM
            user = (core.original_prompt(data['profile'], visited, rd) if run['method'] == 'original'
                    else core.structured_prompt(data['profile'], visited, rd, names, summary))
            reply, receipt = control.call(run, [{'role': 'system', 'content': system},
                                                {'role': 'user', 'content': user}],
                                           rdout / f'round_{rd}.request.json')
            record = {'round': rd, 'source': 'llm', 'reply': reply,
                      'cost_usd': receipt['charged_or_reserved_usd'],
                      'elapsed_seconds': receipt['elapsed_seconds']}
            rounds.append(record)
            dump(rdout / 'result.json', result)
            obj = core.parse(reply)
            if not isinstance(obj, dict) or ('proposals' not in obj and obj.get('stop') is not True):
                raise ValueError('INVALID_SEARCH_RESPONSE_OBJECT')
            if not isinstance(obj.get('proposals', []), list):
                raise ValueError('INVALID_PROPOSAL_LIST')
            pairs, rejected = core.accepted(obj, run['method'], names, seen)
            evaluate_pairs(pairs, score, seen, visited, rd)
            summary = str(obj.get('summary', ''))[:1200]
            record.update(new_evals=len(pairs), rejected=rejected)
            dump(rdout / 'result.json', result)
            if run['method'] == 'original' and obj.get('stop') is True:
                result['stopped_by'] = 'llm_stop'
                break
        result['state'] = 'complete'
    except Exception as exc:
        result.update(state='stopped' if isinstance(exc, StopRequests) else 'failed',
                      error=str(exc), error_type=type(exc).__name__)
    finally:
        # This is the only access to the global optimum and happens after search.
        grade_after_termination(result, score)
        with control.lock:
            entries = [entry for entry in control.ledger if entry['run_id'] == run['run_id']]
            result.update(cost_usd=sum(e['charged_or_reserved_usd'] for e in entries if not e['billing_unknown']),
                          charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                          billing_unknown=any(e['billing_unknown'] for e in entries),
                          api_requests=len(entries))
        result.update(elapsed_seconds=time.time() - started, finished_unix=time.time())
        dump(rdout / 'result.json', result)
        dump(terminal, {key: result.get(key) for key in ('run_id', 'state', 'error', 'n_evals',
                                                      'regret', 'hit', 'near_optimal', 'cost_usd',
                                                      'charged_or_reserved_usd', 'billing_unknown', 'api_requests')})
        print(json.dumps({key: result.get(key) for key in ('run_id', 'state', 'n_evals',
                                                         'regret', 'hit', 'near_optimal', 'cost_usd')},
                         ensure_ascii=False), flush=True)
    return result
OUT = Path(__file__).resolve().parent
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
