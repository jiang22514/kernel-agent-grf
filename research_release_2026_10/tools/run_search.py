"""Optional new API search. Without --allow-paid-api this only previews setup."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/python'))
import search


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['development', 'holdout'], default='holdout')
    parser.add_argument('--case', default='H01')
    parser.add_argument('--method', choices=['control', 'strategy'], default='strategy')
    parser.add_argument('--model', choices=['openai/gpt-6-astra', 'anthropic/claude-fable-5.1'], default='openai/gpt-6-astra')
    parser.add_argument('--allow-paid-api', action='store_true')
    parser.add_argument('--max-usd', type=float)
    parser.add_argument('--run-id', default='new_search')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise SystemExit('Use a simple alphanumeric run ID.')
    assets = search.read(ROOT / 'data' / args.dataset / 'assets.json')
    if args.case not in assets['cases']:
        raise SystemExit('Case not available. Private CPT observations/profile are excluded.')
    print(json.dumps({'mode': 'paid' if args.allow_paid_api else 'preview_only', 'case': args.case,
                      'method': args.method, 'model': args.model, 'candidate_cap': 20,
                      'batch_sizes': search.BATCHES[args.method]}, indent=2))
    if not args.allow_paid_api:
        return
    if args.max_usd is None or not 0 < args.max_usd <= 20:
        raise SystemExit('Paid calls require an explicit --max-usd value in (0,20].')
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise SystemExit('Set OPENROUTER_API_KEY in the environment; never place it in source files.')
    runner = load_module('public_bounded_runner', ROOT / 'code/python/base/runner.py')
    runner.core = load_module('public_search_core', ROOT / 'code/python/base/core.py')
    runner.OUT = ROOT / 'recomputed/api' / args.run_id
    runner.load, runner.dump = search.read, search.dump
    catalog = {r['id']: r for r in search.read(ROOT / 'prompts/model_catalog_snapshot.json')['data']}
    plan = {'max_requests': len(search.BATCHES[args.method]), 'new_paid_cap_usd': args.max_usd,
            'prior_experiment_cost_usd': 0.0, 'total_user_authorization_usd': args.max_usd}
    systems = {k: (ROOT / 'prompts' / (k + '_system.txt')).read_text(encoding='utf-8') for k in ('control', 'strategy')}
    run = {'run_id': args.run_id, 'stage': 'public_replication', 'case': args.case,
           'model': args.model, 'method': args.method, 'reasoning_effort': 'medium', 'initial_pairs': []}
    identity = {**run, 'assets_sha256': search.sha(ROOT / 'data' / args.dataset / 'assets.json'),
                'systems': systems, 'cap_usd': args.max_usd}
    import hashlib
    plan_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with search.exclusive(runner.OUT):
        controller = runner.Controller(plan, catalog)
        result = search.run_one(runner, assets, {}, systems, run, controller, plan_hash)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
