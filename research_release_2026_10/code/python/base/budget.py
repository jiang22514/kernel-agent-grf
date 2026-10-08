"""Unchanged budget-completion policy extracted from the frozen diagnostic."""

import time

import json

OLD_STOP = 'Set "stop":true only when further search is unlikely to lower AIC.'

NEW_STOP = ('Keep "stop":false while fewer than 20 distinct candidates have been evaluated; '
            'use the remaining candidate-access budget even if improvement seems unlikely.')

OLD_ACTION = ('Then either propose up to 5 NEW candidates (not in the table) that could lower AIC '
              'further, or set "stop":true with your "final" pick.')

def messages(assets, run, visited, request_no):
    system = assets['original_system']
    assert system.count(OLD_STOP) == 1
    system = system.replace(OLD_STOP, NEW_STOP)
    rd = request_no + 1  # Initial five candidates are recorded as round 1.
    original = runner.core.original_prompt(assets['cases'][run['case']]['profile'], visited, rd)
    progress = f'This is round {rd} of at most 4.'
    assert original.count(progress) == 1 and original.count(OLD_ACTION) == 1
    remaining = 20-len(visited)
    user = original.replace(progress,
        f'This is API request {request_no} of at most 6. '
        f'{len(visited)} of 20 distinct candidates have been evaluated; {remaining} remain.')
    user = user.replace(OLD_ACTION,
        f'Then propose {min(5, remaining)} legal NEW candidates (not in the table) that could lower AIC further. '
        'Fill all remaining slots for this request with distinct candidates; do not stop because '
        'you expect convergence. If earlier proposals were invalid or repeated, use this request '
        'to continue filling the same total budget of 20 distinct candidates. Keep "stop":false.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def search(assets, run, control, persist=True):
    rdout = OUT / 'runs' / run['run_id']
    if persist:
        rdout.mkdir(parents=True, exist_ok=False)
    score = {(row['kernel'], row['mean_id']): row for row in assets['cases'][run['case']]['rows']}
    visited, rounds, seen = [], [], set()
    start = time.time()
    result = {**run, 'state': 'running', 'visited': visited, 'rounds': rounds, 'started_unix': start}
    def save():
        if persist:
            runner.dump(rdout/'result.json', result)
    try:
        initial = [(pair['kernel'], pair['mean_id']) for pair in run['initial_pairs']]
        runner.evaluate_pairs(initial, score, seen, visited, 1)
        rounds.append({'round': 1, 'source': 'same_predeclared_initial_pairs', 'new_evals': 5,
                       'cost_usd': 0, 'elapsed_seconds': 0})
        save()
        for request_no in range(1, 7):
            if len(visited) == 20:
                break
            path = rdout/f'request_{request_no}.json'
            reply, receipt = control.call(run, messages(assets, run, visited, request_no), path)
            record = {'round': request_no+1, 'source': 'llm', 'reply': reply,
                      'cost_usd': receipt['charged_or_reserved_usd'],
                      'elapsed_seconds': receipt['elapsed_seconds']}
            rounds.append(record)
            save()
            obj = runner.core.parse(reply)
            if not isinstance(obj, dict) or ('proposals' not in obj and obj.get('stop') is not True):
                raise ValueError('INVALID_SEARCH_RESPONSE_OBJECT')
            if not isinstance(obj.get('proposals', []), list):
                raise ValueError('INVALID_PROPOSAL_LIST')
            pairs, rejected = runner.core.accepted(obj, 'original', assets['kernels'], seen)
            allowed = min(5, 20-len(visited))
            deferred = pairs[allowed:]
            pairs = pairs[:allowed]
            runner.evaluate_pairs(pairs, score, seen, visited, request_no+1)
            record.update(new_evals=len(pairs), rejected=rejected,
                          unevaluated_over_budget=deferred,
                          stop_requested=obj.get('stop') is True,
                          early_stop_ignored=obj.get('stop') is True and len(visited)<20)
            save()
        result.update(state='complete', termination='candidate_budget_20' if len(visited)==20 else 'api_request_cap_6')
    except Exception as exc:
        result.update(state='stopped' if isinstance(exc, runner.StopRequests) else 'failed',
                      error=str(exc), error_type=type(exc).__name__)
    finally:
        # Global optimum is used only after generation has ended, for grading.
        runner.grade_after_termination(result, score)
        with control.lock:
            entries = [e for e in control.ledger if e['run_id']==run['run_id']]
            result.update(cost_usd=sum(e['charged_or_reserved_usd'] for e in entries if not e['billing_unknown']),
                          charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                          billing_unknown=any(e['billing_unknown'] for e in entries), api_requests=len(entries))
        result.update(finished_unix=time.time(), elapsed_seconds=time.time()-start,
                      candidate_budget_complete=len(visited)==20)
        save()
        if persist:
            runner.dump(rdout/'terminal.json', {k:result.get(k) for k in
                ('run_id','state','error','termination','n_evals','regret','near_optimal','hit','cost_usd',
                 'charged_or_reserved_usd','billing_unknown','api_requests','candidate_budget_complete')})
            print(json.dumps({k:result.get(k) for k in
                ('run_id','state','n_evals','regret','near_optimal','cost_usd','api_requests')}, ensure_ascii=False), flush=True)
    return result
