"""Retrospective process analysis using every frozen independent-test search.

Comparison: retain each agent's exact first batch, then use either its recorded
continuation or uniformly random unvisited candidates to the same total of 20.
All random probabilities and expected gaps are calculated exactly from the
fixed finite library. This is not a no-feedback LLM ablation.
"""
from pathlib import Path
from math import comb
import csv,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
EXP=ROOT/'output/agent_holdout_test_2026-10-08'
assets=json.loads((EXP/'assets.json').read_text(encoding='utf-8-sig'))
groups=[('astra','control'),('astra','strategy'),('fable51','strategy')]
rows=[]
for model,method in groups:
    for case in [f'H{i:02}' for i in range(1,7)]:
        path=EXP/'api/live/runs'/f'holdout__{case}__{model}__{method}'/'result.json'
        run=json.loads(path.read_text(encoding='utf-8-sig'))
        assert run['state']=='complete' and len(run['visited'])==20
        data=assets['cases'][case]['rows']
        assert len(data)==376 and all(r['ok'] for r in data)
        scores={(r['kernel'],r['mean_id']):r['aic'] for r in data}
        optimum=min(scores.values())
        initial=[r for r in run['visited'] if r['round']==1]
        first_keys={(r['kernel'],r['mean_id']) for r in initial}
        assert len(first_keys)==len(initial)==(5 if method=='control' else 4)
        first_gap=min(r['aic'] for r in initial)-optimum
        final_gap=min(r['aic'] for r in run['visited'])-optimum
        remaining=sorted(v-optimum for k,v in scores.items() if k not in first_keys)
        N,b=len(remaining),20-len(initial)
        denominator=comb(N,b)
        def hit_prob(threshold):
            if first_gap<=threshold:return 1.0
            h=sum(v<=threshold for v in remaining)
            return 1-comb(N-h,b)/denominator if N-h>=b else 1.0
        probabilities=[comb(N-j-1,b-1)/denominator for j in range(N-b+1)]
        assert abs(sum(probabilities)-1)<1e-12
        expected_gap=sum(p*min(first_gap,gap) for p,gap in zip(probabilities,remaining))
        rows.append({'case':case,'model':model,'procedure':method,'first_evals':len(initial),
            'first_gap':first_gap,'final_gap':final_gap,'gap_reduction':first_gap-final_gap,
            'first_near':first_gap<=2,'final_near':final_gap<=2,
            'first_champion':first_gap<=1e-9,'final_champion':final_gap<=1e-9,
            'random_near_probability':hit_prob(2),'random_champion_probability':hit_prob(1e-9),
            'random_expected_final_gap':expected_gap,
            'random_probability_matching_agent':hit_prob(final_gap+1e-9),
            'run_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
with (OUT/'continuation_case_results.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for model,method in groups:
    sub=[r for r in rows if (r['model'],r['procedure'])==(model,method)]
    summary[model+'_'+method]={'cases':len(sub),
        **{k:sum(r[k] for r in sub) for k in ['first_near','final_near','first_champion','final_champion']},
        **{k:float(np.mean([r[k] for r in sub])) for k in ['first_gap','final_gap','gap_reduction','random_near_probability','random_champion_probability','random_expected_final_gap']}}
(OUT/'continuation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
