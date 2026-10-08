"""Portable search core; algorithmic definitions copied unchanged from the frozen pilot."""

import json

import re

MEANS = ['Zero','Constant','Linear','Quadratic']

PRIMITIVES = ['SE','MA1','MA3','MA5','RQ','LIN','PER']

CASES = ['EX1','EX2','EX3','EX4','Borehole']

NEW_SYSTEM = '''You select a covariance and mean function for a Gaussian process.
Objective: minimize AIC with at most 20 UNIQUE candidate evaluations. A numerical
evaluator supplies the AIC only for candidates you request. You have four rounds,
with five evaluations per round. Unobserved scores and the optimum are unknown.

The catalog lists ALL 94 legal kernels without scores or ranking. For each,
choose mean 1=Zero, 2=Constant, 3=Linear, or 4=Quadratic. Nonzero means include an
intercept; Quadratic has coordinate-wise squares and no interaction terms.
SE, MA1, MA3, MA5 and RQ are products of ONE-DIMENSIONAL factors across coordinate
directions, with separate direction length scales. They are not radial ARD
Matern/RQ kernels. MA1/MA3/MA5 have smoothness 1/2, 3/2, 5/2. RQ permits multiple
scales. LIN modulates covariance amplitude through a sum of coordinate products;
it is not simply a deterministic linear mean. PER is a product of periodic
direction factors, with a shared standardized period and embedding length.
Sums superpose independent components; products modulate them. Exclusion of
stationary-stationary products is a library restriction, not an identity.

SEARCH RULES:
1. First round: cover contrasting covariance and mean hypotheses. The trend and
   semivariogram profile is a clue, not a proof excluding any kernel family.
2. Later rounds: use about three slots for controlled changes around promising
   evaluated models, and two slots to explore a different plausible structure.
   The supplied neighborhood IDs are legal alternatives differing by one mean,
   primitive, or added/deleted component. They contain NO unseen scores.
3. A poor A+B fit does NOT show that B is unhelpful in every combination. Compare
   components while holding the other structure and mean fixed where possible.
   Similarly, changing kernel and mean together cannot isolate either effect.
4. Avoid duplicate evaluated pairs. Complete all four batches; do not stop early.
5. Return five PRIMARY distinct pairs plus two reserve pairs. The evaluator uses
   the first five legal unvisited pairs in your ordered list; reserves only fill
   invalid or duplicate primary entries. Every evaluated pair consumes budget.
6. Give a brief decision summary, not a long derivation. The summary from your
   previous round will be provided along with ALL evaluated scores.

Return ONE JSON object and no other text:
{"summary":"<=80 words: supported patterns, unresolved comparison, next test",
 "proposals":[{"kernel_id":1,"mean_id":2,"reason":"<=16 words"}, ...]}
Exactly seven ordered proposal objects are requested. Kernel IDs are catalog
indices, not performance ranks. You cannot request the optimum or all scores.
'''

def canonical(expr, names):
    s=str(expr).upper().replace(' ','')
    if s in names:return s
    m=re.fullmatch(r'\(([A-Z0-9]+)\*([A-Z0-9]+)\)\+([A-Z0-9]+)',s)
    if not m:
        rev=re.fullmatch(r'([A-Z0-9]+)\+\(([A-Z0-9]+)\*([A-Z0-9]+)\)',s)
        if rev:m=(rev[2],rev[3],rev[1])
    if not m and '(' not in s:
        m=re.fullmatch(r'([A-Z0-9]+)\*([A-Z0-9]+)\+([A-Z0-9]+)',s)
    if m:
        a,b,c = m.groups() if hasattr(m,'groups') else m
        if any(x not in PRIMITIVES for x in (a,b,c)):return None
        a,b=sorted((a,b),key=PRIMITIVES.index)
        s=f'({a}*{b})+{c}'
    elif ('+' in s) != ('*' in s):
        op='+' if '+' in s else '*'
        parts=s.split(op)
        if len(parts)!=2 or any(p not in PRIMITIVES for p in parts):return None
        s=op.join(sorted(parts,key=PRIMITIVES.index))
    return s if s in names else None

def parse(reply):
    i,j=reply.find('{'),reply.rfind('}')
    return json.loads(reply[i:j+1]) if i>=0 and j>i else {}

def accepted(obj,method,names,seen):
    ans=[]; invalid=[]
    for p in obj.get('proposals',[]):
        try:
            if method=='structured':
                kid=p['kernel_id']; mi=p['mean_id']
                if type(kid)!=int or type(mi)!=int or not (1<=kid<=94 and 1<=mi<=4):raise ValueError()
                expr=names[kid-1]
            else:
                expr=canonical(p.get('kernel',''),names)
                m=str(p.get('mean','')).lower().strip()
                m={'const':'constant','lin':'linear','quad':'quadratic'}.get(m,m)
                mi=[x.lower() for x in MEANS].index(m)+1
                if not expr:raise ValueError()
            key=(expr,mi)
            if key in seen or key in ans:
                invalid.append({'proposal':p,'reason':'duplicate'})
                continue
            if len(ans)<5:ans.append(key)
        except (ValueError,TypeError,KeyError,IndexError):
            invalid.append({'proposal':p,'reason':'invalid'})
    return ans,invalid

def features(expr):
    return re.sub(r'[A-Z]+[0-9]*','_',expr),re.findall(r'[A-Z]+[0-9]*',expr)

def neighbor(a,b):
    ea,ma=a;eb,mb=b
    if ea==eb:return ma!=mb
    if ma!=mb:return False
    sa,pa=features(ea);sb,pb=features(eb)
    if sa==sb and sum(x!=y for x,y in zip(pa,pb))==1:return True
    if len(pa)==1 and len(pb)==2:return pa[0] in pb
    if len(pb)==1 and len(pa)==2:return pb[0] in pa
    if len(pa)==2 and '+' not in ea and len(pb)==3:return eb.startswith('('+ea+')+')
    if len(pb)==2 and '+' not in eb and len(pa)==3:return ea.startswith('('+eb+')+')
    return False

def original_prompt(profile,visited,rd):
    if not visited:
        note='No candidates evaluated yet.' if rd==1 else 'No valid candidates evaluated yet (previous proposals were invalid or duplicates).'
        return f'{profile}\n\n{note} Propose 5 diverse (kernel, mean) candidates covering different structural hypotheses (trend vs stationary, smooth vs rough, with/without periodicity; consider the (A*B)+C template for composite structures).\nThis is round {rd} of at most 4.'
    rows=[];best=min(r['aic'] for r in visited)
    for i,r in enumerate(sorted(visited,key=lambda x:x['aic']),1):
        rows.append(f"{i:2d}. {r['kernel']:<14s} + {r['mean']:<9s}  AIC={r['aic']:9.2f}  dAIC={r['aic']-best:7.2f}  (k={r['k']}, round {r['round']})")
    return f'{profile}\n\nRESULTS AFTER ROUND {rd-1} (sorted by AIC, lower is better):\n'+ '\n'.join(rows)+f'\n\nThis is round {rd} of at most 4. Reflect on the ranking: which structural ingredients (roughness class, LIN/PER components, T3 additive products, mean order) does the evidence favor? Then either propose up to 5 NEW candidates (not in the table) that could lower AIC further, or set "stop":true with your "final" pick.'

def structured_prompt(profile,visited,rd,names,summary):
    catalog='\n'.join(f'{i+1}: {name}' for i,name in enumerate(names))
    visited_keys={(r['kernel'],r['mean_id']) for r in visited}
    result=[]
    best=min((r['aic'] for r in visited),default=0)
    for r in sorted(visited,key=lambda x:x['aic']):
        result.append(f"K{names.index(r['kernel'])+1}/M{r['mean_id']} {r['kernel']} {r['mean']}: AIC={r['aic']:.6f}, delta={r['aic']-best:.6f}, k={r['k']}, fit=OK")
    near=[]
    for r in sorted(visited,key=lambda x:x['aic'])[:2]:
        a=(r['kernel'],r['mean_id'])
        ids=[f'K{i+1}/M{mi}' for i,k in enumerate(names) for mi in range(1,5) if (k,mi) not in visited_keys and neighbor(a,(k,mi))]
        near.append(f"Neighbors of K{names.index(r['kernel'])+1}/M{r['mean_id']}: "+', '.join(ids))
    return f'{profile}\n\nUNRANKED KERNEL CATALOG:\n{catalog}\n\nROUND {rd}/4. Evaluated {len(visited)}/20.\nEVALUATED RESULTS:\n'+ ('\n'.join(result) or 'None yet.')+'\n\nLEGAL ONE-CHANGE NEIGHBORS (no scores):\n'+ ('\n'.join(near) or 'No evaluated anchors yet.')+'\n\nYour previous short decision summary:\n'+(summary or 'None.')+'\n\nReturn seven ordered distinct unvisited proposals: five primary and two reserves. Only the first five valid unvisited pairs are evaluated.'
