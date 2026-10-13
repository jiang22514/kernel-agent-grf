"""Unchanged pure functions extracted from frozen research source; no API entry point."""
import json
import re

MEANS = ['Zero','Constant','Linear','Quadratic']


PRIMITIVES = ['SE','MA1','MA3','MA5','RQ','LIN','PER']


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
