"""Unchanged pure functions extracted from frozen research source; no API entry point."""
import json
import re

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
