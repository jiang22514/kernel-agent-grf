"""Regenerate the six new synthetic realizations and compare released CSVs."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/python'))
from synthetic_generator import make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Also write reproduced CSVs under recomputed/.')
    args = parser.parse_args()
    rows = []
    for i in range(1, 7):
        case = f'H{i:02d}'
        x, y, _ = make_case(i)
        generated = np.column_stack([x, y])
        published = np.loadtxt(ROOT / 'data/holdout' / (case + '.csv'), delimiter=',', skiprows=1)
        error = float(np.max(np.abs(generated - published)))
        if not np.allclose(generated, published, rtol=0, atol=1e-11):
            raise ValueError('GENERATOR_MISMATCH:' + case)
        rows.append({'case': case, 'observations': len(y), 'max_absolute_error': error})
        if args.write:
            out = ROOT / 'recomputed/synthetic_data'
            out.mkdir(parents=True, exist_ok=True)
            np.savetxt(out / (case + '.csv'), generated, delimiter=',', header='x1,x2,y', comments='')
    print(json.dumps({'status': 'SIX_GENERATORS_REPRODUCED', 'cases': rows, 'API_calls': 0, 'GP_fits': 0}, indent=2))


if __name__ == '__main__':
    main()
