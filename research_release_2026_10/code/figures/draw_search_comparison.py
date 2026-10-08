"""Two-panel search comparison; the only scientific input is the summary JSON.

Default: print a plan. --render is allowed only after all 50 searches terminate.
No experiment, API, fitting, source-data modification or automatic retry occurs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'analysis/results_summary.json'
STAGES = [('screen', ['control', 'enriched', 'strategy']),
          ('confirmation', ['control', 'strategy'])]
NAMES = {'control': 'Control', 'enriched': 'Richer\nsummary', 'strategy': 'Revised\nstrategy'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def inspect(data):
    assert data['planned'] == 50 and len(data['runs']) == 50
    assert len({r['run_id'] for r in data['runs']}) == 50
    assert data['terminal'] == sum(r['terminal'] is True for r in data['runs'])
    table = []
    for stage, methods in STAGES:
        for method in methods:
            summaries = [r for r in data['summaries'] if r['stage'] == stage
                         and r['method'] == method and r['model'] == 'both']
            assert len(summaries) == 1, (stage, method)
            row = summaries[0]
            runs = [r for r in data['runs'] if r['stage'] == stage and r['method'] == method]
            assert len(runs) == row['planned'] == 10
            assert len({(r['model_label'], r['case']) for r in runs}) == 10
            assert {r['model_label'] for r in runs} == {'astra', 'fable51'}
            assert {r['case'] for r in runs} == {'EX1', 'EX2', 'EX3', 'EX4', 'Borehole'}
            complete = [r for r in runs if r['terminal'] and r['state'] == 'complete']
            assert row['completed'] == len(complete)
            assert row['near'] == sum(r['near'] is True for r in complete)
            assert row['champion'] == sum(r['champion'] is True for r in complete)
            assert 0 <= row['champion'] <= row['near'] <= row['completed'] <= 10
            assert all(0 <= r['n_evals'] <= 20 for r in runs)
            table.append({k: row[k] for k in ['stage', 'method', 'planned', 'completed', 'near', 'champion']})
    return table


def render(data, table, source_bytes):
    assert data['terminal'] == 50 and all(r['terminal'] for r in data['runs']), 'WAIT: 50 terminal searches required'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.75,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'savefig.facecolor': 'white',
    })
    colors = {'near': '#527DA5', 'champion': '#CB965D'}
    fig, axes = plt.subplots(1, 2, figsize=(7.20472440945, 3.58267716535), sharey=True,
                             gridspec_kw={'width_ratios': [1.5, 1]})
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.77, wspace=0.22)
    titles = ['Screening: method selection', 'Confirmation: fresh calls']
    for index, (axis, (stage, methods)) in enumerate(zip(axes, STAGES)):
        rows = [next(r for r in table if r['stage'] == stage and r['method'] == m) for m in methods]
        positions = np.arange(len(methods))
        for key, shift in [('near', -0.18), ('champion', 0.18)]:
            for x, row in zip(positions, rows):
                value = row[key]/row['planned']*100
                axis.bar(x+shift, value, width=0.32, color=colors[key], linewidth=0,
                         hatch='' if key == 'near' else '//', zorder=3)
                inside = value >= 95
                axis.text(x+shift, value-4 if inside else value+2.5,
                          f'{row[key]}/{row["planned"]}', ha='center',
                          va='top' if inside else 'bottom', fontsize=8,
                          color='white' if inside else '#222222', clip_on=False)
        axis.set_ylim(0, 100)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_xlim(-0.6, len(methods)-0.4)
        axis.set_xticks(positions, [NAMES[m] for m in methods])
        axis.tick_params(axis='x', length=0, pad=7)
        axis.yaxis.grid(True, color='#E3E3E3', linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.set_title(titles[index], loc='left', fontsize=9, pad=16)
        axis.text(-0.115, 1.09, chr(97+index), transform=axis.transAxes,
                  fontsize=10, fontweight='bold', ha='left', va='bottom')
    axes[0].set_ylabel('Completed success (%)')
    handles = [Patch(facecolor=colors['near'], label='Near-optimal (AIC gap ≤2)'),
               Patch(facecolor=colors['champion'], hatch='//', label=r'Exact champion (gap $\leq10^{-9}$)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.99),
               ncol=2, frameon=False, columnspacing=2, handlelength=1.5)
    fig.text(0.53, 0.065, 'Each method: two models × five cases; denominator = 10 planned searches',
             ha='center', va='center', fontsize=8, color='#444444')
    fig.canvas.draw()
    stem = HERE/'search_development_comparison'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)

    caption = (
        'Figure | Covariance-search development under a budget of at most 20 distinct candidates per search. '
        '(a) Screening compared the current control, enriched data summaries and revised search strategy, '
        'with ten searches per method, and was used for method selection. '
        '(b) Confirmation compares the selected revised strategy with the control using new API calls '
        'on the same five development cases. Each method contains one search by each of Astra and Fable '
        'on every case. Blue bars show completed near-optimal searches (AIC gap ≤2); hatched ochre bars '
        'show completed exact-champion recoveries (gap ≤10⁻⁹). Champions are included among near-optimal '
        'outcomes. Labels give success count/planned count, and all planned searches remain in the '
        'denominator. The common vertical scale is 0–100%. These descriptive frequencies measure '
        'performance on the same cases; confirmation does not introduce new datasets. '
        'No significance test or uncertainty interval is shown. '
        'Completed searches by stage and method: '+ '; '.join(
            f'{r["stage"]}/{r["method"]} {r["completed"]}/{r["planned"]}' for r in table)+'.\n')
    (HERE/'caption.txt').write_text(caption, encoding='utf-8')
    (HERE/'summary_source.json').write_bytes(source_bytes)
    with (HERE/'plotted_counts.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    artifacts = [stem.with_suffix('.'+s) for s in ['pdf', 'svg', 'png']]
    artifacts += [HERE/'caption.txt', HERE/'summary_source.json', HERE/'plotted_counts.csv']
    receipt = {'status': 'RENDERED_PENDING_VISUAL_QA', 'source': str(SOURCE),
               'source_sha256': sha_bytes(source_bytes), 'script_sha256': sha_bytes(Path(__file__).read_bytes()),
               'planned': 50, 'terminal': 50, 'figure_dimensions_mm': [183, 91],
               'denominator': 'all 10 planned searches per method', 'raw_runs_excluded': 0,
               'artifacts': [{'path': str(p), 'sha256': sha_bytes(p.read_bytes())} for p in artifacts]}
    (HERE/'render_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes.decode('utf-8-sig'))
    table = inspect(data)
    if not args.render:
        print(json.dumps({'mode': 'plan', 'terminal': data['terminal'], 'required': 50,
                          'ready_to_render': data['terminal'] == 50,
                          'panels': STAGES, 'source_sha256': sha_bytes(source_bytes)}, indent=2))
        return
    if data['terminal'] != 50:
        raise SystemExit('WAIT: no figure rendered; all 50 searches must be terminal.')
    receipt = render(data, table, source_bytes)
    print(json.dumps({'status': receipt['status'], 'source_sha256': receipt['source_sha256'],
                      'receipt': str(HERE/'render_receipt.json')}, indent=2))


if __name__ == '__main__':
    main()
