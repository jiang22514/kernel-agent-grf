"""Plot all six paired cases per model; no fitted scores or outcomes are altered."""
from pathlib import Path
import argparse
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=HERE/'ablation_analysis/paired_case_results.csv')
parser.add_argument('--output', type=Path, default=HERE/'ablation_analysis')
args = parser.parse_args()
with args.input.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 12 and all(r['pair_complete'].lower() == 'true' for r in rows)
for r in rows:
    for key in ('feedback_gap', 'blind_gap', 'first_gap'):
        r[key] = float(r[key])
        assert math.isfinite(r[key]) and r[key] >= -1e-9
groups = [('astra', 'a  Astra · structured search'),
          ('fable51', 'b  Fable · competing-structure search')]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
WIDTH_MM, HEIGHT_MM = 183.0, 93.0
fig, axs = plt.subplots(1, 2, figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), sharex=True, sharey=True)
fig.subplots_adjust(left=.077, right=.98, bottom=.22, top=.80, wspace=.22)
maxgap = max(max(r['feedback_gap'], r['blind_gap']) for r in rows)
xmax = max(6, math.ceil(maxgap/2)*2) * 1.06
plotted = []
for ax, (model, title) in zip(axs, groups):
    data = sorted([r for r in rows if r.get('model_label', r.get('model')) == model], key=lambda r:r['case'])
    assert [r['case'] for r in data] == [f'H{i:02}' for i in range(1, 7)]
    ax.axvspan(0, 2, color='#DCE8DB', alpha=.75, zorder=0)
    ax.axvline(2, color='#637C63', linestyle=':', linewidth=.85, zorder=1)
    for i, row in enumerate(data):
        fb, blind = row['feedback_gap'], row['blind_gap']
        ax.plot([blind, fb], [i+.12, i-.12], color='#89939B', linewidth=.95, zorder=2)
        ax.scatter([blind], [i+.12], s=34, marker='^', facecolor='#C2783B', edgecolor='white', linewidth=.5, zorder=3)
        ax.scatter([fb], [i-.12], s=33, marker='o', facecolor='#2D638B', edgecolor='white', linewidth=.5, zorder=4)
        plotted.append({'model':model,'case':row['case'],'feedback_gap':fb,'blind_gap':blind})
    ax.set_title(title, loc='left', fontweight='bold', pad=11)
    ax.set_yticks(range(6), [f'H{i:02}' for i in range(1, 7)])
    ax.set_ylim(5.55, -.55)
    # Keep zero and the near-optimal region readable alongside larger misses.
    ax.set_xscale('symlog', linthresh=2, linscale=.8)
    ax.set_xlim(-.14, xmax)
    ax.set_xticks([v for v in [0,1,2,5,10,20,40,80,160] if v <= xmax])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('AIC gap to library minimum (smaller is better)', labelpad=7)
    ax.grid(axis='x', color='#E3E7EA', linewidth=.55)
    ax.tick_params(axis='both', length=3, labelleft=True)
    near_fb = sum(r['feedback_gap'] <= 2 for r in data)
    near_blind = sum(r['blind_gap'] <= 2 for r in data)
    ax.text(.02, -.255, f'Near-optimal: feedback {near_fb}/6  |  masked {near_blind}/6',
            transform=ax.transAxes, fontsize=9, ha='left', va='top')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#2D638B',markersize=6,label='Numerical feedback'),
                    Line2D([],[],marker='^',color='none',markerfacecolor='#C2783B',markersize=6,label='Scores masked'),
                    Patch(facecolor='#DCE8DB',label='Near-optimal region (gap ≤ 2)')],
           loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=9,
           handlelength=1.3,columnspacing=1.5)
args.output.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output/'paired_feedback_ablation.pdf', facecolor='white')
fig.savefig(args.output/'paired_feedback_ablation.svg', facecolor='white')
fig.savefig(args.output/'paired_feedback_ablation.png', dpi=600, facecolor='white')
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
for t in fig.findobj(matplotlib.text.Text):
    if not t.get_visible() or not t.get_text(): continue
    b=t.get_window_extent(renderer)
    if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:
        outside.append(t.get_text())
assert not outside, outside
(args.output/'figure_source_check.json').write_text(json.dumps({'pairs':len(plotted),'models':2,'cases_per_model':6,
    'points':24,'input':str(args.input),'axis':'AIC gap, linear to 2 and logarithmic above 2; lower is better','all_pairs_included':True,
    'no_clipped_text':True,'plotted_data':plotted},indent=2),encoding='utf-8')
print('Created paired_feedback_ablation PDF/SVG/600dpi PNG from all 12 pairs.')
