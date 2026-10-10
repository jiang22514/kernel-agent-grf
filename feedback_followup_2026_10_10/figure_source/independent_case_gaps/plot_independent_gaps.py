"""Reproduce all 24 independent-test outcomes from the included full-precision CSV."""
from pathlib import Path
import csv
import json
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT/'regenerated'
FIG.mkdir(exist_ok=True)
with (OUT/'independent_search_case_gaps.csv').open(encoding='utf-8',newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 24
for row in rows:
    row['aic_gap'] = float(row['aic_gap'])
    row['champion'] = row['champion'] == 'True'
    row['near_optimal'] = row['near_optimal'] == 'True'
    assert row['champion'] == (row['aic_gap'] <= 1e-9)
    assert row['near_optimal'] == (row['aic_gap'] <= 2)
methods = [('astra', 'control'), ('astra', 'strategy'), ('fable51', 'strategy'), ('greedy', 'width_two')]
names = ['GPT-6 Astra\nStructured', 'GPT-6 Astra\nCompeting structures',
         'Claude Fable 5.1\nCompeting structures', 'Width-two\nGreedy']
colors = ['#28618B', '#628FB2', '#B16636', '#676D73']
markers = ['o', 's', 'D', '^']
cases = [f'H{i:02}' for i in range(1,7)]
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8.5, 'xtick.labelsize':8,
    'ytick.labelsize':7.7, 'axes.linewidth':0.7, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
fig, axs = plt.subplots(2,3,figsize=(7.2047244,4.4094488))  # 183 x 112 mm
fig.subplots_adjust(left=.215,right=.985,bottom=.14,top=.94,wspace=.22,hspace=.42)
for ci,(case,ax) in enumerate(zip(cases,axs.flat)):
    ax.axvspan(0,2,facecolor='#EAF1F4',zorder=0)
    ax.axvline(2,color='#526976',linestyle=(0,(3,2)),linewidth=.8,zorder=1)
    for mi,row in enumerate(rows[ci*4:ci*4+4]):
        gap=row['aic_gap']; y=3-mi
        ax.hlines(y,0,gap,color=colors[mi],linewidth=1.3,zorder=2)
        ax.scatter(gap,y,s=24,marker=markers[mi],facecolor=colors[mi],edgecolor='white',linewidth=.4,zorder=3)
        # Full five decimal places are essential for H05's just-above-two gap.
        text='0' if row['champion'] else (f'{gap:.5f}' if case=='H05' and mi==1 else f'{gap:.2f}')
        ax.annotate(text,(gap,y),xytext=(5,0),textcoords='offset points',ha='left',va='center',fontsize=7.7)
    ax.set_xlim(-.18,7.3); ax.set_ylim(-.45,3.45)
    ax.set_xticks([0,2,4,6]); ax.set_yticks(range(4))
    ax.set_yticklabels(list(reversed(names)) if ci%3==0 else ['']*4)
    ax.tick_params(axis='y',length=0,pad=6);ax.tick_params(axis='x',length=3)
    ax.spines['left'].set_visible(False)
    ax.set_title(f'{chr(97+ci)}  {case}',loc='left',fontweight='bold',pad=7)
fig.supxlabel('AIC gap from the fitted library minimum (smaller is better)',x=.60,y=.035,fontsize=8.5)
prefix=FIG/'independent_search_case_gaps'
fig.savefig(prefix.with_suffix('.pdf'),facecolor='white')
fig.savefig(prefix.with_suffix('.svg'),facecolor='white')
fig.savefig(prefix.with_suffix('.png'),dpi=300,facecolor='white')
fig.savefig(prefix.with_name(prefix.name+'_600dpi').with_suffix('.png'),dpi=600,facecolor='white')
plt.close(fig)
