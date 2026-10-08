"""Regenerate search-count and per-access gap figures from safe public records."""
from pathlib import Path
from collections import defaultdict
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = json.loads((ROOT / 'results/holdout/search_visits.json').read_text())['runs']
    groups = [('astra', 'control'), ('astra', 'strategy'), ('fable51', 'strategy')]
    labels = ['Astra\ncontrol', 'Astra\nrevised', 'Fable 5.1\nrevised']
    counts = []
    for model, method in groups:
        rows = [r for r in runs if r['model_label'] == model and r['method'] == method]
        counts.append((sum(r['state'] == 'complete' and r['regret'] <= 1e-9 for r in rows),
                       sum(r['state'] == 'complete' and r['regret'] <= 2 for r in rows), len(rows)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (champion, near, n) in enumerate(counts):
        for offset, count, color, label in [(-.19, champion, '#285780', 'Exact library optimum'),
                                           (.19, near, '#d59640', 'Within 2 AIC units (includes optimum)')]:
            ax.bar(i + offset, 100 * count / n, width=.36, color=color, label=label if i == 0 else None)
            ax.text(i + offset, 100 * count / n + 2, f'{count}/{n}', ha='center')
    ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 112), ylabel='Successful searches (%)',
           title='Six new synthetic cases; 20 candidate evaluations per search')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    out = ROOT / 'recomputed/figures'; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'holdout_search_counts.png', dpi=300); fig.savefig(out / 'holdout_search_counts.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    colors = {'control': '#777777', 'strategy': '#285780'}
    for case, ax in zip([f'H{i:02d}' for i in range(1, 7)], axes.flat):
        for r in [r for r in runs if r['case'] == case]:
            values = r['regret_trajectory']
            color = '#d59640' if r['model_label'] == 'fable51' else colors[r['method']]
            ax.plot(range(1, len(values) + 1), values, color=color, label=r['model_label'] + ' / ' + r['method'])
        ax.axhline(2, color='black', linestyle=':', linewidth=.8)
        ax.set(title=case, xlabel='Candidate evaluations', ylabel='Best AIC gap (lower is better)', xlim=(1, 20))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'holdout_search_trajectories.png', dpi=300)
    fig.savefig(out / 'holdout_search_trajectories.pdf'); plt.close(fig)
    print('Figures saved under recomputed/figures; no API calls or GP fitting.')


if __name__ == '__main__':
    main()
