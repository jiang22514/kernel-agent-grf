"""Plot complete observation-event probabilities; no fitting, sampling or API use.

Style-only reuse of common.py: Arial/DejaVu Sans, 183 mm, PDF/SVG and 600 dpi PNG.
An incomplete input fails before any output directory or figure is created.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from common import HERE, ROOT, Sources, csv_write, need, output_dir, plot_setup, read, save_figure


FILES = ('summary.json', 'pointwise_predictions.csv', 'metrics_by_borehole.csv',
         'pooled_metrics.csv', 'paired_borehole_differences.csv', 'calibration_bins.csv')
COLORS = ('#686868', '#0072B2', '#D55E00')
MARKERS = ('s', '^', 'o')
LABELS = ('SE + quadratic mean', 'RQ + constant mean', 'Selected kernel')


def csv_read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def close(a, b, message):
    need(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= 2e-11, message)


def load_data(folder, sources):
    folder = Path(folder)
    for name in FILES:
        sources.add(folder / name)
    summary = read(folder / 'summary.json')
    need(summary['status'] == 'complete', 'Need complete merged predictions')
    need(summary['n_points'] == 744 and summary['n_boreholes'] == 6,
         'Expected all 744 observations and six held-out boreholes')
    need(summary['threshold_MPa'] == 4.14, 'Observation-event threshold changed')
    names = [r['model'] for r in summary['pooled']]
    need(len(names) == 3 and names[:2] == ['SE', 'RQ'] and len(set(names)) == 3,
         'Expected SE, RQ and one distinct display name for the selected structure')
    points = csv_read(folder / 'pointwise_predictions.csv')
    folds = csv_read(folder / 'metrics_by_borehole.csv')
    pooled = csv_read(folder / 'pooled_metrics.csv')
    pairs = csv_read(folder / 'paired_borehole_differences.csv')
    bins = csv_read(folder / 'calibration_bins.csv')
    need(len(points) == 3*744 and len(folds) == 3*6 and len(pairs) == 6
         and len(bins) == 3*5 and len(pooled) == 3, 'Incomplete input tables')
    for rows in (points, folds, pooled, bins):
        need(set(r['model'] for r in rows) == set(names), 'Model labels disagree across tables')
    counts = {}
    base_identity = None
    for model in names:
        rows = sorted((r for r in points if r['model'] == model), key=lambda r: int(r['original_index']))
        need([int(r['original_index']) for r in rows] == list(range(1, 745)),
             'Missing or repeated observation index')
        identity = [(int(r['original_index']), int(r['fold']), float(r['observed_MPa']),
                     int(r['event_qc_below_4_14'])) for r in rows]
        if base_identity is None:
            base_identity = identity
        need(identity == base_identity, 'Models must predict exactly the same held-out observations')
        probability = np.array([float(r['p_qc_below_4_14']) for r in rows])
        events = np.array([int(r['event_qc_below_4_14']) for r in rows])
        values = np.array([float(r['observed_MPa']) for r in rows])
        need(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
             'Invalid predicted event probability')
        need(np.array_equal(events, (values < 4.14).astype(int)), 'Observed event mismatch')
        need(int(events.sum()) == summary['events'], 'Event total mismatch')
        brier = (probability-events)**2
        need(np.allclose(brier, [float(r['brier']) for r in rows], atol=2e-11, rtol=0),
             'Saved pointwise Brier scores disagree with probabilities')
        for fold in range(1, 7):
            ids = np.array([int(r['fold']) == fold for r in rows])
            fr = [r for r in folds if r['model'] == model and int(r['fold']) == fold]
            need(len(fr) == 1 and int(fr[0]['n']) == int(ids.sum()), 'Fold count mismatch')
            close(float(fr[0]['brier']), float(brier[ids].mean()), 'Fold Brier mismatch')
            counts[fold] = {'n': int(ids.sum()), 'events': int(events[ids].sum())}
        pr = [r for r in pooled if r['model'] == model]
        need(len(pr) == 1 and int(pr[0]['n']) == 744, 'Pooled count mismatch')
        close(float(pr[0]['brier']), float(brier.mean()), 'Pooled Brier mismatch')
        br = sorted((r for r in bins if r['model'] == model), key=lambda r: float(r['bin_low']))
        for index, row in enumerate(br):
            lo, hi = index/5, (index+1)/5
            close(float(row['bin_low']), lo, 'Fixed-bin lower edge changed')
            close(float(row['bin_high']), hi, 'Fixed-bin upper edge changed')
            ids = (probability >= lo) & ((probability < hi) | ((index == 4) & (probability == 1)))
            need(int(row['n']) == int(ids.sum()), 'Probability-bin count mismatch')
            if ids.any():
                close(float(row['mean_predicted_probability']), float(probability[ids].mean()), 'Bin mean mismatch')
                close(float(row['observed_event_fraction']), float(events[ids].mean()), 'Bin frequency mismatch')
            else:
                need(row['mean_predicted_probability'] == row['observed_event_fraction'] == '',
                     'Empty bins must have blank means, not fabricated zeros')
    need(sorted(int(r['fold']) for r in pairs) == list(range(1, 7)), 'Missing/repeated paired fold')
    for pair in pairs:
        fold = int(pair['fold'])
        fr = {r['model']: float(r['brier']) for r in folds if int(r['fold']) == fold}
        for reference in ('SE', 'RQ'):
            close(float(pair[f'champion_minus_{reference}_brier']), fr[names[2]]-fr[reference],
                  'Paired Brier difference mismatch')
    return dict(summary=summary, names=names, points=points, folds=folds, pooled=pooled,
                pairs=sorted(pairs, key=lambda r: int(r['fold'])), bins=bins, counts=counts)


def render(data, out):
    plt = plot_setup()
    width_mm = 183
    height_mm = 115
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.10, 1], height_ratios=[3.5, 1.2],
                           left=.095, right=.975, top=.89, bottom=.17, wspace=.50, hspace=.48)
    calibration = fig.add_subplot(grid[0, 0])
    counts_ax = fig.add_subplot(grid[1, 0])
    paired_ax = fig.add_subplot(grid[:, 1])
    calibration.plot([0, 1], [0, 1], ':', color='.55', lw=.85, zorder=0)
    ordered = []
    for model, color, marker, label in zip(data['names'], COLORS, MARKERS, LABELS):
        rows = sorted((r for r in data['bins'] if r['model'] == model), key=lambda r: float(r['bin_low']))
        ordered.append(rows)
        # Empty bins are missing values; they do not appear as probability zero.
        x = [float(r['mean_predicted_probability']) if int(r['n']) else np.nan for r in rows]
        y = [float(r['observed_event_fraction']) if int(r['n']) else np.nan for r in rows]
        calibration.plot(x, y, marker=marker, color=color, ms=4, lw=1, label=label)
    calibration.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02),
                    xlabel='Mean predicted probability', ylabel='Observed event frequency',
                    xticks=np.arange(0, 1.01, .2), yticks=np.arange(0, 1.01, .2))
    calibration.set_title(r'Observation event: $q_c < 4.14$ MPa', pad=6)
    calibration.text(-.23, 1.08, 'a', transform=calibration.transAxes, fontweight='bold', fontsize=9)
    counts_ax.axis('off')
    table = counts_ax.table(cellText=[[r['n'] for r in rows] for rows in ordered],
                            colLabels=['0–.2', '.2–.4', '.4–.6', '.6–.8', '.8–1'],
                            loc='center', cellLoc='center', bbox=[0, 0, 1, .93])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row:
            cell.set_text_props(color=COLORS[row-1])
            if row == 3:
                cell.set_text_props(weight='bold')
    for row, (color, marker) in enumerate(zip(COLORS, MARKERS)):
        counts_ax.plot(-.035, .93*(1-(row+1.5)/4), marker=marker, color=color,
                       markersize=3.5, transform=counts_ax.transAxes, clip_on=False)
    counts_ax.set_title('Observations per probability bin', loc='left', fontsize=7.5, pad=3)
    counts_ax.text(-.23, 1.12, 'b', transform=counts_ax.transAxes, fontweight='bold', fontsize=9)
    y = np.arange(6)
    paired_ax.axvline(0, color='.55', ls=':', lw=.9)
    for shift, reference, color, marker in [(-.10, 'SE', COLORS[0], MARKERS[0]),
                                           (.10, 'RQ', COLORS[1], MARKERS[1])]:
        x = np.array([float(r[f'champion_minus_{reference}_brier']) for r in data['pairs']])
        for yy, xx in zip(y+shift, x):
            paired_ax.plot([0, xx], [yy, yy], color=color, lw=.8, alpha=.75)
        paired_ax.scatter(x, y+shift, c=color, marker=marker, s=25, label=f'Selected − {reference}', zorder=3)
    labels = [f"BH {f}\n{data['counts'][f]['events']}/{data['counts'][f]['n']} events" for f in range(1, 7)]
    paired_ax.set_yticks(y, labels)
    paired_ax.invert_yaxis()
    paired_ax.set_ylim(5.5, -.6)
    paired_ax.set_xlabel('Brier difference\n(selected − reference)')
    paired_ax.set_title('Held-out borehole comparisons', pad=6)
    paired_ax.tick_params(axis='y', labelsize=6.5)
    limit = max(abs(float(r[f'champion_minus_{ref}_brier'])) for r in data['pairs'] for ref in ('SE', 'RQ'))
    limit = max(limit*1.20, .005)
    paired_ax.set_xlim(-limit, limit)
    paired_ax.text(.0, -.20, '← Selected better', transform=paired_ax.transAxes, fontsize=6.5, ha='left')
    paired_ax.text(1, -.20, 'Reference better →', transform=paired_ax.transAxes, fontsize=6.5, ha='right')
    paired_ax.text(-.31, 1.06, 'c', transform=paired_ax.transAxes, fontweight='bold', fontsize=9)
    handles, labels = calibration.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.52, .995), columnspacing=1.4)
    save_figure(fig, out, 'heldout_probability_calibration_brier')
    plt.close(fig)


def caption(data):
    s = data['summary']
    return (
        f"Held-out prediction of the observation event qc < 4.14 MPa for three fixed covariance structures "
        f"({s['n_points']} measurements, {s['n_boreholes']} boreholes, {s['events']} observed events). "
        f"The selected structure is {data['names'][2]}; SE uses a quadratic mean and RQ a constant mean. "
        "(a) Five fixed probability bins compare the mean Gaussian observation-event probability, including "
        "the fitted observation noise, with the empirical event frequency. The dotted diagonal denotes "
        "agreement; empty bins are omitted, and lines only guide the eye. (b) Counts in each bin follow "
        "the same model order, colors and markers as panel a; the upper edge is excluded except for the "
        "last bin. (c) Within-borehole differences in mean Brier score, selected model minus each reference; "
        "negative values favor the selected model. Labels give observed events/measurements for each "
        "borehole, ordered by horizontal coordinate. Parameters are refitted using the other five boreholes. "
        "Model structures and the event threshold were selected using the full site dataset, so this is "
        "a descriptive diagnostic of fixed structures, not end-to-end agent validation or a test of latent "
        "inter-borehole weak-zone area accuracy. All measurements are included. No confidence intervals "
        "or significance tests treat correlated depths as independent replicates."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT/'output/revision/review_fixes_2026-10-06/heldout/v2_probability_comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    sources = Sources()
    sources.add(__file__)
    sources.add(HERE/'common.py')
    data = load_data(args.input, sources)
    out = output_dir(args.output)
    for name, rows in [('calibration_source.csv', data['bins']), ('borehole_brier_source.csv', data['folds']),
                       ('paired_brier_source.csv', data['pairs']), ('pooled_source.csv', data['pooled'])]:
        csv_write(out/name, rows)
    csv_write(out/'borehole_event_counts.csv', [dict(fold=f, **r) for f, r in data['counts'].items()])
    if not args.validate_only:
        render(data, out)
    (out/'caption.txt').write_text(caption(data), encoding='utf-8')
    sources.finish(out, dict(status='VALIDATED' if args.validate_only else 'RENDERED_PENDING_VISUAL_QA',
                            figures_generated=not args.validate_only, models=data['names'],
                            n_points_per_model=744, n_boreholes=6, point_rows_included=3*744,
                            excluded_point_rows=0, confidence_intervals='none', statistical_tests='none',
                            replicate_definition='six held-out boreholes; correlated depth observations',
                            bin_edges=[0, .2, .4, .6, .8, 1],
                            shared_style='common.py; style-only inheritance',
                            champion_identity_sha256=data['summary']['champion_identity_sha256']))


if __name__ == '__main__':
    main()
