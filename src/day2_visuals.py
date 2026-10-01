"""Diagnostic figures that connect Day 2 errors to batch and model choices."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'day2' / 'figures'
COLORS = {'batch1': '#2a7f90', 'batch2': '#d56b48', 'batch3': '#8065a8'}


def feature_target_shift(cells):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    batches = ['batch1', 'batch2', 'batch3']
    rng = np.random.default_rng(42)
    for ax, column, title, ylabel in [
        (axes[0], 'delta_q_logvar', 'Early ΔQ(V) variance by batch', 'log10 var(ΔQ(V))'),
        (axes[1], 'cycle_life', 'Observed cycle life by batch', 'Cycle life (cycles)'),
    ]:
        values = [cells.loc[cells.batch == b, column].dropna().to_numpy() for b in batches]
        plot = ax.boxplot(values, positions=[1, 2, 3], widths=.52, patch_artist=True,
                          showfliers=False, medianprops={'color': '#263340', 'linewidth': 1.8})
        for box, b in zip(plot['boxes'], batches):
            box.set_facecolor(COLORS[b])
            box.set_alpha(.34)
        for pos, v, b in zip([1, 2, 3], values, batches):
            ax.scatter(pos + rng.normal(0, .07, len(v)), v, s=12, alpha=.55,
                       color=COLORS[b], edgecolors='none')
        ax.set(xticks=[1, 2, 3], xticklabels=['Batch 1', 'Batch 2', 'Batch 3'],
               ylabel=ylabel, title=title)
        ax.grid(axis='y', alpha=.18)
        for pos, v in enumerate(values, 1):
            median_text = f'{np.median(v):.2f}' if column == 'delta_q_logvar' else str(int(np.floor(np.median(v) + .5)))
            ax.text(pos, max(v) + (max(v)-min(v))*.05, f'med {median_text}',
                    ha='center', va='bottom', fontsize=8)
        ax.margins(y=.16)
    fig.tight_layout()
    fig.savefig(OUT / 'feature_target_shift.png', dpi=180)
    plt.close(fig)


def batch2_dq_calibration(cells, predictions):
    b2 = predictions[predictions['set'] == 'test_batch2'].copy()
    b2 = b2.merge(cells[['cell_id', 'delta_q_logvar']], on='cell_id', validate='one_to_one')
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for _, row in b2.iterrows():
        ax.plot([row.delta_q_logvar]*2, [row.actual, row.predicted],
                color='#aab4bc', alpha=.55, linewidth=.8)
    ax.scatter(b2.delta_q_logvar, b2.actual, s=27, color='#34495a',
               alpha=.78, label='Observed')
    ax.scatter(b2.delta_q_logvar, b2.predicted, s=27, color='#d56b48',
               alpha=.78, label='Predicted')
    ax.set(xlabel='Early log10 var(ΔQ(V))', ylabel='Cycle life (cycles)',
           title='Batch 2: early signal and life calibration')
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'batch2_dq_calibration.png', dpi=180)
    plt.close(fig)


def error_by_life_band(predictions):
    p = predictions.copy()
    p['band'] = np.select([p.actual < 500, p.actual > 1000], ['<500', '>1000'], default='500–1000')
    sets = [('valid_batch1', 'Batch 1 hold-out', COLORS['batch1']),
            ('test_batch2', 'Batch 2', COLORS['batch2']),
            ('additional_batch3', 'Batch 3', COLORS['batch3'])]
    bands = ['<500', '500–1000', '>1000']
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = .24
    for offset, (name, label, color) in enumerate(sets):
        labeled = False
        for band_i, band in enumerate(bands):
            g = p[(p['set'] == name) & (p.band == band)]
            if g.empty:
                continue
            x = band_i + (offset-1)*width
            value = g.error.mean()
            ax.bar(x, value, width=width, color=color, alpha=.85,
                   label=label if not labeled else None)
            labeled = True
            ax.text(x, value + (12 if value >= 0 else -12), f'n={len(g)}',
                    ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='#333333', linewidth=.9)
    ax.set(xticks=range(3), xticklabels=bands, xlabel='Observed cycle life band',
           ylabel='Mean prediction − actual (cycles)', title='Direction of error by life band')
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.grid(axis='y', alpha=.18)
    ax.set_ylim(-440, 330)
    fig.tight_layout()
    fig.savefig(OUT / 'error_by_life_band.png', dpi=180)
    plt.close(fig)


def candidate_transfer(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)
    linear = {'linear', 'ridge', 'elastic', 'robust', 'target_transform'}
    comparison = comparison.copy()
    comparison['group'] = np.where(comparison.family.isin(linear), 'Linear / transformed',
                                   'Tree / neighbor / kernel')
    palette = {'Linear / transformed': '#2a7f90', 'Tree / neighbor / kernel': '#d56b48'}
    labels = {'log_ridge_core': 'Chosen log Ridge', 'log_ridge_policy': 'Log Ridge + policy',
              'tree_d2': 'Shallow tree', 'random_forest': 'Random Forest'}
    for ax, column, title in [(axes[0], 'batch2', 'Batch 2 MAPE'),
                              (axes[1], 'batch3', 'Batch 3 MAPE')]:
        for group, data in comparison.groupby('group'):
            ax.scatter(data.cv_mape_pct, data[column], s=38, alpha=.78,
                       color=palette[group], label=group)
        for name, label in labels.items():
            row = comparison.set_index('model').loc[name]
            offset = (5, -15) if name == 'log_ridge_policy' and column == 'batch3' else (5, 5)
            ax.annotate(label, (row.cv_mape_pct, row[column]), xytext=offset,
                        textcoords='offset points', fontsize=7)
        rho = comparison.cv_mape_pct.corr(comparison[column], method='spearman')
        ax.text(.03, .96, f'Spearman ρ = {rho:.2f}', transform=ax.transAxes,
                va='top', fontsize=8)
        ax.set(xlabel='Batch 1 development CV MAPE (%)', ylabel=title, title=title)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'candidate_transfer.png', dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / 'results' / 'paper_screened_cells.csv')
    cells = cells[cells.cycle_life.notna()]
    preds = pd.read_csv(ROOT / 'results' / 'day2' / 'cell_predictions.csv')
    comparison = pd.read_csv(ROOT / 'results' / 'day2' / 'model_comparison' / 'comparison_table.csv')
    feature_target_shift(cells)
    batch2_dq_calibration(cells, preds)
    error_by_life_band(preds)
    candidate_transfer(comparison)
    print('Saved four diagnostic figures to', OUT)


if __name__ == '__main__':
    main()
