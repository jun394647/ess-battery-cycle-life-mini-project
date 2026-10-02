"""Summarize three retrospective methods without changing the submitted model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from day2_model import ROOT

OUT = ROOT / 'results' / 'day2' / 'alternative_methods'
BASE = ROOT / 'results' / 'day2'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(BASE / 'performance.csv').set_index('set')
    inverse = pd.read_csv(BASE / 'inverse_life_experiment/screen.csv')
    inverse_eval = pd.read_csv(BASE / 'inverse_life_experiment/evaluation.csv').set_index('set')
    pca = pd.read_csv(BASE / 'curve_pca_experiment/screen.csv')
    pca_eval = pd.read_csv(BASE / 'curve_pca_experiment/evaluation.csv').set_index(['candidate', 'set'])
    early = pd.read_csv(BASE / 'curve_dynamics_experiment/screen.csv')
    rows = [{
        'method': 'current_log_ridge',
        'batch1_cv_mape_pct': perf.loc['Train (Batch 1 CV)', 'mape_pct'],
        'batch1_holdout_mape_pct': perf.loc['Valid (Batch 1 hold-out)', 'mape_pct'],
        'batch2_mape_pct': perf.loc['Test (Batch 2)', 'mape_pct'],
    }]
    for n in (1, 2):
        row = pca[(pca.curve == 'shape') & (pca.components == n) & np.isclose(pca.alpha, .1)].iloc[0]
        rows.append({'method': f'shape_pca{n}_a0.1',
                     'batch1_cv_mape_pct': row.cv_mape_pct,
                     'batch1_holdout_mape_pct': row.holdout_mape_pct,
                     'batch2_mape_pct': pca_eval.loc[(f'shape_pca{n}_a0.1', 'batch2'), 'mape_pct']})
    inv = inverse[(inverse.features == 'dq_qd') & np.isclose(inverse.alpha, 1)].iloc[0]
    rows.append({'method': 'inverse_life_dq_qd_a1',
                 'batch1_cv_mape_pct': inv.cv_mape_pct,
                 'batch1_holdout_mape_pct': inv.holdout_mape_pct,
                 'batch2_mape_pct': inverse_eval.loc['batch2', 'mape_pct']})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'comparison.csv', index=False)
    assert early.loc[early.candidate != 'current_3', 'cv_mape_pct'].min() > early.loc[early.candidate == 'current_3', 'cv_mape_pct'].iloc[0]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(table))
    width = .25
    for shift, column, label, color in [
        (-1, 'batch1_cv_mape_pct', 'Batch 1 CV', '#497c93'),
        (0, 'batch1_holdout_mape_pct', 'Batch 1 hold-out', '#e5a34d'),
        (1, 'batch2_mape_pct', 'Batch 2', '#bc5e53'),
    ]:
        bars = ax.bar(x + shift * width, table[column], width, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
    ax.set_xticks(x, ['Current\nlog Ridge', 'Curve shape\nPCA 1 axis',
                      'Curve shape\nPCA 2 axes', 'Inverse life\nRidge'])
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Internal validation and batch transfer differ')
    ax.set_ylim(0, max(table.batch2_mape_pct.max(), table.batch1_holdout_mape_pct.max()) * 1.18)
    ax.legend(ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=180)
    plt.close(fig)
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
