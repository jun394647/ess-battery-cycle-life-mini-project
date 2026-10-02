"""Reproducible Batch 1 CV ablation for early ΔQ(V), QD, and Tavg."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

from day2_model import ROOT, candidates, load_batches, split_batch1


OUT = ROOT / 'results' / 'day2'
FEATURE_SETS = {
    'dq_only': ['delta_q_logvar'],
    'dq_qd': ['delta_q_logvar', 'early_mean_QD'],
    'dq_tavg': ['delta_q_logvar', 'early_mean_Tavg'],
    'dq_qd_tavg': ['delta_q_logvar', 'early_mean_QD', 'early_mean_Tavg'],
}


def main():
    batch1 = load_batches()['batch1']
    development, _ = split_batch1(batch1)
    factory = candidates()['log_ridge_dq_qd'][1]
    folds = []
    cv = GroupKFold(n_splits=5)
    for name, columns in FEATURE_SETS.items():
        for number, (fit_idx, valid_idx) in enumerate(cv.split(development, groups=development.policy), 1):
            fit, valid = development.iloc[fit_idx], development.iloc[valid_idx]
            model = factory()
            model.fit(fit[columns], fit.cycle_life)
            pred = model.predict(valid[columns])
            folds.append({'feature_set': name, 'fold': number, 'n_valid': len(valid),
                          'mape_pct': 100 * mean_absolute_percentage_error(valid.cycle_life, pred)})
    detail = pd.DataFrame(folds)
    summary = detail.groupby('feature_set', as_index=False).agg(
        cv_mape_pct=('mape_pct', 'mean'), cv_mape_sd=('mape_pct', 'std'))
    detail.to_csv(OUT / 'feature_ablation_folds.csv', index=False)
    summary.to_csv(OUT / 'feature_ablation.csv', index=False)

    order = list(FEATURE_SETS)
    labels = ['ΔQ', 'ΔQ + QD', 'ΔQ + Tavg', 'ΔQ + QD + Tavg']
    means = [summary.set_index('feature_set').loc[name, 'cv_mape_pct'] for name in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, means, color=['#a7becb', '#2a7f90', '#a7becb', '#a7becb'])
    ax.bar_label(bars, fmt='%.2f%%', padding=3)
    ax.set(ylabel='Batch 1 development CV MAPE (%)',
           title='Same grouped folds and log Ridge; only the feature set changes', ylim=(0, max(means) * 1.22))
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / 'feature_ablation.png', dpi=180)
    plt.close(fig)
    print(summary.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
