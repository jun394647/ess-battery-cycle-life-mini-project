"""Post-exposure, Batch 1-only refinement of early-signal log Ridge models.

The historical Batch 2 result was already known before this experiment. This
script never reads Batch 2/3, but its later test scores are retrospective.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day2_model import LEAN, WITH_IR, ROOT, load_batches, split_batch1

OUT = ROOT / 'results' / 'day2'


def log_ridge(alpha):
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy='median'),
                                StandardScaler(), Ridge(alpha=alpha)),
        func=np.log, inverse_func=np.exp)


TRIALS = {
    'dq_qd_a1': (LEAN, 1.0),
    'dq_qd_a0.1': (LEAN, .1),
    'dq_qd_ir_a1': (WITH_IR, 1.0),
    'dq_qd_ir_a0.1': (WITH_IR, .1),
    'dq_qd_meanabs_a0.1': (LEAN + ['delta_q_mean_abs'], .1),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch1 = load_batches('batch1')['batch1']
    dev, _ = split_batch1(batch1)
    rows = []
    for name, (features, alpha) in TRIALS.items():
        for fold, (fit_idx, valid_idx) in enumerate(
                GroupKFold(n_splits=5).split(dev, groups=dev.policy), 1):
            fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
            model = log_ridge(alpha)
            model.fit(fit[features], fit.cycle_life)
            pred = model.predict(valid[features])
            rows.append({'trial': name, 'fold': fold, 'features': '+'.join(features),
                         'alpha': alpha, 'n_valid': len(valid),
                         'mape_pct': 100 * mean_absolute_percentage_error(valid.cycle_life, pred)})
    folds = pd.DataFrame(rows)
    summary = folds.groupby(['trial', 'features', 'alpha'], as_index=False).agg(
        cv_mape_pct=('mape_pct', 'mean'), cv_mape_sd=('mape_pct', 'std'))
    summary = summary.sort_values('cv_mape_pct').reset_index(drop=True)
    folds.to_csv(OUT / 'refinement_folds.csv', index=False)
    summary.to_csv(OUT / 'refinement_summary.csv', index=False)
    dev[['delta_q_logvar', 'delta_q_mean_abs', 'early_mean_QD', 'early_mean_IR']].corr(
        method='pearson').to_csv(OUT / 'refinement_correlations.csv')

    robustness = []
    for n_splits in (4, 5, 6):
        for name in ('dq_qd_a1', 'dq_qd_ir_a0.1'):
            features, alpha = TRIALS[name]
            errors = []
            for fit_idx, valid_idx in GroupKFold(n_splits=n_splits).split(dev, groups=dev.policy):
                fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
                model = log_ridge(alpha)
                model.fit(fit[features], fit.cycle_life)
                errors.append(100 * mean_absolute_percentage_error(
                    valid.cycle_life, model.predict(valid[features])))
            robustness.append({'n_splits': n_splits, 'trial': name,
                               'mean_mape_pct': float(np.mean(errors)),
                               'sd_mape_pct': float(np.std(errors, ddof=1))})
    pd.DataFrame(robustness).to_csv(OUT / 'refinement_robustness.csv', index=False)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    order = list(TRIALS)
    view = summary.set_index('trial').loc[order]
    bars = ax.barh(order, view.cv_mape_pct, xerr=view.cv_mape_sd,
                   color=['#91a5b1', '#91a5b1', '#8db8a6', '#2a7f90', '#b8a49b'])
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in view.cv_mape_pct], padding=3)
    ax.invert_yaxis()
    ax.set(xlabel='Batch 1 development GroupKFold MAPE (%)',
           title='Early-signal refinement: mean ± fold SD', xlim=(0, 11))
    ax.grid(axis='x', alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / 'refinement.png', dpi=180)
    plt.close(fig)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
