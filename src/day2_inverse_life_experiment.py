"""Test inverse cycle life as an early degradation-rate proxy.

The Batch 2 evaluation is retrospective; earlier project iterations saw its labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day2_model import ROOT, load_batches, split_batch1

OUT = ROOT / 'results' / 'day2' / 'inverse_life_experiment'
FEATURES = {
    'dq_qd': ['delta_q_logvar', 'early_mean_QD'],
    'dq_qd_ir': ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR'],
}


def model(alpha):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))


def predict(reg, x):
    inverse_life = reg.predict(x)
    if np.any(inverse_life <= 0):
        raise ValueError('Predicted inverse life must be positive')
    return 1 / inverse_life


def mape(y, pred):
    return 100 * mean_absolute_percentage_error(y, pred)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch1 = load_batches('batch1')['batch1']
    dev, hold = split_batch1(batch1)
    rows = []
    for name, cols in FEATURES.items():
        for alpha in (0.1, 1.0, 10.0):
            folds = []
            for fit_idx, val_idx in GroupKFold(5).split(dev, groups=dev.policy):
                fit, val = dev.iloc[fit_idx], dev.iloc[val_idx]
                reg = model(alpha)
                reg.fit(fit[cols], 1 / fit.cycle_life)
                folds.append(mape(val.cycle_life, predict(reg, val[cols])))
            reg = model(alpha)
            reg.fit(dev[cols], 1 / dev.cycle_life)
            rows.append({'features': name, 'alpha': alpha, 'cv_mape_pct': np.mean(folds),
                         'holdout_mape_pct': mape(hold.cycle_life, predict(reg, hold[cols])),
                         'fold_mape_pct': repr([round(score, 4) for score in folds])})
    summary = pd.DataFrame(rows).sort_values('cv_mape_pct')
    summary.to_csv(OUT / 'screen.csv', index=False)
    # Evaluate the Batch 1 CV leader as a diagnostic, despite its holdout failure.
    batches = load_batches('batch2', 'batch3')
    cols = FEATURES['dq_qd']
    reg = model(1.0)
    reg.fit(batch1[cols], 1 / batch1.cycle_life)
    evaluation, predictions = [], []
    for name in ('batch2', 'batch3'):
        batch = batches[name]
        pred = predict(reg, batch[cols])
        evaluation.append({'features': 'dq_qd', 'alpha': 1.0, 'set': name,
                           'n_cells': len(batch), 'mape_pct': mape(batch.cycle_life, pred)})
        predictions.extend({'set': name, 'cell_id': row.cell_id, 'actual': row.cycle_life,
                            'predicted': value, 'ape_pct': 100 * abs(value-row.cycle_life) / row.cycle_life}
                           for (_, row), value in zip(batch.iterrows(), pred))
    pd.DataFrame(evaluation).to_csv(OUT / 'evaluation.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(evaluation).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
