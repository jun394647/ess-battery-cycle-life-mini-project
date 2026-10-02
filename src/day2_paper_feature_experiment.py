"""Screen the paper's discharge-feature family using Batch 1 only.

This is a retrospective feature experiment. It does not change the mandatory
Batch 1 -> Batch 2 submitted model or read Batch 2 labels for selection.
"""
from __future__ import annotations

import argparse
import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day1_eda import FILES
from day2_model import ROOT, load_batches, split_batch1
from day2_trajectory_experiment import model as log_ridge


OUT = ROOT / 'results' / 'day2' / 'paper_feature_experiment'
PAPER6 = ['delta_q_logvar', 'delta_q_log_abs_min', 'delta_q_log_abs_skew',
          'delta_q_log_abs_kurtosis', 'qd_2', 'qd_max_minus_2']
ROBUST_PAPER6 = PAPER6[:-1] + ['qd_valid_max_minus_2']


def raw_curves(batch_name: str) -> pd.DataFrame:
    rows = []
    with h5py.File(ROOT / 'data' / FILES[batch_name]) as f:
        batch = f['batch']
        for idx in range(batch['cycles'].shape[0]):
            cycles = f[batch['cycles'][idx, 0]]
            q10 = np.asarray(f[cycles['Qdlin'][9, 0]][()]).reshape(-1)
            q100 = np.asarray(f[cycles['Qdlin'][99, 0]][()]).reshape(-1)
            dq = q100 - q10
            rows.append({'cell_id': f'{batch_name}_{idx:03d}',
                         'delta_q_log_abs_skew': np.log10(abs(skew(dq))),
                         'delta_q_log_abs_kurtosis': np.log10(abs(kurtosis(dq, fisher=False)))})
    return pd.DataFrame(rows)


def features(batch_name: str) -> pd.DataFrame:
    batch = load_batches(batch_name)[batch_name]
    summary = pd.read_csv(ROOT / 'results' / f'{batch_name}_cycles.csv.gz',
                          usecols=['cell_id', 'cycle', 'QD'])
    rows = []
    for cell_id, frame in summary.groupby('cell_id', sort=False):
        early = frame.loc[frame.cycle.between(2, 100), 'QD']
        qd2 = frame.loc[frame.cycle == 2, 'QD'].iloc[0]
        rows.append({'cell_id': cell_id, 'qd_2': qd2,
                     'qd_max_minus_2': early.max() - qd2,
                     'qd_valid_max_minus_2': early[early.between(0.5, 1.3)].max() - qd2})
    batch = batch.merge(pd.DataFrame(rows), on='cell_id', validate='one_to_one')
    batch = batch.merge(raw_curves(batch_name), on='cell_id', validate='one_to_one')
    batch['delta_q_log_abs_min'] = np.log10(abs(batch.delta_q_min))
    return batch


def log_elastic(alpha: float):
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000)),
        func=np.log, inverse_func=np.exp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true',
                        help='Retrospectively evaluate the development-CV-leading robust paper candidate')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = features('batch1')
    b1[['cell_id'] + PAPER6 + ['qd_valid_max_minus_2']].to_csv(
        OUT / 'batch1_features.csv', index=False)
    dev, hold = split_batch1(b1)
    sets = {'paper_discharge6': PAPER6,
            'paper_discharge6_valid_capacity': ROBUST_PAPER6,
            'paper_dq4': PAPER6[:4],
            'paper_dq2': PAPER6[:2],
            'current3': ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']}
    models = {'log_ridge_a0.1': lambda: log_ridge(0.1),
              'log_ridge_a1': lambda: log_ridge(1.0),
              'log_elastic_a0.01': lambda: log_elastic(0.01),
              'log_elastic_a0.1': lambda: log_elastic(0.1)}
    rows = []
    for set_name, columns in sets.items():
        for model_name, factory in models.items():
            fold_scores = []
            for train_idx, val_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
                train, val = dev.iloc[train_idx], dev.iloc[val_idx]
                reg = factory()
                reg.fit(train[columns], train.cycle_life)
                fold_scores.append(100 * mean_absolute_percentage_error(
                    val.cycle_life, reg.predict(val[columns])))
            reg = factory()
            reg.fit(dev[columns], dev.cycle_life)
            rows.append({'features': set_name, 'model': model_name,
                         'cv_mape_pct': np.mean(fold_scores),
                         'cv_sd_pct': np.std(fold_scores, ddof=1),
                         'holdout_mape_pct': 100 * mean_absolute_percentage_error(
                             hold.cycle_life, reg.predict(hold[columns])),
                         'fold_mape_pct': repr([round(x, 3) for x in fold_scores])})
    result = pd.DataFrame(rows).sort_values('cv_mape_pct')
    result.to_csv(OUT / 'screen.csv', index=False)
    print(result.round(3).to_string(index=False))
    if args.evaluate:
        columns = ROBUST_PAPER6
        reg = log_elastic(0.01)
        reg.fit(b1[columns], b1.cycle_life)
        scores, predictions = [], []
        for batch_name in ('batch2', 'batch3'):
            sample = features(batch_name)
            pred = reg.predict(sample[columns])
            scores.append({'set': batch_name, 'n_cells': len(sample),
                           'mape_pct': 100 * mean_absolute_percentage_error(sample.cycle_life, pred)})
            predictions.extend({'set': batch_name, 'cell_id': row.cell_id,
                                'actual': row.cycle_life, 'predicted': value}
                               for (_, row), value in zip(sample.iterrows(), pred))
        pd.DataFrame(scores).to_csv(OUT / 'evaluation.csv', index=False)
        pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
        print('\nRetrospective transfer scores:')
        print(pd.DataFrame(scores).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
