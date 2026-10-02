"""Batch 1-only screening of early trajectory features.

This is a retrospective experiment: earlier Batch 2 results were already seen.
Screening reads Batch 1 only. Evaluation of another batch requires a separately
selected candidate and is deliberately outside this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day2_model import ROOT, load_batches, split_batch1


OUT = ROOT / 'results' / 'day2' / 'trajectory_experiment'
BASE = ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']
SETS = {
    'current_3': BASE,
    'valid_qd_mean_3': ['delta_q_logvar', 'qd_valid_mean_2_100', 'early_mean_IR'],
    'qd_median_3': ['delta_q_logvar', 'qd_median_2_100', 'early_mean_IR'],
    'replace_qd2': ['delta_q_logvar', 'qd_2', 'early_mean_IR'],
    'replace_ir2': ['delta_q_logvar', 'early_mean_QD', 'ir_2'],
    'replace_ir100': ['delta_q_logvar', 'early_mean_QD', 'ir_100'],
    'replace_ir_min': ['delta_q_logvar', 'early_mean_QD', 'ir_min_2_100'],
    'replace_ir_median': ['delta_q_logvar', 'early_mean_QD', 'ir_median_2_100'],
    'add_dq_min': BASE + ['delta_q_log_abs_min'],
    'add_qd_late_slope': BASE + ['qd_slope_91_100'],
    'add_qd_early_slope': BASE + ['qd_slope_2_100'],
    'add_qd_ratio': BASE + ['qd_ratio_100_2'],
    'add_charge_change': BASE + ['charge_change_100_5'],
    'add_ir_change': BASE + ['ir_change_100_2'],
    'dq_qd_min_slope': ['delta_q_logvar', 'early_mean_QD', 'delta_q_log_abs_min', 'qd_slope_91_100'],
    'mixed_early_6': ['delta_q_logvar', 'delta_q_log_abs_min', 'qd_2', 'qd_100',
                      'qd_slope_91_100', 'charge_change_100_5'],
    'combined_6': BASE + ['delta_q_log_abs_min', 'qd_slope_91_100', 'charge_change_100_5'],
    'add_dq_mean_abs': BASE + ['delta_q_mean_abs'],
}


def slope(frame: pd.DataFrame, field: str, lo: int, hi: int) -> float:
    selected = frame.loc[frame.cycle.between(lo, hi), ['cycle', field]].dropna()
    selected = selected.loc[selected[field] > 0]
    if len(selected) < max(3, (hi - lo + 1) // 2):
        return np.nan
    return float(np.polyfit(selected.cycle, selected[field], 1)[0])


def at(frame: pd.DataFrame, field: str, cycle: int) -> float:
    selected = frame.loc[frame.cycle == cycle, field]
    if len(selected) != 1 or not np.isfinite(selected.iloc[0]) or selected.iloc[0] <= 0:
        return np.nan
    return float(selected.iloc[0])


def trajectory(batch_name: str) -> pd.DataFrame:
    path = ROOT / 'results' / f'{batch_name}_cycles.csv.gz'
    cycle = pd.read_csv(path, usecols=['cell_id', 'cycle', 'QD', 'IR', 'chargetime'])
    rows = []
    for cell_id, frame in cycle.groupby('cell_id', sort=False):
        early_qd = frame.loc[frame.cycle.between(2, 100), 'QD']
        valid_qd = early_qd[early_qd.between(0.5, 1.3)]
        qd2, qd100 = at(frame, 'QD', 2), at(frame, 'QD', 100)
        charge5, charge100 = at(frame, 'chargetime', 5), at(frame, 'chargetime', 100)
        ir2, ir100 = at(frame, 'IR', 2), at(frame, 'IR', 100)
        early_ir = frame.loc[frame.cycle.between(2, 100), 'IR']
        valid_ir = early_ir[early_ir > 0]
        rows.append({'cell_id': cell_id, 'qd_2': qd2, 'qd_100': qd100,
                     'qd_valid_mean_2_100': valid_qd.mean(),
                     'qd_median_2_100': valid_qd.median(),
                     'ir_2': ir2, 'ir_100': ir100,
                     'ir_min_2_100': valid_ir.min(),
                     'ir_median_2_100': valid_ir.median(),
                     'qd_ratio_100_2': qd100 / qd2 if qd2 > 0 else np.nan,
                     'qd_slope_2_100': slope(frame, 'QD', 2, 100),
                     'qd_slope_91_100': slope(frame, 'QD', 91, 100),
                     'charge_change_100_5': charge100 - charge5,
                     'ir_change_100_2': ir100 - ir2})
    return pd.DataFrame(rows)


def features(batch_name: str) -> pd.DataFrame:
    batch = load_batches(batch_name)[batch_name]
    batch = batch.merge(trajectory(batch_name), on='cell_id', how='left', validate='one_to_one')
    batch['delta_q_log_abs_min'] = np.log10(np.abs(batch.delta_q_min))
    return batch


def model(alpha: float):
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha)),
        func=np.log, inverse_func=np.exp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-qd2', action='store_true',
                        help='Evaluate the Batch 1 CV-leading QD2 replacement retrospectively')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = features('batch1')
    dev, hold = split_batch1(b1)
    rows = []
    for name, columns in SETS.items():
        for alpha in (0.1, 1.0, 10.0):
            fold_scores = []
            for fit_idx, valid_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
                fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
                estimator = model(alpha)
                estimator.fit(fit[columns], fit.cycle_life)
                fold_scores.append(100 * mean_absolute_percentage_error(
                    valid.cycle_life, estimator.predict(valid[columns])))
            estimator = model(alpha)
            estimator.fit(dev[columns], dev.cycle_life)
            hold_mape = 100 * mean_absolute_percentage_error(
                hold.cycle_life, estimator.predict(hold[columns]))
            rows.append({'set': name, 'alpha': alpha, 'n_features': len(columns),
                         'cv_mape_pct': np.mean(fold_scores), 'cv_sd': np.std(fold_scores, ddof=1),
                         'holdout_mape_pct': hold_mape, 'fold_mape_pct': repr([round(x, 3) for x in fold_scores])})
    result = pd.DataFrame(rows).sort_values('cv_mape_pct')
    result.to_csv(OUT / 'screen.csv', index=False)
    print(result.head(20).round(3).to_string(index=False))
    print('\nMissing values in Batch 1 development:')
    print(dev[list(set().union(*SETS.values()))].isna().sum().sort_values(ascending=False).to_string())
    if args.evaluate_qd2:
        columns = SETS['replace_qd2']
        reg = model(0.1)
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
        pd.DataFrame(scores).to_csv(OUT / 'qd2_evaluation.csv', index=False)
        pd.DataFrame(predictions).to_csv(OUT / 'qd2_predictions.csv', index=False)
        print('\nRetrospective QD2 transfer scores:')
        print(pd.DataFrame(scores).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
