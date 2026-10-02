"""Stress-test Batch 1-only extrapolation toward short-lived cells.

The Batch 2 target distribution was seen before this retrospective experiment.
Selection here uses only Batch 1 grouped CV and low-life stress tests; Batch 2
evaluation is a separate optional command.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from day2_model import ROOT, load_batches, split_batch1


OUT = ROOT / 'results' / 'day2' / 'shortlife_experiment'
FEATURES = ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']
POWERS = [0.0, 0.5, 1.0, 2.0, 3.0]


def predict(fit: pd.DataFrame, valid: pd.DataFrame, power: float) -> np.ndarray:
    scaler = StandardScaler().fit(fit[FEATURES])
    y = fit.cycle_life.to_numpy()
    weights = (np.median(y) / y) ** power
    weights /= np.mean(weights)
    reg = Ridge(alpha=0.1).fit(scaler.transform(fit[FEATURES]), np.log(y),
                               sample_weight=weights)
    return np.exp(reg.predict(scaler.transform(valid[FEATURES])))


def mape(actual, predicted):
    return 100 * mean_absolute_percentage_error(actual, predicted)


def screen(b1: pd.DataFrame) -> pd.DataFrame:
    dev, hold = split_batch1(b1)
    rows = []
    for power in POWERS:
        folds = []
        for train_idx, valid_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
            fit, valid = dev.iloc[train_idx], dev.iloc[valid_idx]
            folds.append(mape(valid.cycle_life, predict(fit, valid, power)))
        stress = []
        for threshold in (700, 750, 800):
            fit = dev[dev.cycle_life >= threshold]
            valid = dev[dev.cycle_life < threshold]
            if len(fit) >= 15 and len(valid) >= 5:
                stress.append(mape(valid.cycle_life, predict(fit, valid, power)))
        rows.append({'power': power, 'cv_mape_pct': np.mean(folds),
                     'holdout_mape_pct': mape(hold.cycle_life, predict(dev, hold, power)),
                     'low_life_stress_mape_pct': np.mean(stress),
                     'folds': repr([round(x, 3) for x in folds]),
                     'stress': repr([round(x, 3) for x in stress])})
    return pd.DataFrame(rows).sort_values('power')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-power', type=float, choices=POWERS,
                        help='Evaluate a previously selected weight power on external batches')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = load_batches('batch1')['batch1']
    result = screen(b1)
    result.to_csv(OUT / 'screen.csv', index=False)
    print(result.round(3).to_string(index=False))
    if args.evaluate_power is not None:
        rows = []
        for batch_name in ('batch2', 'batch3'):
            test = load_batches(batch_name)[batch_name]
            pred = predict(b1, test, args.evaluate_power)
            rows.append({'power': args.evaluate_power, 'set': batch_name,
                         'n_cells': len(test), 'mape_pct': mape(test.cycle_life, pred),
                         'mean_bias_cycles': float(np.mean(pred - test.cycle_life))})
            pd.DataFrame({'cell_id': test.cell_id, 'actual': test.cycle_life,
                          'predicted': pred}).to_csv(OUT / f'{batch_name}_predictions.csv', index=False)
        evaluation = pd.DataFrame(rows)
        evaluation.to_csv(OUT / 'evaluation.csv', index=False)
        print('\nRetrospective transfer scores:')
        print(evaluation.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
