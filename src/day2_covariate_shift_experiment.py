"""Explore transductive, label-free covariate weighting for a new batch.

Only target input features enter the domain classifier; target cycle life is
never used in weighting. This is an optional retrospective experiment, not the
mandatory conventional Batch 1 -> Batch 2 benchmark.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from day2_model import ROOT, load_batches, split_batch1


OUT = ROOT / 'results' / 'day2' / 'covariate_shift_experiment'
FEATURES = ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']


def predict(fit: pd.DataFrame, target: pd.DataFrame,
            strength: float, cap: float, mixture: float) -> np.ndarray:
    domain_x = pd.concat([fit[FEATURES], target[FEATURES]], ignore_index=True)
    domain_scaler = StandardScaler().fit(domain_x)
    domain_y = np.r_[np.zeros(len(fit)), np.ones(len(target))]
    domain = LogisticRegression(C=strength, max_iter=2000)
    domain.fit(domain_scaler.transform(domain_x), domain_y)
    propensity = domain.predict_proba(domain_scaler.transform(fit[FEATURES]))[:, 1]
    ratio = propensity / (1 - propensity) * len(fit) / len(target)
    ratio = np.clip(ratio, 1 / cap, cap)
    weights = (1 - mixture) + mixture * ratio / np.mean(ratio)
    scaler = StandardScaler().fit(fit[FEATURES])
    reg = Ridge(alpha=0.1).fit(scaler.transform(fit[FEATURES]),
                               np.log(fit.cycle_life), sample_weight=weights)
    return np.exp(reg.predict(scaler.transform(target[FEATURES])))


def mape(actual, predicted) -> float:
    return 100 * mean_absolute_percentage_error(actual, predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the prespecified mild candidate on Batch 2 and 3')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = load_batches('batch1')['batch1']
    dev, hold = split_batch1(b1)
    rows = []
    for strength in (0.1, 1.0):
        for cap in (3.0, 10.0):
            for mixture in (0.5, 1.0):
                fold_scores = []
                for fit_idx, valid_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
                    fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
                    fold_scores.append(mape(valid.cycle_life,
                                            predict(fit, valid, strength, cap, mixture)))
                rows.append({'strength': strength, 'cap': cap, 'mixture': mixture,
                             'cv_mape_pct': np.mean(fold_scores),
                             'holdout_mape_pct': mape(hold.cycle_life,
                                                      predict(dev, hold, strength, cap, mixture)),
                             'fold_mape_pct': repr([round(x, 3) for x in fold_scores])})
    result = pd.DataFrame(rows).sort_values('cv_mape_pct')
    result.to_csv(OUT / 'screen.csv', index=False)
    print(result.round(3).to_string(index=False))
    if args.evaluate:
        scores = []
        # A conservative setting and the Batch 1 CV leader. Both are retrospective.
        for variant, params in {
                'mild': {'strength': 0.1, 'cap': 3.0, 'mixture': 0.5},
                'cv_leader': {'strength': 0.1, 'cap': 10.0, 'mixture': 1.0},
        }.items():
            for name in ('batch2', 'batch3'):
                test = load_batches(name)[name]
                pred = predict(b1, test, **params)
                scores.append({'variant': variant, 'set': name, 'n_cells': len(test),
                               'mape_pct': mape(test.cycle_life, pred),
                               'mean_bias_cycles': float(np.mean(pred - test.cycle_life))})
        evaluation = pd.DataFrame(scores)
        evaluation.to_csv(OUT / 'evaluation.csv', index=False)
        print('\nTransductive external evaluation:')
        print(evaluation.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
