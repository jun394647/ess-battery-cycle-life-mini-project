"""Retrospectively compare the assignment split with the paper's mixed split.

The same current model is fitted on a different 41-cell training set. This is
not a new submitted test score or an exact reproduction of the paper's model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from day2_model import ROOT, candidates, load_batches


OUT = ROOT / 'results' / 'day2' / 'paper_split_diagnostic'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batches = load_batches('batch1', 'batch2')
    combined = pd.concat([batches['batch1'], batches['batch2']], ignore_index=True)
    assert len(combined) == 84
    # Official LoadData.m: 1-based test_ind=[1:2:84,84], after its exclusions.
    test_idx = np.r_[np.arange(0, 84, 2), 83]
    train_idx = np.setdiff1d(np.arange(84), test_idx)
    train, test = combined.iloc[train_idx], combined.iloc[test_idx]
    assert len(train) == 41 and len(test) == 43
    features, factory = candidates()['log_ridge_dq_qd_ir']
    reg = factory()
    reg.fit(train[features], train.cycle_life)
    predicted = reg.predict(test[features])
    detail = test[['batch', 'cell_id', 'cycle_life']].copy()
    detail['predicted'] = predicted
    detail['ape_pct'] = 100 * abs(detail.predicted - detail.cycle_life) / detail.cycle_life
    detail.to_csv(OUT / 'predictions.csv', index=False)
    summary = detail.groupby('batch', as_index=False).agg(
        n_test=('cell_id', 'size'), mape_pct=('ape_pct', 'mean'))
    summary.loc[len(summary)] = ['combined', len(detail),
                                 100 * mean_absolute_percentage_error(detail.cycle_life, predicted)]
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)[
        ['batch', 'cell_id', 'policy', 'split']].to_csv(OUT / 'allocation.csv', index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
