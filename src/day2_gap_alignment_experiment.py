"""Check whether removing time-gap cycles changes the early ΔQ(V) predictor.

The alternate alignment follows the documented MathWorks example. It is kept
separate from the submitted extractor until grouped Batch 1 validation shows
a benefit. Batch 2 labels are not loaded in this screening experiment.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

from day1_eda import FILES
from day2_model import ROOT, load_batches, split_batch1
from day2_trajectory_experiment import model


OUT = ROOT / 'results' / 'day2' / 'gap_alignment_experiment'


def extract(batch_name: str) -> pd.DataFrame:
    rows = []
    with h5py.File(ROOT / 'data' / FILES[batch_name]) as f:
        batch = f['batch']
        for raw_index in range(batch['cycles'].shape[0]):
            cycles = f[batch['cycles'][raw_index, 0]]
            good = [0]
            bad = []
            for j in range(1, cycles['t'].shape[0]):
                times = np.asarray(f[cycles['t'][j, 0]][()]).reshape(-1)
                dt = np.diff(times)
                is_gap = len(dt) > 0 and np.max(dt) > 5 * np.mean(dt)
                if is_gap:
                    bad.append(j + 1)
                else:
                    good.append(j)
                if len(good) >= 100:
                    break
            if len(good) < 100:
                continue
            q10 = np.asarray(f[cycles['Qdlin'][good[9], 0]][()]).reshape(-1)
            q100 = np.asarray(f[cycles['Qdlin'][good[99], 0]][()]).reshape(-1)
            rows.append({'cell_id': f'{batch_name}_{raw_index:03d}',
                         'aligned_delta_q_logvar': np.log10(np.var(q100 - q10)),
                         'source_cycle_10': good[9] + 1,
                         'source_cycle_100': good[99] + 1,
                         'bad_before_100': len(bad)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aligned = extract('batch1')
    aligned.to_csv(OUT / 'batch1_alignment.csv', index=False)
    b1 = load_batches('batch1')['batch1'].merge(aligned, on='cell_id', validate='one_to_one')
    dev, hold = split_batch1(b1)
    scores = []
    for feature in ('delta_q_logvar', 'aligned_delta_q_logvar'):
        columns = [feature, 'early_mean_QD', 'early_mean_IR']
        fold_scores = []
        for fit_idx, valid_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
            fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
            estimator = model(0.1)
            estimator.fit(fit[columns], fit.cycle_life)
            fold_scores.append(100 * mean_absolute_percentage_error(
                valid.cycle_life, estimator.predict(valid[columns])))
        estimator = model(0.1)
        estimator.fit(dev[columns], dev.cycle_life)
        scores.append({'feature': feature, 'n_changed_batch1': int(np.sum(
            ~np.isclose(b1.delta_q_logvar, b1.aligned_delta_q_logvar))),
            'cv_mape_pct': float(np.mean(fold_scores)),
            'holdout_mape_pct': 100 * mean_absolute_percentage_error(
                hold.cycle_life, estimator.predict(hold[columns])),
            'fold_mape_pct': repr([round(x, 3) for x in fold_scores])})
    result = pd.DataFrame(scores)
    result.to_csv(OUT / 'screen.csv', index=False)
    print(result.round(3).to_string(index=False))
    print('Bad-cycle counts in analyzed Batch 1:', b1.bad_before_100.value_counts().to_dict())


if __name__ == '__main__':
    main()
