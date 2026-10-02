"""Exploratory few-label calibration on Batch 2 with policy-disjoint evaluation."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupShuffleSplit

from day2_model import ROOT, load_batches, candidates

OUT = Path(__file__).resolve().parents[1] / 'results' / 'day2' / 'batch_calibration'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batches = load_batches()
    batch1, batch2, batch3 = (batches[k] for k in ('batch1', 'batch2', 'batch3'))
    chosen = json.loads((ROOT / 'results/day2/protocol.json').read_text())['chosen_model']
    features, factory = candidates()[chosen]
    fixed = factory()
    fixed.fit(batch1[features], batch1.cycle_life)
    base = fixed.predict(batch2[features])
    actual = batch2.cycle_life.to_numpy()
    splitter = GroupShuffleSplit(n_splits=50, train_size=1 / 3, random_state=42)
    rows = []
    for run, (cal_idx, eval_idx) in enumerate(splitter.split(batch2, groups=batch2.policy), 1):
        assert not set(batch2.iloc[cal_idx].policy) & set(batch2.iloc[eval_idx].policy)
        additive = float(np.median(actual[cal_idx] - base[cal_idx]))
        multiplicative = float(np.median(actual[cal_idx] / base[cal_idx]))
        variants = {'unadjusted': base[eval_idx],
                    'median_residual_shift': base[eval_idx] + additive,
                    'median_ratio': base[eval_idx] * multiplicative}
        for method, pred in variants.items():
            rows.append({'run': run, 'method': method,
                         'n_calibration': len(cal_idx), 'n_evaluation': len(eval_idx),
                         'n_calibration_policies': batch2.iloc[cal_idx].policy.nunique(),
                         'mape_pct': 100 * mean_absolute_percentage_error(actual[eval_idx], pred),
                         'mae_cycles': mean_absolute_error(actual[eval_idx], pred),
                         'bias_cycles': float(np.mean(pred - actual[eval_idx])),
                         'additive_shift': additive, 'multiplicative_ratio': multiplicative})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'repeated_policy_splits.csv', index=False)
    summary = detail.groupby('method', as_index=False).agg(
        mean_mape_pct=('mape_pct', 'mean'), median_mape_pct=('mape_pct', 'median'),
        sd_mape_pct=('mape_pct', 'std'), mean_mae_cycles=('mae_cycles', 'mean'),
        mean_bias_cycles=('bias_cycles', 'mean'),
        median_n_calibration=('n_calibration', 'median'),
        median_n_evaluation=('n_evaluation', 'median')).sort_values('mean_mape_pct')
    summary.to_csv(OUT / 'summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods = ['unadjusted', 'median_residual_shift', 'median_ratio']
    data = [detail.loc[detail.method == m, 'mape_pct'].to_numpy() for m in methods]
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.set(ylabel='MAPE on unseen Batch 2 policies (%)',
           title='Exploratory adaptation: 50 policy-disjoint splits')
    ax.tick_params(axis='x', labelrotation=15)
    fig.tight_layout()
    fig.savefig(OUT / 'calibration_comparison.png', dpi=170)
    plt.close(fig)

    full_shift = float(np.median(actual - base))
    full_ratio = float(np.median(actual / base))
    b3_base = fixed.predict(batch3[features])
    print(summary.round(2).to_string(index=False))
    print('Batch 2 full-label median shift (diagnostic only):', round(full_shift, 2),
          'ratio:', round(full_ratio, 3))
    print('Batch 3 MAPE with Batch 2 full-label shift (not an evaluated model):',
          round(100 * mean_absolute_percentage_error(batch3.cycle_life, b3_base + full_shift), 2))
    print('Batch 3 MAPE with Batch 2 full-label ratio (not an evaluated model):',
          round(100 * mean_absolute_percentage_error(batch3.cycle_life, b3_base * full_ratio), 2))


if __name__ == '__main__':
    main()
