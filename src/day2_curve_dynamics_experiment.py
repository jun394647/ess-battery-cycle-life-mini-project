"""Test whether the timing of early Qdlin curve change adds predictive information.

Screen candidates using Batch 1 only. Batch 2/3 labels are read only with
--evaluate, after a candidate name has been fixed. All scores are retrospective:
earlier experiments in this project have already inspected Batch 2 labels.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

from day1_eda import FILES
from day2_model import ROOT, load_batches, split_batch1
from day2_trajectory_experiment import model


OUT = ROOT / 'results' / 'day2' / 'curve_dynamics_experiment'
BASE = ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']
FEATURE_SETS = {
    'current_3': BASE,
    'add_early_change': BASE + ['dq_50_10_logvar'],
    'add_late_change': BASE + ['dq_100_50_logvar'],
    'add_acceleration': BASE + ['dq_rate_acceleration'],
    'early_late_instead_of_total': ['dq_50_10_logvar', 'dq_100_50_logvar', 'early_mean_QD', 'early_mean_IR'],
}


def curve_features(batch_name: str, cell_ids: set[str]) -> pd.DataFrame:
    rows = []
    with h5py.File(ROOT / 'data' / FILES[batch_name]) as raw:
        refs = raw['batch']['cycles']
        for cell_id in sorted(cell_ids):
            index = int(cell_id.rsplit('_', 1)[1])
            cycles = raw[refs[index, 0]]['Qdlin']
            if len(cycles) < 100:
                raise ValueError(f'{cell_id}: fewer than 100 curves')
            q = {n: np.asarray(raw[cycles[n-1, 0]][()]).reshape(-1) for n in (10, 50, 100)}
            if any(len(curve) != 1000 or not np.isfinite(curve).all() for curve in q.values()):
                raise ValueError(f'{cell_id}: invalid voltage grid or curve')
            d_early, d_late, d_total = q[50]-q[10], q[100]-q[50], q[100]-q[10]
            variance = lambda x: np.log10(np.var(x)) if np.var(x) > 0 else np.nan
            rows.append({
                'cell_id': cell_id,
                'dq_100_10_logvar_raw': variance(d_total),
                'dq_50_10_logvar': variance(d_early),
                'dq_100_50_logvar': variance(d_late),
                # Compare change per cycle, accounting for the 40 vs 50 cycle windows.
                'dq_rate_acceleration': variance(d_late / 50) - variance(d_early / 40),
            })
    return pd.DataFrame(rows)


def features(batch_name: str) -> pd.DataFrame:
    batch = load_batches(batch_name)[batch_name]
    curves = curve_features(batch_name, set(batch.cell_id))
    merged = batch.merge(curves, on='cell_id', how='left', validate='one_to_one')
    if merged[FEATURE_SETS['current_3'] + ['dq_100_10_logvar_raw']].isna().any().any():
        raise ValueError(f'{batch_name}: missing base or curve feature')
    if not np.allclose(merged.delta_q_logvar, merged.dq_100_10_logvar_raw, atol=1e-10):
        raise ValueError(f'{batch_name}: raw Qdlin disagrees with saved ΔQ feature')
    return merged


def mape(y, pred) -> float:
    return 100 * mean_absolute_percentage_error(y, pred)


def screen(batch1: pd.DataFrame) -> pd.DataFrame:
    dev, hold = split_batch1(batch1)
    rows = []
    for name, columns in FEATURE_SETS.items():
        fold_scores = []
        for fit_idx, val_idx in GroupKFold(n_splits=5).split(dev, groups=dev.policy):
            fit, val = dev.iloc[fit_idx], dev.iloc[val_idx]
            reg = model(0.1)
            reg.fit(fit[columns], fit.cycle_life)
            fold_scores.append(mape(val.cycle_life, reg.predict(val[columns])))
        reg = model(0.1)
        reg.fit(dev[columns], dev.cycle_life)
        rows.append({'candidate': name, 'n_features': len(columns),
                     'cv_mape_pct': np.mean(fold_scores),
                     'cv_sd_pct': np.std(fold_scores, ddof=1),
                     'holdout_mape_pct': mape(hold.cycle_life, reg.predict(hold[columns])),
                     'fold_mape_pct': repr([round(score, 4) for score in fold_scores])})
    summary = pd.DataFrame(rows).sort_values('cv_mape_pct').reset_index(drop=True)
    summary.to_csv(OUT / 'screen.csv', index=False)
    return summary


def evaluate(batch1: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == 'current_3' or candidate not in FEATURE_SETS:
        raise ValueError('Select a new candidate from the Batch 1 screen')
    columns = FEATURE_SETS[candidate]
    reg = model(0.1)
    reg.fit(batch1[columns], batch1.cycle_life)
    scores, predictions = [], []
    for batch_name in ('batch2', 'batch3'):
        sample = features(batch_name)
        pred = reg.predict(sample[columns])
        scores.append({'candidate': candidate, 'set': batch_name, 'n_cells': len(sample),
                       'mape_pct': mape(sample.cycle_life, pred)})
        predictions.extend({'candidate': candidate, 'set': batch_name,
                            'cell_id': row.cell_id, 'actual': row.cycle_life, 'predicted': value,
                            'ape_pct': 100 * abs(value - row.cycle_life) / row.cycle_life}
                           for (_, row), value in zip(sample.iterrows(), pred))
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    result = pd.DataFrame(scores)
    result.to_csv(OUT / 'evaluation.csv', index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', choices=[name for name in FEATURE_SETS if name != 'current_3'],
                        help='Fixed candidate selected from Batch 1 screen; reads Batch 2/3 labels')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    batch1 = features('batch1')
    batch1[['cell_id', 'cycle_life', 'delta_q_logvar', 'dq_50_10_logvar',
            'dq_100_50_logvar', 'dq_rate_acceleration']].to_csv(OUT / 'batch1_features.csv', index=False)
    summary = screen(batch1)
    print(summary.round(3).to_string(index=False))
    if args.evaluate:
        print('\nRetrospective external evaluation:')
        print(evaluate(batch1, args.evaluate).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
