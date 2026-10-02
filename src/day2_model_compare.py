"""Staged model comparison. Screen on Batch 1 development; diagnose prior-exposed sets separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor

from day2_model import CORE, LEAN, WITH_POLICY, ROOT, load_batches, split_batch1

OUT = ROOT / 'results' / 'day2' / 'model_comparison'


def definitions():
    scaled = lambda model: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
    trees = lambda model: make_pipeline(SimpleImputer(strategy='median'), model)
    log_ridge = lambda columns: TransformedTargetRegressor(
        regressor=scaled(Ridge(alpha=1.0)), func=np.log, inverse_func=np.exp)
    return {
        'median': ('baseline', CORE, lambda: DummyRegressor(strategy='median')),
        'linear_core': ('linear', CORE, lambda: scaled(LinearRegression())),
        'ridge_core_a0.1': ('ridge', CORE, lambda: scaled(Ridge(alpha=.1))),
        'ridge_core_a1': ('ridge', CORE, lambda: scaled(Ridge(alpha=1.0))),
        'ridge_core_a10': ('ridge', CORE, lambda: scaled(Ridge(alpha=10.0))),
        'ridge_policy_a1': ('ridge', WITH_POLICY, lambda: scaled(Ridge(alpha=1.0))),
        'elastic_core_a0.01': ('elastic', CORE, lambda: scaled(ElasticNet(alpha=.01, l1_ratio=.5, max_iter=20000))),
        'elastic_core_a0.1': ('elastic', CORE, lambda: scaled(ElasticNet(alpha=.1, l1_ratio=.5, max_iter=20000))),
        'huber_core': ('robust', CORE, lambda: scaled(HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1000))),
        'log_ridge_core': ('target_transform', CORE, lambda: log_ridge(CORE)),
        'log_ridge_dq_qd': ('target_transform', LEAN, lambda: log_ridge(LEAN)),
        'log_ridge_policy': ('target_transform', WITH_POLICY, lambda: log_ridge(WITH_POLICY)),
        'svr_rbf_c1': ('kernel', CORE, lambda: TransformedTargetRegressor(
            regressor=scaled(SVR(kernel='rbf', C=1.0, epsilon=.1)), transformer=StandardScaler())),
        'svr_rbf_c10': ('kernel', CORE, lambda: TransformedTargetRegressor(
            regressor=scaled(SVR(kernel='rbf', C=10.0, epsilon=.1)), transformer=StandardScaler())),
        'knn_3': ('neighbors', CORE, lambda: scaled(KNeighborsRegressor(n_neighbors=3))),
        'knn_5': ('neighbors', CORE, lambda: scaled(KNeighborsRegressor(n_neighbors=5))),
        'tree_d2': ('tree', CORE, lambda: trees(DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, random_state=42))),
        'random_forest': ('ensemble', CORE, lambda: trees(RandomForestRegressor(
            n_estimators=250, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1))),
        'extra_trees': ('ensemble', CORE, lambda: trees(ExtraTreesRegressor(
            n_estimators=250, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1))),
        'gradient_boosting': ('ensemble', CORE, lambda: trees(GradientBoostingRegressor(
            n_estimators=100, max_depth=2, min_samples_leaf=3, learning_rate=.05, random_state=42))),
        'random_forest_policy': ('ensemble', WITH_POLICY, lambda: trees(RandomForestRegressor(
            n_estimators=250, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1))),
    }


def screen(dev):
    fold_rows = []
    predictions = []
    cv = GroupKFold(n_splits=5)
    for name, (family, features, factory) in definitions().items():
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(dev, groups=dev.policy), 1):
            fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
            model = factory()
            model.fit(fit[features], fit.cycle_life)
            pred = model.predict(valid[features])
            fold_rows.append({'model': name, 'family': family, 'fold': fold, 'n_valid': len(valid),
                              'mape_pct': 100 * mean_absolute_percentage_error(valid.cycle_life, pred),
                              'mae_cycles': mean_absolute_error(valid.cycle_life, pred)})
            for (_, row), value in zip(valid.iterrows(), pred):
                predictions.append({'model': name, 'fold': fold, 'cell_id': row.cell_id,
                                    'actual': row.cycle_life, 'predicted': value,
                                    'ape_pct': 100 * abs(value - row.cycle_life) / row.cycle_life})
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(predictions)
    summary = folds.groupby(['model', 'family'], as_index=False).agg(
        cv_mape_pct=('mape_pct', 'mean'), cv_mape_sd=('mape_pct', 'std'),
        cv_mae_cycles=('mae_cycles', 'mean')).sort_values('cv_mape_pct')
    folds.to_csv(OUT / 'screen_folds.csv', index=False)
    preds.to_csv(OUT / 'screen_predictions.csv', index=False)
    summary.to_csv(OUT / 'screen_summary.csv', index=False)
    short = summary[summary.family != 'baseline'].drop_duplicates('family').head(8)
    (OUT / 'cv_shortlist.json').write_text(json.dumps(short.model.tolist(), indent=2))
    return summary


def diagnose(dev, hold, batches):
    screen = pd.read_csv(OUT / 'screen_summary.csv')
    rows = []
    predictions = []
    for name, (family, features, factory) in definitions().items():
        dev_fit = factory()
        dev_fit.fit(dev[features], dev.cycle_life)
        all_fit = factory()
        all_fit.fit(batches['batch1'][features], batches['batch1'].cycle_life)
        for label, data, model in [('batch1_holdout', hold, dev_fit),
                                    ('batch2', batches['batch2'], all_fit),
                                    ('batch3', batches['batch3'], all_fit)]:
            y = data.cycle_life.to_numpy()
            pred = model.predict(data[features])
            rows.append({'model': name, 'family': family, 'set': label, 'n': len(data),
                         'mape_pct': 100 * mean_absolute_percentage_error(y, pred),
                         'mae_cycles': mean_absolute_error(y, pred),
                         'bias_cycles': np.mean(pred - y)})
            for (_, row), value in zip(data.iterrows(), pred):
                predictions.append({'model': name, 'set': label, 'cell_id': row.cell_id,
                                    'actual': row.cycle_life, 'predicted': value,
                                    'error': value - row.cycle_life,
                                    'ape_pct': 100 * abs(value - row.cycle_life) / row.cycle_life})
    diagnosis = pd.DataFrame(rows)
    diagnosis.to_csv(OUT / 'diagnosis_scores.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'diagnosis_predictions.csv', index=False)
    wide = diagnosis.pivot(index='model', columns='set', values='mape_pct').reset_index()
    wide = screen[['model', 'family', 'cv_mape_pct', 'cv_mape_sd']].merge(wide, on='model')
    wide = wide.sort_values('cv_mape_pct')
    wide.to_csv(OUT / 'comparison_table.csv', index=False)
    show = wide.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    yy = np.arange(len(show))
    ax.barh(yy-.18, show.cv_mape_pct, height=.35, label='Batch 1 development CV', color='#2a7f90')
    ax.barh(yy+.18, show.batch1_holdout, height=.35, label='Batch 1 hold-out', color='#e7a053')
    ax.set_yticks(yy, show.model)
    ax.set(xlabel='MAPE (%)', title='Model comparison on Batch 1 (selection is CV only)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'batch1_model_comparison.png', dpi=170)
    plt.close(fig)
    return wide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen', 'diagnose'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    batches = load_batches()
    dev, hold = split_batch1(batches['batch1'])
    if args.stage == 'screen':
        summary = screen(dev)
        print(summary.round(2).to_string(index=False))
        print('CV-only family shortlist:', json.loads((OUT / 'cv_shortlist.json').read_text()))
    else:
        if not (OUT / 'screen_summary.csv').exists():
            parser.error('Run --stage screen first')
        wide = diagnose(dev, hold, batches)
        print(wide.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
