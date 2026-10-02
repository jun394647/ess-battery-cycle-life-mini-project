"""Day 2: grouped model selection, then frozen hold-out and batch evaluation."""
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'day2'
DATA = ROOT / 'results' / 'paper_screened_cells.csv'
SEED = 42
CORE = ['delta_q_logvar', 'early_mean_QD', 'early_mean_Tavg']
LEAN = ['delta_q_logvar', 'early_mean_QD']
WITH_POLICY = CORE + ['policy_rate_proxy']


def candidates():
    ridge = lambda a: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))
    elastic = lambda a: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                      ElasticNet(alpha=a, l1_ratio=.5, max_iter=20000))
    log_ridge = lambda: TransformedTargetRegressor(
        regressor=ridge(1.0), func=np.log, inverse_func=np.exp)
    return {
        'median_baseline': (CORE, lambda: DummyRegressor(strategy='median')),
        'ridge_core_a0.1': (CORE, lambda: ridge(.1)),
        'ridge_core_a1': (CORE, lambda: ridge(1.0)),
        'ridge_core_a10': (CORE, lambda: ridge(10.0)),
        'ridge_policy_a1': (WITH_POLICY, lambda: ridge(1.0)),
        'log_ridge_core': (CORE, log_ridge),
        'log_ridge_dq_qd': (LEAN, log_ridge),
        'log_ridge_policy': (WITH_POLICY, log_ridge),
        'elastic_core_a0.01': (CORE, lambda: elastic(.01)),
        'elastic_core_a0.1': (CORE, lambda: elastic(.1)),
        'tree_core_d2': (CORE, lambda: make_pipeline(SimpleImputer(strategy='median'),
                                                     DecisionTreeRegressor(max_depth=2, min_samples_leaf=4,
                                                                           random_state=SEED))),
    }


def load_batches():
    df = pd.read_csv(DATA)
    batches = {name: df[(df.batch == name) & df.cycle_life.notna()].copy().reset_index(drop=True)
               for name in ('batch1', 'batch2', 'batch3')}
    assert tuple(len(batches[k]) for k in batches) == (41, 43, 40)
    assert not batches['batch1'].cell_id.isin(batches['batch2'].cell_id).any()
    return batches


def split_batch1(batch1):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    dev_idx, hold_idx = next(splitter.split(batch1, groups=batch1.policy))
    dev, hold = batch1.iloc[dev_idx].copy(), batch1.iloc[hold_idx].copy()
    assert not set(dev.policy) & set(hold.policy)
    allocation = batch1[['batch', 'cell_id', 'policy', 'cycle_life']].copy()
    allocation['split'] = 'development'
    allocation.loc[hold_idx, 'split'] = 'holdout'
    allocation.to_csv(OUT / 'batch1_split.csv', index=False)
    return dev, hold


def select(dev):
    folds = []
    cv = GroupKFold(n_splits=5)
    for name, (features, factory) in candidates().items():
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(dev, groups=dev.policy), 1):
            fit, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
            model = factory()
            model.fit(fit[features], fit.cycle_life)
            pred = model.predict(valid[features])
            folds.append({'model': name, 'fold': fold, 'n_valid': len(valid),
                          'mape_pct': 100 * mean_absolute_percentage_error(valid.cycle_life, pred),
                          'mae_cycles': mean_absolute_error(valid.cycle_life, pred)})
    fold_df = pd.DataFrame(folds)
    summary = fold_df.groupby('model', as_index=False).agg(
        cv_mape_pct=('mape_pct', 'mean'), cv_mape_sd=('mape_pct', 'std'),
        cv_mae_cycles=('mae_cycles', 'mean'))
    summary = summary.sort_values('cv_mape_pct').reset_index(drop=True)
    fold_df.to_csv(OUT / 'candidate_cv_folds.csv', index=False)
    summary.to_csv(OUT / 'candidate_cv_summary.csv', index=False)
    return summary


def scores(actual, predicted):
    return {'mape_pct': 100 * mean_absolute_percentage_error(actual, predicted),
            'mae_cycles': mean_absolute_error(actual, predicted),
            'rmse_cycles': np.sqrt(mean_squared_error(actual, predicted)),
            'r2': r2_score(actual, predicted)}


def evaluate(model_name, dev, hold, batch1, batch2, batch3, summary):
    if model_name not in candidates():
        raise ValueError(f'Unknown model: {model_name}')
    features, factory = candidates()[model_name]
    cv_row = summary.set_index('model').loc[model_name]
    first = factory()
    first.fit(dev[features], dev.cycle_life)
    hold_pred = first.predict(hold[features])
    hold_scores = scores(hold.cycle_life, hold_pred)

    final = factory()
    final.fit(batch1[features], batch1.cycle_life)
    predictions = []
    test_scores = {}
    for label, sample, pred in [('valid_batch1', hold, hold_pred),
                                 ('test_batch2', batch2, final.predict(batch2[features])),
                                 ('additional_batch3', batch3, final.predict(batch3[features]))]:
        test_scores[label] = scores(sample.cycle_life, pred)
        for (_, row), value in zip(sample.iterrows(), pred):
            predictions.append({'set': label, 'cell_id': row.cell_id, 'policy': row.policy,
                                'actual': row.cycle_life, 'predicted': value,
                                'error': value - row.cycle_life,
                                'abs_error': abs(value - row.cycle_life),
                                'ape_pct': 100 * abs(value - row.cycle_life) / row.cycle_life})
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUT / 'cell_predictions.csv', index=False)
    pred_df['life_band'] = np.select(
        [pred_df.actual < 500, pred_df.actual > 1000],
        ['<500', '>1000'], default='500–1000')
    error_groups = pred_df.groupby(['set', 'life_band'], as_index=False).agg(
        n_cells=('cell_id', 'size'), actual_median=('actual', 'median'),
        mean_mape_pct=('ape_pct', 'mean'), mean_bias_cycles=('error', 'mean'))
    error_groups.to_csv(OUT / 'error_group_summary.csv', index=False)
    pred_df.sort_values('abs_error', ascending=False).groupby('set', group_keys=False).head(5).to_csv(
        OUT / 'worst_errors.csv', index=False)

    rows = [{'set': 'Train (Batch 1 CV)', 'n_cells': len(dev),
             'mape_pct': cv_row.cv_mape_pct, 'mae_cycles': cv_row.cv_mae_cycles},
            {'set': 'Valid (Batch 1 hold-out)', 'n_cells': len(hold), **hold_scores},
            {'set': 'Test (Batch 2)', 'n_cells': len(batch2), **test_scores['test_batch2']},
            {'set': 'Additional (Batch 3)', 'n_cells': len(batch3), **test_scores['additional_batch3']}]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'performance.csv', index=False)
    gap = {'valid_minus_train_pp': hold_scores['mape_pct'] - cv_row.cv_mape_pct,
           'test_minus_valid_pp': test_scores['test_batch2']['mape_pct'] - hold_scores['mape_pct'],
           'test_minus_paper_9.1_pp': test_scores['test_batch2']['mape_pct'] - 9.1,
           'batch3_minus_batch2_pp': test_scores['additional_batch3']['mape_pct'] - test_scores['test_batch2']['mape_pct'],
           'batch3_minus_paper_9.1_pp': test_scores['additional_batch3']['mape_pct'] - 9.1}
    reporting = pd.DataFrame([
        {'구분': 'Train (Batch 1 CV)', 'MAPE (%)': cv_row.cv_mape_pct, '비고': f'개발용 {len(dev)}셀, 정책별 5분할 평균'},
        {'구분': 'Valid (Batch 1 Hold-out)', 'MAPE (%)': hold_scores['mape_pct'], '비고': f'정책이 겹치지 않는 {len(hold)}셀'},
        {'구분': 'Test (Batch 2)', 'MAPE (%)': test_scores['test_batch2']['mape_pct'], '비고': f'Batch 1 전체 학습 후 {len(batch2)}셀 평가'},
        {'구분': 'Gap (Train-Valid)', 'MAPE (%)': gap['valid_minus_train_pp'], '비고': 'Valid − Train, +는 검증 오차 증가, 단위 %p'},
        {'구분': 'Gap (Valid-Test)', 'MAPE (%)': gap['test_minus_valid_pp'], '비고': 'Batch 2 − Valid, +는 배치 이동 시 오차 증가, 단위 %p'},
        {'구분': 'Gap (Target-Test)', 'MAPE (%)': gap['test_minus_paper_9.1_pp'], '비고': 'Batch 2 − 논문 참고값 9.1%, 단위 %p'},
        {'구분': 'Test (Batch 3)', 'MAPE (%)': test_scores['additional_batch3']['mape_pct'], '비고': '추가 평가 40셀'},
        {'구분': 'Gap (Batch2-Batch3)', 'MAPE (%)': gap['batch3_minus_batch2_pp'], '비고': 'Batch 3 − Batch 2, +는 Batch 3 오차 증가, 단위 %p'},
        {'구분': 'Gap (Target-Test, Batch 3)', 'MAPE (%)': gap['batch3_minus_paper_9.1_pp'], '비고': 'Batch 3 − 논문 참고값 9.1%, 단위 %p'},
    ])
    reporting['MAPE (%)'] = reporting['MAPE (%)'].round(2)
    reporting.to_csv(ROOT / 'results' / 'model_performance.csv', index=False)
    info = {'chosen_model': model_name, 'features': features, 'seed': SEED,
            'split': 'GroupShuffleSplit by charging policy, test_size=0.2',
            'cv': 'GroupKFold(5) on development cells only', 'n_development': len(dev),
            'n_holdout': len(hold), 'gaps': gap}
    (OUT / 'protocol.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    palette = {'valid_batch1': '#2a7f90', 'test_batch2': '#c85c42', 'additional_batch3': '#8065a8'}
    for label, group in pred_df.groupby('set'):
        axes[0].scatter(group.actual, group.predicted, label=label, alpha=.75, color=palette[label])
        axes[1].scatter(group.actual, group.error, label=label, alpha=.75, color=palette[label])
    bound = [min(pred_df.actual.min(), pred_df.predicted.min()),
             max(pred_df.actual.max(), pred_df.predicted.max())]
    axes[0].plot(bound, bound, '--', color='#555555', linewidth=1)
    axes[1].axhline(0, linestyle='--', color='#555555', linewidth=1)
    axes[0].set(xlabel='Actual cycle life', ylabel='Predicted cycle life', title='Prediction by evaluation set')
    axes[1].set(xlabel='Actual cycle life', ylabel='Prediction - actual (cycles)', title='Residual by cycle life')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'prediction_diagnostics.png', dpi=170)
    plt.close(fig)
    return metrics, gap, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['select', 'evaluate'], required=True)
    parser.add_argument('--model', help='Frozen model name chosen from development CV')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    batches = load_batches()
    dev, hold = split_batch1(batches['batch1'])
    summary = select(dev)
    print(f'Batch 1 development={len(dev)}, hold-out={len(hold)}; policy overlap=0')
    print(summary.round(2).to_string(index=False))
    if args.stage == 'evaluate':
        if not args.model:
            parser.error('--model is required for evaluate')
        metrics, gap, predictions = evaluate(args.model, dev, hold, *[batches[k] for k in ('batch1','batch2','batch3')], summary)
        print(metrics.round(2).to_string(index=False))
        print('Gaps (percentage points):', {k: round(v, 2) for k, v in gap.items()})
        print('Largest Batch 2 errors:')
        print(predictions[predictions['set'] == 'test_batch2'].nlargest(5, 'abs_error').round(2).to_string(index=False))


if __name__ == '__main__':
    main()
