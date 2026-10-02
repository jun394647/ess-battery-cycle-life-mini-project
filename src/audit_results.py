"""Check raw-file identity, cell lineage, early features and reported metrics."""
import argparse
from pathlib import Path
import json
import re
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

from day1_eda import FILES, SCREEN_EXCLUSIONS, CONTINUATION_LENGTHS
from day2_model import CORE, ROOT, candidates

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
EXPECTED_RAW = {'batch1': 46, 'batch2': 48, 'batch3': 46}
EXPECTED_SCREENED = {'batch1': 41, 'batch2': 43, 'batch3': 40}
TARGET_FILES = {
    'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}


def raw_checks(all_cells, screened):
    checks = {}
    for batch, filename in TARGET_FILES.items():
        assert FILES[batch] == filename
        raw = all_cells[all_cells.batch == batch].set_index('cell_id')
        clean = screened[screened.batch == batch].set_index('cell_id')
        assert len(raw) == EXPECTED_RAW[batch]
        assert len(clean) == EXPECTED_SCREENED[batch]
        assert not clean[CORE].isna().any().any()
        with h5py.File(DATA / filename) as f:
            b = f['batch']
            assert b['cycle_life'].shape[0] == EXPECTED_RAW[batch]
            for i in range(EXPECTED_RAW[batch]):
                cell_id = f'{batch}_{i:03d}'
                raw_life = float(f[b['cycle_life'][i, 0]][()].reshape(-1)[0])
                table_life = raw.loc[cell_id, 'cycle_life']
                assert (np.isnan(raw_life) and np.isnan(table_life)) or np.isclose(raw_life + (CONTINUATION_LENGTHS.get(i, 0) if batch == 'batch1' else 0), table_life)
                raw_policy = ''.join(chr(code) for code in f[b['policy_readable'][i, 0]][()].reshape(-1))
                assert raw.loc[cell_id, 'policy'] == raw_policy
                assert bool(raw.loc[cell_id, 'reference_exclusion']) == (i in SCREEN_EXCLUSIONS[batch])
                if i in SCREEN_EXCLUSIONS[batch]:
                    assert cell_id not in clean.index
                    continue
                assert cell_id in clean.index
                summary = f[b['summary'][i, 0]]
                cycle = summary['cycle'][()].reshape(-1)
                mask = (cycle >= 1) & (cycle <= 100)
                for source, field in [('QDischarge', 'early_mean_QD'), ('Tavg', 'early_mean_Tavg')]:
                    values = summary[source][()].reshape(-1)[:len(cycle)]
                    selected = values[mask].astype(float)
                    selected[selected <= 0] = np.nan
                    assert np.isclose(np.nanmean(selected), clean.loc[cell_id, field], rtol=1e-9, atol=1e-9)
                cycles = f[b['cycles'][i, 0]]['Qdlin']
                q10 = f[cycles[9, 0]][()].reshape(-1)
                q100 = f[cycles[99, 0]][()].reshape(-1)
                assert len(q10) == len(q100) == 1000
                logvar = np.log10(np.var(q100 - q10))
                assert np.isclose(logvar, clean.loc[cell_id, 'delta_q_logvar'], rtol=1e-9, atol=1e-9)
        checks[batch] = {'raw_cells': len(raw), 'analysis_cells': len(clean),
                         'excluded': sorted(SCREEN_EXCLUSIONS[batch])}
    with h5py.File(DATA / TARGET_FILES['batch2']) as f:
        b = f['batch']
        for i, extra in zip([7, 8, 9, 15, 16], CONTINUATION_LENGTHS.values()):
            s = f[b['summary'][i, 0]]
            assert s['cycle'][()].size == extra
    for i in CONTINUATION_LENGTHS:
        row = screened.set_index('cell_id').loc[f'batch1_{i:03d}']
        assert np.isnan(row.knee_cycle)
    return checks


def model_checks(screened):
    for batch in ('batch1', 'batch2', 'batch3'):
        expected = screened[screened.batch == batch].reset_index(drop=True)
        actual = pd.read_csv(RESULTS / 'model_inputs' / f'{batch}.csv')
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    split = pd.read_csv(RESULTS / 'day2/batch1_split.csv')
    assert set(split.cell_id) == set(screened[screened.batch == 'batch1'].cell_id)
    dev = split[split.split == 'development']
    hold = split[split.split == 'holdout']
    assert (len(dev), len(hold)) == (31, 10)
    assert not set(dev.policy) & set(hold.policy)
    protocol = json.loads((RESULTS / 'day2/protocol.json').read_text())
    model_name = protocol['chosen_model']
    feature_names, model_factory = candidates()[model_name]
    assert feature_names == protocol['features']
    sidx = screened.set_index('cell_id')
    development = sidx.loc[dev.cell_id].reset_index()
    saved_predictions = pd.read_csv(RESULTS / 'day2/model_comparison/screen_predictions.csv')
    saved_predictions = saved_predictions[saved_predictions.model == model_name]
    fold_scores = pd.read_csv(RESULTS / 'day2/candidate_cv_folds.csv')
    fold_scores = fold_scores[fold_scores.model == model_name].set_index('fold')
    assert len(saved_predictions) == len(development)
    for fold, (fit_idx, valid_idx) in enumerate(
            GroupKFold(n_splits=5).split(development, groups=development.policy), 1):
        fit, valid = development.iloc[fit_idx], development.iloc[valid_idx]
        assert not set(fit.policy) & set(valid.policy)
        model = model_factory()
        model.fit(fit[feature_names], fit.cycle_life)
        recalculated = model.predict(valid[feature_names])
        saved = saved_predictions[saved_predictions.fold == fold].set_index('cell_id').loc[valid.cell_id]
        assert len(saved) == len(valid)
        assert np.allclose(saved.actual, valid.cycle_life)
        assert np.allclose(saved.predicted, recalculated, rtol=1e-10, atol=1e-10)
        assert np.allclose(saved.ape_pct, 100 * abs(recalculated - valid.cycle_life) / valid.cycle_life)
        assert np.isclose(fold_scores.loc[fold, 'mape_pct'],
                          100 * mean_absolute_percentage_error(valid.cycle_life, recalculated))
        assert np.isclose(fold_scores.loc[fold, 'mae_cycles'],
                          mean_absolute_error(valid.cycle_life, recalculated))
    cv_mape = fold_scores.mape_pct.mean()
    cv_mae = fold_scores.mae_cycles.mean()
    for path in ('day2/candidate_cv_summary.csv', 'day2/model_comparison/screen_summary.csv'):
        summary = pd.read_csv(RESULTS / path).set_index('model').loc[model_name]
        assert np.isclose(summary.cv_mape_pct, cv_mape)
        assert np.isclose(summary.cv_mae_cycles, cv_mae)
    preds = pd.read_csv(RESULTS / 'day2/cell_predictions.csv')
    perf = pd.read_csv(RESULTS / 'day2/performance.csv')
    key = {'valid_batch1': 'Valid (Batch 1 hold-out)',
           'test_batch2': 'Test (Batch 2)',
           'additional_batch3': 'Additional (Batch 3)'}
    validation_model = model_factory()
    validation_model.fit(development[feature_names], development.cycle_life)
    full_batch1 = screened[screened.batch == 'batch1']
    final_model = model_factory()
    final_model.fit(full_batch1[feature_names], full_batch1.cycle_life)
    for set_name, model in [('valid_batch1', validation_model),
                            ('test_batch2', final_model),
                            ('additional_batch3', final_model)]:
        saved = preds[preds['set'] == set_name]
        recalculated = model.predict(sidx.loc[saved.cell_id, feature_names])
        assert np.allclose(saved.predicted, recalculated, rtol=1e-10, atol=1e-10)
    for name, g in preds.groupby('set'):
        assert (g.actual.to_numpy() == sidx.loc[g.cell_id, 'cycle_life'].to_numpy()).all()
        assert np.allclose(g.error, g.predicted - g.actual)
        assert np.allclose(g.ape_pct, 100 * abs(g.error) / g.actual)
        row = perf.set_index('set').loc[key[name]]
        assert len(g) == row.n_cells
        assert np.isclose(100 * mean_absolute_percentage_error(g.actual, g.predicted), row.mape_pct)
        assert np.isclose(mean_absolute_error(g.actual, g.predicted), row.mae_cycles)
    table = pd.read_csv(RESULTS / 'model_performance.csv').set_index('구분')['MAPE (%)']
    values = perf.set_index('set').mape_pct
    expected = {
        'Train (Batch 1 CV)': cv_mape,
        'Valid (Batch 1 Hold-out)': values['Valid (Batch 1 hold-out)'],
        'Test (Batch 2)': values['Test (Batch 2)'],
        'Test (Batch 3)': values['Additional (Batch 3)'],
        'Gap (Train-Valid)': values['Valid (Batch 1 hold-out)'] - cv_mape,
        'Gap (Valid-Test)': values['Test (Batch 2)'] - values['Valid (Batch 1 hold-out)'],
        'Gap (Target-Test)': values['Test (Batch 2)'] - 9.1,
        'Gap (Batch2-Batch3)': values['Additional (Batch 3)'] - values['Test (Batch 2)'],
        'Gap (Target-Test, Batch 3)': values['Additional (Batch 3)'] - 9.1,
    }
    assert set(table.index) == set(expected)
    for label, value in expected.items():
        assert np.isclose(table[label], round(value, 2), atol=1e-8), label
    assert np.isclose(perf.set_index('set').loc['Train (Batch 1 CV)', 'mape_pct'], cv_mape)
    return {'development_cells': len(dev), 'holdout_cells': len(hold),
            'model': model_name, 'cv_folds_refit': len(fold_scores),
            'batch2_mape_pct': float(perf.set_index('set').loc['Test (Batch 2)', 'mape_pct'])}


def document_checks():
    day1 = (ROOT / 'DAY1-REPORT.md').read_text()
    day2 = (ROOT / 'DAY2-REPORT.md').read_text()
    readme = (ROOT / 'README.md').read_text()
    for content in (day1, day2, readme):
        assert '2017-06-30' in content
        assert '41·43·40' in content or all(x in content for x in ['41셀', '43셀', '40셀'])
        for path in re.findall(r'!\[[^]]*\]\(([^)]+)\)', content):
            assert (ROOT / path).is_file(), path
    assert '842·481·965' in day1
    assert '-0.88·-0.65·-0.76' in day1
    assert '42셀' in day1  # One Batch 2 policy cannot be parsed as a two-stage C-rate.
    assert '수명 레이블에만' in day1  # Continuation curves were not concatenated.
    table = pd.read_csv(RESULTS / 'model_performance.csv')
    clean_day2 = day2.replace('−', '-').replace('–', '-')
    for _, row in table.iterrows():
        assert f'{row["MAPE (%)"]:+.2f}' in clean_day2 or f'{row["MAPE (%)"]:.2f}' in clean_day2
    batch2_mape = table.set_index('구분').loc['Test (Batch 2)', 'MAPE (%)']
    assert f'| Test (Batch 2) | **{batch2_mape:.2f}**' in day2
    assert f'| Test (Batch 2) | **{batch2_mape:.2f}**' in readme
    screen = pd.read_csv(RESULTS / 'day2/model_comparison/screen_summary.csv').set_index('model')
    assert len(screen) == 21
    for name in ('log_ridge_dq_qd', 'log_ridge_core', 'log_ridge_policy'):
        assert f'{screen.loc[name, "cv_mape_pct"]:.2f}' in day2
    ablation = pd.read_csv(RESULTS / 'day2/feature_ablation.csv').set_index('feature_set')
    assert set(ablation.index) == {'dq_only', 'dq_qd', 'dq_tavg', 'dq_qd_tavg'}
    assert np.isclose(ablation.loc['dq_qd', 'cv_mape_pct'], screen.loc['log_ridge_dq_qd', 'cv_mape_pct'])
    assert np.isclose(ablation.loc['dq_qd_tavg', 'cv_mape_pct'], screen.loc['log_ridge_core', 'cv_mape_pct'])
    for score in ablation.cv_mape_pct:
        assert f'{score:.2f}%' in day2
    pred = pd.read_csv(RESULTS / 'day2/cell_predictions.csv')
    b2 = pred[pred['set'] == 'test_batch2']
    b3 = pred[pred['set'] == 'additional_batch3']
    assert f'{b2.predicted.median():.0f}사이클' in day2
    assert f'{(b2[b2.actual < 500].error.mean()):.0f}사이클' in day2
    assert f'{abs(b3[b3.actual > 1000].error.mean()):.0f}사이클' in day2
    calibration = pd.read_csv(RESULTS / 'day2/batch_calibration/summary.csv').set_index('method')
    assert f'{calibration.loc["median_ratio", "mean_mape_pct"]:.2f}%' in day2
    return {'day1_figures': len(re.findall(r'!\[[^]]*\]\([^)]+\)', day1)),
            'day2_figures': len(re.findall(r'!\[[^]]*\]\([^)]+\)', day2)),
            'performance_rows_checked': len(table), 'candidate_configs_checked': len(screen),
            'feature_ablation_sets_checked': len(ablation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-only', action='store_true',
                        help='Verify committed tables, predictions, documents and refit models without raw .mat files')
    args = parser.parse_args()
    all_cells = pd.read_csv(RESULTS / 'all_cells.csv')
    screened = pd.read_csv(RESULTS / 'paper_screened_cells.csv')
    report = {'mode': 'artifacts-only' if args.artifacts_only else 'full',
              'raw_and_feature_checks': 'skipped' if args.artifacts_only else raw_checks(all_cells, screened),
              'model_checks': model_checks(screened),
              'document_checks': document_checks()}
    if not args.artifacts_only:
        (RESULTS / 'audit_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
