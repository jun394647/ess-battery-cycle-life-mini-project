"""Check raw-file identity, cell lineage, early features and reported metrics."""
from pathlib import Path
import json
import re
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from day1_eda import FILES, SCREEN_EXCLUSIONS, CONTINUATION_LENGTHS
from day2_model import CORE, LEAN, ROOT, candidates

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
    split = pd.read_csv(RESULTS / 'day2/batch1_split.csv')
    assert set(split.cell_id) == set(screened[screened.batch == 'batch1'].cell_id)
    dev = split[split.split == 'development']
    hold = split[split.split == 'holdout']
    assert (len(dev), len(hold)) == (31, 10)
    assert not set(dev.policy) & set(hold.policy)
    folds = pd.read_csv(RESULTS / 'day2/model_comparison/screen_predictions.csv')
    chosen = folds[folds.model == 'log_ridge_dq_qd']
    for fold, valid in chosen.groupby('fold'):
        other = dev[~dev.cell_id.isin(valid.cell_id)]
        assert not set(valid.cell_id) & set(other.cell_id)
        assert not set(valid.cell_id.map(split.set_index('cell_id').policy)) & set(other.policy)
    preds = pd.read_csv(RESULTS / 'day2/cell_predictions.csv')
    perf = pd.read_csv(RESULTS / 'day2/performance.csv')
    key = {'valid_batch1': 'Valid (Batch 1 hold-out)',
           'test_batch2': 'Test (Batch 2)',
           'additional_batch3': 'Additional (Batch 3)'}
    sidx = screened.set_index('cell_id')
    feature_names, model_factory = candidates()['log_ridge_dq_qd']
    assert feature_names == LEAN
    development = sidx.loc[dev.cell_id]
    validation = sidx.loc[hold.cell_id]
    validation_model = model_factory()
    validation_model.fit(development[LEAN], development.cycle_life)
    full_batch1 = screened[screened.batch == 'batch1']
    final_model = model_factory()
    final_model.fit(full_batch1[LEAN], full_batch1.cycle_life)
    for set_name, model in [('valid_batch1', validation_model),
                            ('test_batch2', final_model),
                            ('additional_batch3', final_model)]:
        saved = preds[preds['set'] == set_name]
        recalculated = model.predict(sidx.loc[saved.cell_id, LEAN])
        assert np.allclose(saved.predicted, recalculated, rtol=1e-10, atol=1e-10)
    for name, g in preds.groupby('set'):
        assert (g.actual.to_numpy() == sidx.loc[g.cell_id, 'cycle_life'].to_numpy()).all()
        assert np.allclose(g.error, g.predicted - g.actual)
        assert np.allclose(g.ape_pct, 100 * abs(g.error) / g.actual)
        row = perf.set_index('set').loc[key[name]]
        assert len(g) == row.n_cells
        assert np.isclose(100 * mean_absolute_percentage_error(g.actual, g.predicted), row.mape_pct)
        assert np.isclose(mean_absolute_error(g.actual, g.predicted), row.mae_cycles)
    table = pd.read_csv(RESULTS / 'model_performance.csv').set_index('구분')
    assert np.isclose(table.loc['Train (Batch 1 CV)', 'MAPE (%)'], 7.59)
    assert np.isclose(table.loc['Valid (Batch 1 Hold-out)', 'MAPE (%)'], 7.82)
    assert np.isclose(table.loc['Test (Batch 2)', 'MAPE (%)'], 24.11)
    assert np.isclose(table.loc['Test (Batch 3)', 'MAPE (%)'], 11.96)
    return {'development_cells': len(dev), 'holdout_cells': len(hold),
            'model': json.loads((RESULTS / 'day2/protocol.json').read_text())['chosen_model'],
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
    assert '| Test (Batch 2) | **24.11**' in day2
    assert '| Test (Batch 2) | **24.11**' in readme
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
    all_cells = pd.read_csv(RESULTS / 'all_cells.csv')
    screened = pd.read_csv(RESULTS / 'paper_screened_cells.csv')
    report = {'raw_and_feature_checks': raw_checks(all_cells, screened),
              'model_checks': model_checks(screened),
              'document_checks': document_checks()}
    (RESULTS / 'audit_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
