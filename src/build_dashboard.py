"""Export the audited cell-level results for the public interactive dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from day2_model import ROOT


OUTPUT = ROOT / 'docs' / 'data' / 'cells.json'
FIELDS = {
    'batch': 'batch',
    'cell_id': 'cell_id',
    'policy': 'policy',
    'cycle_life': 'actual',
    'first_c_rate': 'first_c_rate',
    'delta_q_logvar': 'delta_q_logvar',
    'early_mean_QD': 'early_mean_QD',
    'early_mean_IR': 'early_mean_IR',
}


def main() -> None:
    frames = [pd.read_csv(ROOT / 'results' / 'model_inputs' / f'{name}.csv')
              for name in ('batch1', 'batch2', 'batch3')]
    cells = pd.concat(frames, ignore_index=True)
    assert cells.batch.value_counts().to_dict() == {'batch1': 41, 'batch2': 43, 'batch3': 40}
    assert not cells.cell_id.duplicated().any()
    predictions = pd.read_csv(ROOT / 'results' / 'day2' / 'cell_predictions.csv')
    predictions = predictions[predictions['set'].isin(['test_batch2', 'additional_batch3'])]
    assert predictions['set'].value_counts().to_dict() == {'test_batch2': 43, 'additional_batch3': 40}
    joined = cells.merge(predictions[['cell_id', 'actual', 'predicted', 'ape_pct']],
                         on='cell_id', how='left', validate='one_to_one', suffixes=('', '_prediction'))
    evaluated = joined[joined.batch.isin(['batch2', 'batch3'])]
    assert evaluated.predicted.notna().all()
    assert np.allclose(evaluated.cycle_life, evaluated.actual)
    assert np.allclose(evaluated.ape_pct, 100 * abs(evaluated.predicted - evaluated.cycle_life)
                       / evaluated.cycle_life)
    assert joined.loc[joined.batch == 'batch1', 'predicted'].isna().all()
    batch2 = evaluated[evaluated.batch == 'batch2']
    score = 100 * np.mean(abs(batch2.predicted - batch2.cycle_life) / batch2.cycle_life)
    assert np.isclose(score, 19.392440956961455)

    records = []
    for _, row in joined.sort_values(['batch', 'cell_id']).iterrows():
        item = {name: row[source] for source, name in FIELDS.items()}
        item['predicted'] = row.predicted
        item['ape_pct'] = row.ape_pct
        item = {key: (None if pd.isna(value) else round(float(value), 8)
                      if isinstance(value, (float, np.floating, int, np.integer)) else value)
                for key, value in item.items()}
        records.append(item)
    output = {
        'description': 'One row per screened cell. Predictions exist only for Batch 2 and Batch 3.',
        'source': ['results/model_inputs/batch1.csv', 'results/model_inputs/batch2.csv',
                   'results/model_inputs/batch3.csv', 'results/day2/cell_predictions.csv'],
        'counts': {'batch1': 41, 'batch2': 43, 'batch3': 40},
        'batch2_mape_pct': round(score, 8),
        'cells': records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Wrote {len(records)} cells to {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
