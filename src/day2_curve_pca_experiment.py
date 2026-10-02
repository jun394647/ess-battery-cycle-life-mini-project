"""Compare early ΔQ(V) curve shape PCA with the three-feature baseline.

PCA and scaling are fit inside each Batch 1 CV fold. Batch 2/3 are read only
with --evaluate. These are retrospective scores after earlier Batch 2 reviews.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day1_eda import FILES
from day2_model import ROOT, load_batches, split_batch1
from day2_trajectory_experiment import model as baseline_model


OUT = ROOT / 'results' / 'day2' / 'curve_pca_experiment'
META = ['delta_q_logvar', 'early_mean_QD', 'early_mean_IR']


def load_curves(batch: pd.DataFrame, batch_name: str) -> np.ndarray:
    curves = []
    with h5py.File(ROOT / 'data' / FILES[batch_name]) as raw:
        refs = raw['batch']['cycles']
        for cell_id in batch.cell_id:
            i = int(cell_id.rsplit('_', 1)[1])
            qdlin = raw[refs[i, 0]]['Qdlin']
            q10 = np.asarray(raw[qdlin[9, 0]][()]).reshape(-1)
            q100 = np.asarray(raw[qdlin[99, 0]][()]).reshape(-1)
            change = q100 - q10
            if len(change) != 1000 or not np.isfinite(change).all():
                raise ValueError(f'{cell_id}: invalid Qdlin curve')
            if not np.isclose(np.log10(np.var(change)),
                              batch.loc[batch.cell_id.eq(cell_id), 'delta_q_logvar'].iloc[0], atol=1e-10):
                raise ValueError(f'{cell_id}: raw curve disagrees with saved feature')
            curves.append(change)
    return np.stack(curves)


def inputs(batch: pd.DataFrame, curves: np.ndarray, *, shape: bool) -> np.ndarray:
    if shape:
        curves = curves / np.std(curves, axis=1, keepdims=True)
    return np.column_stack((batch[META].to_numpy(), curves))


def estimator(n_components: int, alpha: float):
    prep = ColumnTransformer([
        ('meta', StandardScaler(), list(range(len(META)))),
        ('curve', make_pipeline(PCA(n_components=n_components), StandardScaler()),
         list(range(len(META), len(META) + 1000))),
    ])
    return TransformedTargetRegressor(regressor=make_pipeline(prep, Ridge(alpha=alpha)),
                                      func=np.log, inverse_func=np.exp)


def mape(y, pred) -> float:
    return 100 * mean_absolute_percentage_error(y, pred)


def screen(batch: pd.DataFrame, curves: np.ndarray) -> pd.DataFrame:
    dev, hold = split_batch1(batch)
    rows = []
    for shape in (False, True):
        x = inputs(batch, curves, shape=shape)
        for n_components in (1, 2, 4):
            for alpha in (0.1, 1.0, 10.0):
                scores = []
                for fit_idx, val_idx in GroupKFold(5).split(dev, groups=dev.policy):
                    reg = estimator(n_components, alpha)
                    reg.fit(x[dev.index[fit_idx]], dev.cycle_life.iloc[fit_idx])
                    scores.append(mape(dev.cycle_life.iloc[val_idx], reg.predict(x[dev.index[val_idx]])))
                reg = estimator(n_components, alpha)
                reg.fit(x[dev.index], dev.cycle_life)
                rows.append({'curve': 'shape' if shape else 'raw', 'components': n_components,
                             'alpha': alpha, 'cv_mape_pct': np.mean(scores),
                             'holdout_mape_pct': mape(hold.cycle_life, reg.predict(x[hold.index])),
                             'fold_mape_pct': repr([round(score, 4) for score in scores])})
    # An explicit baseline calculated on the same cells and folds.
    scores = []
    for fit_idx, val_idx in GroupKFold(5).split(dev, groups=dev.policy):
        reg = baseline_model(.1)
        reg.fit(dev.iloc[fit_idx][META], dev.cycle_life.iloc[fit_idx])
        scores.append(mape(dev.cycle_life.iloc[val_idx], reg.predict(dev.iloc[val_idx][META])))
    reg = baseline_model(.1)
    reg.fit(dev[META], dev.cycle_life)
    rows.append({'curve': 'baseline', 'components': 0, 'alpha': 0.1,
                 'cv_mape_pct': np.mean(scores),
                 'holdout_mape_pct': mape(hold.cycle_life, reg.predict(hold[META])),
                 'fold_mape_pct': repr([round(score, 4) for score in scores])})
    result = pd.DataFrame(rows).sort_values('cv_mape_pct').reset_index(drop=True)
    result.to_csv(OUT / 'screen.csv', index=False)
    return result


def evaluate(batch1: pd.DataFrame, curves1: np.ndarray) -> pd.DataFrame:
    # Two Batch 1 candidates: lowest CV (2 axes), and lowest joint CV/holdout (1 axis).
    rows, predictions = [], []
    external = {}
    for name in ('batch2', 'batch3'):
        batch = load_batches(name)[name]
        external[name] = (batch, inputs(batch, load_curves(batch, name), shape=True))
    for components in (1, 2):
        candidate = f'shape_pca{components}_a0.1'
        reg = estimator(components, .1)
        reg.fit(inputs(batch1, curves1, shape=True), batch1.cycle_life)
        for name, (batch, x) in external.items():
            pred = reg.predict(x)
            rows.append({'candidate': candidate, 'set': name, 'n_cells': len(batch),
                         'mape_pct': mape(batch.cycle_life, pred)})
            predictions.extend({'candidate': candidate, 'set': name, 'cell_id': row.cell_id,
                                'actual': row.cycle_life, 'predicted': value,
                                'ape_pct': 100 * abs(value-row.cycle_life) / row.cycle_life}
                               for (_, row), value in zip(batch.iterrows(), pred))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'evaluation.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Read Batch 2/3 labels after Batch 1 screening')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    batch1 = load_batches('batch1')['batch1']
    curves = load_curves(batch1, 'batch1')
    summary = screen(batch1, curves)
    print(summary.head(10).round(3).to_string(index=False))
    if args.evaluate:
        print('\nRetrospective external evaluation:')
        print(evaluate(batch1, curves).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
