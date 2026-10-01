"""Cell-level EDA for the MIT–Stanford battery batches.

Run each batch in a separate process to release the large MATLAB object afterwards.
The script writes a per-cycle table, one row per cell, data checks, and figures.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mat73
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILES = {
    'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
NUMERIC = {'QD': 'QDischarge', 'QC': 'QCharge', 'IR': 'IR', 'Tavg': 'Tavg',
           'Tmax': 'Tmax', 'Tmin': 'Tmin', 'chargetime': 'chargetime'}


def vector(value):
    if value is None:
        return np.array([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def scalar(value):
    arr = vector(value)
    return float(arr[0]) if len(arr) else np.nan


def policy_c_rate(policy: str):
    """First step C-rate, e.g. 3.6C(80%)-3.6C or 3_6C-80PER_3_6C."""
    match = re.search(r'(\d+(?:[._]\d+)?)\s*C', policy, re.I)
    return float(match.group(1).replace('_', '.')) if match else np.nan


def cycle_value(cell, field, idx):
    cycles = cell.get('cycles', {})
    try:
        value = cycles[field][idx] if isinstance(cycles, dict) else cycles[idx][field]
    except (KeyError, IndexError, TypeError):
        return None
    a = vector(value)
    return a if len(a) and np.isfinite(a).all() else None


def knee_point(group: pd.DataFrame):
    """Exploratory two-line fit; this is not a validated physical knee estimate."""
    g = group.loc[group['QD'].between(.5, 1.3), ['cycle', 'QD']].dropna()
    g = g.drop_duplicates('cycle').sort_values('cycle')
    if len(g) < 150:
        return np.nan, np.nan, np.nan
    x, y = g['cycle'].to_numpy(), g['QD'].to_numpy()
    # Smooth isolated measurement spikes; segment boundary search is kept away from edges.
    y = pd.Series(y).rolling(21, center=True, min_periods=10).median().to_numpy()
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 150:
        return np.nan, np.nan, np.nan
    # Downsample to keep the search affordable for long-lived cells.
    x, y = x[::5], y[::5]
    lo, hi = max(15, int(.2 * len(x))), min(len(x) - 15, int(.8 * len(x)))
    best = (np.inf, np.nan, np.nan, np.nan)
    for k in range(lo, hi):
        p1, p2 = np.polyfit(x[:k], y[:k], 1), np.polyfit(x[k:], y[k:], 1)
        error = np.sum((y[:k] - np.polyval(p1, x[:k]))**2) + np.sum((y[k:] - np.polyval(p2, x[k:]))**2)
        if error < best[0]:
            best = (error, x[k], p1[0], p2[0])
    _, where, before, after = best
    return float(where), float(before), float(after)


def extract(batch_name: str, data_dir: Path, out_dir: Path):
    path = data_dir / FILES[batch_name]
    if not path.exists():
        raise FileNotFoundError(path)
    print(f'Loading {path}', flush=True)
    mat = mat73.loadmat(str(path), verbose=False)
    raw = mat['batch']
    batch = ([{key: value[i] for key, value in raw.items()}
              for i in range(len(next(iter(raw.values()))))]
             if isinstance(raw, dict) else list(raw))
    cycles, cells, issues, dq_curves = [], [], [], []
    for cid, cell in enumerate(batch):
        cell_id = f'{batch_name}_{cid:03d}'
        summary = cell['summary']
        arrays = {name: vector(summary[source]) for name, source in NUMERIC.items()}
        raw_cycle = vector(summary.get('cycle'))
        n = min(*(len(a) for a in arrays.values()), len(raw_cycle))
        if len({len(a) for a in arrays.values()} | {len(raw_cycle)}) > 1:
            issues.append({'cell_id': cell_id, 'issue': 'summary field lengths differ'})
        life = scalar(cell.get('cycle_life'))
        if not np.isfinite(life):
            issues.append({'cell_id': cell_id, 'issue': 'cycle_life missing; excluded from target-based analysis'})
        policy = str(cell.get('policy_readable') or cell.get('policy') or 'unknown')
        frame = pd.DataFrame({'batch': batch_name, 'cell_id': cell_id,
                              'cycle': raw_cycle[:n], 'cycle_life': life,
                              'policy': policy, **{key: a[:n] for key, a in arrays.items()}})
        cycles.append(frame)
        early = frame.loc[frame.cycle.between(1, 100)].copy()
        if (early['Tmax'] > 100).any():
            issues.append({'cell_id': cell_id, 'issue': 'Tmax > 100 C in early cycles; inspect sensor or sentinel value'})
        for name in NUMERIC:
            early.loc[early[name] <= 0, name] = np.nan
        row = {'batch': batch_name, 'cell_id': cell_id, 'cycle_life': life,
               'policy': policy, 'first_c_rate': policy_c_rate(policy),
               'n_cycles': len(frame), 'n_early': len(early)}
        for name in NUMERIC:
            row[f'early_mean_{name}'] = early[name].mean()
        q10, q100 = cycle_value(cell, 'Qdlin', 9), cycle_value(cell, 'Qdlin', 99)
        if q10 is not None and q100 is not None and len(q10) == len(q100):
            dq = q100 - q10
            row['delta_q_var'] = float(np.var(dq))
            row['delta_q_logvar'] = float(np.log10(np.var(dq))) if np.var(dq) > 0 else np.nan
            row['delta_q_mean_abs'] = float(np.mean(np.abs(dq)))
            row['delta_q_min'] = float(np.min(dq))
            dq_curves.append((cell_id, life, dq))
        else:
            issues.append({'cell_id': cell_id, 'issue': 'Qdlin cycle 10/100 missing or unequal length'})
        row['knee_cycle'], row['slope_before'], row['slope_after'] = knee_point(frame)
        cells.append(row)
        if cid % 10 == 0:
            print(f'{batch_name}: {cid+1}/{len(batch)} cells', flush=True)
    cycle_df = pd.concat(cycles, ignore_index=True)
    cell_df = pd.DataFrame(cells)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    cycle_df.to_csv(out_dir / f'{batch_name}_cycles.csv.gz', index=False, compression='gzip')
    cell_df.to_csv(out_dir / f'{batch_name}_cells.csv', index=False)
    (out_dir / f'{batch_name}_issues.json').write_text(json.dumps(issues, ensure_ascii=False, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cell_df.cycle_life, bins=np.arange(150, 2401, 100), edgecolor='white')
    ax.set(xlabel='Cycle life', ylabel='Cells', title=f'{batch_name}: cycle life')
    fig.tight_layout(); fig.savefig(fig_dir / f'{batch_name}_life.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for frame in cycles:
        ok = frame['QD'].between(.5, 1.3)
        ax.plot(frame.loc[ok, 'cycle'], frame.loc[ok, 'QD'], alpha=.45, lw=.8)
    ax.set(xlabel='Cycle', ylabel='QD (Ah)', title=f'{batch_name}: discharge capacity')
    fig.tight_layout(); fig.savefig(fig_dir / f'{batch_name}_qd.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for cell_id, life, dq in dq_curves:
        color = 'tab:blue' if life > 1000 else 'tab:red' if life < 500 else '0.6'
        # The source paper interpolates Qdlin at 1,000 points from 3.5 V to 2.0 V.
        ax.plot(np.linspace(3.5, 2.0, len(dq)), dq, color=color, alpha=.45, lw=.7)
    ax.set(xlabel='Interpolated voltage (V)', ylabel='Q100(V) - Q10(V) (Ah)',
           title=f'{batch_name}: delta Q(V), blue >1000 / red <500 / gray other')
    fig.tight_layout(); fig.savefig(fig_dir / f'{batch_name}_delta_q.png', dpi=150); plt.close(fig)
    print(f'Wrote {batch_name}: {len(cell_df)} cells, {len(cycle_df)} cycles, {len(issues)} issues', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', choices=FILES)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    extract(args.batch, args.data_dir, args.out_dir)
