"""Two focused Day 1 figures from the already screened cell table."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'paper_screened_cells.csv'
OUT = ROOT / 'results' / 'figures'


def main():
    cells = pd.read_csv(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    batches = ('batch1', 'batch2', 'batch3')
    colors = {'shorter': '#c75b4b', 'longer': '#317e9c'}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), sharey=True)
    for ax, batch in zip(axes, batches):
        group = cells[cells.batch == batch].copy()
        median_life = group.cycle_life.median()
        group['life_half'] = np.where(group.cycle_life <= median_life, 'shorter', 'longer')
        for pos, label in enumerate(('shorter', 'longer')):
            values = group.loc[group.life_half == label, 'delta_q_logvar'].dropna()
            ax.scatter(np.full(len(values), pos) + np.linspace(-.11, .11, len(values)), values,
                       color=colors[label], s=20, alpha=.65)
            ax.plot([pos-.18, pos+.18], [values.median()] * 2, color='black', lw=2.2)
            ax.text(pos, .96, f'n={len(values)}', ha='center', va='top', fontsize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks((0, 1), ('shorter half', 'longer half'))
        ax.set_title(f'{batch} | median life {median_life:g}')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('log10 var[Q100(V) - Q10(V)]')
    fig.suptitle('Early curve change within each batch: points are cells, bars are medians')
    fig.tight_layout()
    fig.savefig(OUT / 'within_batch_delta_q.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, batch in zip(axes, batches):
        group = cells[(cells.batch == batch) & cells.knee_cycle.notna()]
        ax.scatter(group.cycle_life, group.knee_cycle, alpha=.68, s=22, color='#357f9e')
        ax.plot([0, 2400], [0, 2400], '--', color='#888888', lw=.8)
        ax.set_title(f'{batch} | n={len(group)}')
        ax.set_xlabel('Cycle life')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Exploratory knee cycle')
    fig.suptitle('Knee estimate requires the full later-life QD curve')
    fig.tight_layout()
    fig.savefig(OUT / 'knee_vs_life.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
