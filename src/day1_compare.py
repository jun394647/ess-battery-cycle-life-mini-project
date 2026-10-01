"""Compare three batch-level EDA outputs without training a model."""
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from day1_eda import SCREEN_EXCLUSIONS, CONTINUATION_LENGTHS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FIG = OUT / 'figures'
BATCHES = ['batch1', 'batch2', 'batch3']
frames = [pd.read_csv(OUT / f'{batch}_cells.csv') for batch in BATCHES]
cell = pd.concat(frames, ignore_index=True)
policy_pattern = re.compile(r'^(\d+(?:\.\d+)?)C\((\d+)%\)-(\d+(?:\.\d+)?)C')
def policy_components(policy):
    match = policy_pattern.search(policy)
    if not match:
        return pd.Series({'policy_switch_soc': np.nan, 'second_c_rate': np.nan,
                          'policy_rate_proxy': np.nan})
    first, switch, second = map(float, match.groups())
    return pd.Series({'policy_switch_soc': switch, 'second_c_rate': second,
                      'policy_rate_proxy': (first * switch + second * (80 - switch)) / 80})
cell = pd.concat([cell, cell.policy.apply(policy_components)], axis=1)
cell['life_adjustment_cycles'] = 0
for index, extra in CONTINUATION_LENGTHS.items():
    mask = (cell.batch == 'batch1') & (cell.cell_id == f'batch1_{index:03d}')
    assert mask.sum() == 1
    cell.loc[mask, 'cycle_life'] += extra
    cell.loc[mask, 'life_adjustment_cycles'] = extra
    # The Batch 1 summary ends before the continuation. A knee estimate from that
    # truncated curve must not be treated as an observed full-life knee.
    cell.loc[mask, ['knee_cycle', 'slope_before', 'slope_after']] = np.nan
cell['reference_exclusion'] = cell.apply(
    lambda r: int(r.cell_id.rsplit('_', 1)[1]) in SCREEN_EXCLUSIONS.get(r.batch, set()), axis=1)
cell.to_csv(OUT / 'all_cells.csv', index=False)
analysis = cell.loc[~cell.reference_exclusion].copy()
analysis.to_csv(OUT / 'paper_screened_cells.csv', index=False)

summary = analysis.groupby('batch').agg(
    cells=('cell_id', 'size'), labeled=('cycle_life', 'count'), mean_life=('cycle_life', 'mean'),
    median_life=('cycle_life', 'median'), min_life=('cycle_life', 'min'),
    q25_life=('cycle_life', lambda x: x.quantile(.25)),
    q75_life=('cycle_life', lambda x: x.quantile(.75)),
    max_life=('cycle_life', 'max'),
    pct_short=('cycle_life', lambda x: 100 * (x.dropna() < 500).mean()),
    pct_long=('cycle_life', lambda x: 100 * (x.dropna() > 1000).mean()),
    dq_available=('delta_q_var', lambda x: int(x.notna().sum())),
    knee_available=('knee_cycle', lambda x: int(x.notna().sum())),
    median_knee=('knee_cycle', 'median'),
)
summary.to_csv(OUT / 'batch_summary.csv')
print(summary.round(2).to_string())

features = ['first_c_rate','policy_rate_proxy','early_mean_QD','early_mean_IR','early_mean_Tavg',
            'early_mean_Tmax','early_mean_chargetime','delta_q_logvar',
            'delta_q_mean_abs']
rows = []
for batch, group in analysis.dropna(subset=['cycle_life']).groupby('batch'):
    for feature in features:
        clean = group[[feature, 'cycle_life']].replace([np.inf,-np.inf], np.nan).dropna()
        rows.append({'batch': batch, 'feature': feature, 'n': len(clean),
                     'pearson': clean[feature].corr(clean.cycle_life) if len(clean)>2 else np.nan,
                     'spearman': clean[feature].corr(clean.cycle_life, method='spearman') if len(clean)>2 else np.nan})
corr = pd.DataFrame(rows)
corr.to_csv(OUT / 'feature_correlations.csv', index=False)
print('\nCorrelations:')
print(corr.pivot(index='feature',columns='batch',values='spearman').round(2).to_string())

# Show feature redundancy in the Batch 1 cells used for the model probe.
heat_cols = {
    'Delta Q log var': 'delta_q_logvar',
    'Policy rate': 'policy_rate_proxy',
    'Charge time': 'early_mean_chargetime',
    'Mean QD': 'early_mean_QD',
    'Mean Tavg': 'early_mean_Tavg',
    'Mean Tmax': 'early_mean_Tmax',
    'Mean IR': 'early_mean_IR',
    'Cycle life': 'cycle_life',
}
heat_data = analysis.loc[(analysis.batch == 'batch1') & analysis.cycle_life.notna(),
                         list(heat_cols.values())].rename(columns={v:k for k,v in heat_cols.items()})
heat_corr = heat_data.corr(method='pearson')
fig,ax=plt.subplots(figsize=(8.5,7.2))
im=ax.imshow(heat_corr,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set_xticks(range(len(heat_corr)),heat_corr.columns,rotation=40,ha='right',fontsize=8)
ax.set_yticks(range(len(heat_corr)),heat_corr.index,fontsize=8)
for row in range(len(heat_corr)):
    for col in range(len(heat_corr)):
        value=heat_corr.iloc[row,col]
        ax.text(col,row,f'{value:+.2f}',ha='center',va='center',fontsize=7,
                color='white' if abs(value)>.68 else '#1b2933')
ax.set_title(f'Batch 1: Pearson correlations among early features (n={len(heat_data)})',fontsize=11,pad=14)
fig.colorbar(im,ax=ax,shrink=.78,label='Pearson r')
fig.tight_layout();fig.savefig(FIG/'batch1_feature_heatmap.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5))
for batch, group in analysis.dropna(subset=['cycle_life']).groupby('batch'):
    ax.hist(group.cycle_life,bins=np.arange(150,2401,100),alpha=.45,label=f'{batch} (n={len(group)})')
ax.set(xlabel='Cycle life',ylabel='Cells',title='Cycle life distribution by batch')
ax.legend();fig.tight_layout();fig.savefig(FIG/'compare_life.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
life_by_cell=analysis.set_index('cell_id').cycle_life.to_dict()
included=set(analysis.cell_id)
for ax,batch in zip(axes,BATCHES):
    cycle_data=pd.read_csv(OUT/f'{batch}_cycles.csv.gz',usecols=['cell_id','cycle','QD'])
    cycle_data=cycle_data[cycle_data.cell_id.isin(included) & cycle_data.QD.between(.5,1.3)]
    for cell_id,group in cycle_data.groupby('cell_id'):
        life=life_by_cell[cell_id]
        color='tab:blue' if life>1000 else 'tab:red' if life<500 else '0.55'
        ax.plot(group.cycle,group.QD,color=color,alpha=.38,lw=.65)
    ax.set(title=batch,xlabel='Cycle',xlim=(0,2000),ylim=(.5,1.2))
axes[0].set_ylabel('Discharge capacity QD (Ah)')
fig.suptitle('Discharge capacity by batch: blue >1000 / red <500 / gray other')
fig.tight_layout();fig.savefig(FIG/'compare_qd.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(batch,group) in zip(axes,analysis.dropna(subset=['cycle_life']).groupby('batch')):
    ax.scatter(group.first_c_rate,group.cycle_life,alpha=.6)
    ax.set(title=batch,xlabel='First-step C-rate',ylabel='Cycle life')
fig.tight_layout();fig.savefig(FIG/'compare_c_rate.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(batch,group) in zip(axes,analysis.dropna(subset=['cycle_life']).groupby('batch')):
    ax.scatter(group.policy_rate_proxy,group.cycle_life,alpha=.6)
    ax.set(title=batch,xlabel='SOC-weighted policy C-rate proxy',ylabel='Cycle life')
fig.tight_layout();fig.savefig(FIG/'compare_policy_rate.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(batch,group) in zip(axes,analysis.dropna(subset=['cycle_life']).groupby('batch')):
    ax.scatter(group.delta_q_logvar,group.cycle_life,alpha=.6)
    ax.set(title=batch,xlabel='log10 var ΔQ(V)',ylabel='Cycle life')
fig.tight_layout();fig.savefig(FIG/'compare_delta_q.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(batch,group) in zip(axes,analysis.dropna(subset=['cycle_life']).groupby('batch')):
    ax.scatter(group.early_mean_chargetime,group.cycle_life,alpha=.6)
    ax.set(title=batch,xlabel='Mean charge time, cycles 1–100',ylabel='Cycle life')
fig.tight_layout();fig.savefig(FIG/'compare_charge_time.png',dpi=160);plt.close(fig)

# Experimental knee: compare late and early fitted slopes, but never as a predictor.
analysis['knee_accelerates'] = np.where(
    analysis.slope_before.notna() & analysis.slope_after.notna(),
    analysis.slope_after < analysis.slope_before, np.nan)
knee = analysis.groupby('batch').agg(valid=('knee_cycle','count'),
                                  accelerates=('knee_accelerates','mean'),
                                  median_knee=('knee_cycle','median'))
knee.to_csv(OUT / 'knee_summary.csv')

policy = analysis.groupby(['batch','policy']).agg(n=('cell_id','size'),mean_life=('cycle_life','mean'),
                                             sd_life=('cycle_life','std'),first_c_rate=('first_c_rate','first')).reset_index()
policy.to_csv(OUT/'policy_summary.csv',index=False)

quality={}
for batch in BATCHES:
    issues=json.loads((OUT/f'{batch}_issues.json').read_text())
    quality[batch]={'issues':len(issues),'details':issues}
(OUT/'quality_summary.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))

# Multicollinearity among measurements available at prediction time.
for batch,group in analysis.dropna(subset=['cycle_life']).groupby('batch'):
    print(f'\n{batch}: Tavg/Tmax r={group.early_mean_Tavg.corr(group.early_mean_Tmax):.3f}; '
          f'charge time/C-rate r={group.early_mean_chargetime.corr(group.first_c_rate):.3f}')
