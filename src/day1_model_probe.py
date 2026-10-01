"""Exploratory Batch 1-only, policy-grouped CV for Day 1 strategy decisions.

This is a precheck of feature/model candidates. Batch 2 is never loaded here.
The final training/hold-out/test protocol belongs to Day 2.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
df=pd.read_csv(OUT/'paper_screened_cells.csv')
df=df[(df.batch=='batch1') & df.cycle_life.notna()].copy().reset_index(drop=True)
assert len(df)==46

basic=['early_mean_QD','early_mean_Tavg','early_mean_chargetime']
delta=['delta_q_logvar']
policy_proxy=['policy_rate_proxy']
compact=['delta_q_logvar','early_mean_QD','early_mean_Tavg']
compact_with_policy=compact+policy_proxy
experiments={
 'median_baseline': (basic, lambda: DummyRegressor(strategy='median')),
 'ridge_basic': (basic, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0))),
 'ridge_delta_only': (delta, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0))),
 'ridge_policy_proxy_only': (policy_proxy, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0))),
 'ridge_compact': (compact, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0))),
 'ridge_compact_with_policy': (compact_with_policy, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0))),
 'elasticnet_compact': (compact, lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),ElasticNet(alpha=.1,l1_ratio=.5,max_iter=10000))),
 'tree_compact': (compact, lambda: make_pipeline(SimpleImputer(strategy='median'),DecisionTreeRegressor(max_depth=2,min_samples_leaf=4,random_state=42))),
}
cv=GroupKFold(n_splits=5)
rows=[]; predictions=[]
for name,(features,create) in experiments.items():
    fold_map={};yhat=np.full(len(df),np.nan)
    for fold,(train_idx,valid_idx) in enumerate(cv.split(df,df.cycle_life,groups=df.policy),start=1):
        model=create()
        model.fit(df.iloc[train_idx][features],df.iloc[train_idx].cycle_life)
        pred=model.predict(df.iloc[valid_idx][features])
        yhat[valid_idx]=pred
        y=df.iloc[valid_idx].cycle_life.to_numpy()
        rows.append({'experiment':name,'fold':fold,'n_valid':len(valid_idx),
                     'features':'+'.join(features), 'mape_pct':100*mean_absolute_percentage_error(y,pred),
                     'mae_cycles':mean_absolute_error(y,pred)})
        for i,p in zip(valid_idx,pred): fold_map[i]=fold
    assert np.isfinite(yhat).all()
    for i,p in enumerate(yhat):
        predictions.append({'experiment':name,'cell_id':df.iloc[i].cell_id,'policy':df.iloc[i].policy,
                            'fold':fold_map[i],'actual':df.iloc[i].cycle_life,'predicted':p,
                            'abs_error':abs(df.iloc[i].cycle_life-p)})
folds=pd.DataFrame(rows)
preds=pd.DataFrame(predictions)
summary=folds.groupby('experiment').agg(mean_mape_pct=('mape_pct','mean'),sd_mape_pct=('mape_pct','std'),
                                         mean_mae_cycles=('mae_cycles','mean')).reset_index()
pooled=preds.groupby('experiment').apply(lambda g: pd.Series({
    'pooled_mape_pct':100*mean_absolute_percentage_error(g.actual,g.predicted),
    'pooled_mae_cycles':mean_absolute_error(g.actual,g.predicted)}),include_groups=False).reset_index()
summary=summary.merge(pooled,on='experiment').sort_values('mean_mape_pct')
folds.to_csv(OUT/'day1_probe_folds.csv',index=False)
preds.to_csv(OUT/'day1_probe_predictions.csv',index=False)
summary.to_csv(OUT/'day1_probe_summary.csv',index=False)
print(summary.round(2).to_string(index=False))
fig,ax=plt.subplots(figsize=(9,4))
ax.barh(summary.experiment,summary.mean_mape_pct,xerr=summary.sd_mape_pct,color='#4a86a8',alpha=.85)
ax.invert_yaxis();ax.set(xlabel='Batch 1 policy-grouped CV MAPE (%)',title='Day 1 exploratory model probe (5 folds)')
fig.tight_layout();fig.savefig(OUT/'figures/day1_probe.png',dpi=160);plt.close(fig)
