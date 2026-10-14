"""M2: improved features. Candidate feature sets (v1 best vs v2 multi-scale, raw / baseline-calibrated / baseline z-scored,
and v1+v2), models XGBoost and extremely randomized trees, smoothing 15 or 45 s; nested LOPO on the 24 dev participants."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'improve')
from framework import *; from features_v1 import add_v1, ACC
W = pd.read_csv('features/windows.csv'); F2 = pd.read_csv('features/windows_v2.csv', index_col=0).drop(columns=['eda_method'])
W = W.join(F2)                                   # same row order as windows.csv (row_id = index)
V2 = list(F2.columns)
base = W[(W.protocol == 'STRESS') & (W.block == 'baseline')].groupby('participant')[V2]
mu, sd = base.mean(), base.std()
for c in V2:
    W['c2_' + c] = W[c] - W.participant.map(mu[c])
    W['z2_' + c] = (W[c] - W.participant.map(mu[c])) / W.participant.map(sd[c]).replace(0, np.nan)
W, FS = add_v1(W)
FS2 = {'v1_cal_ctx': FS['cal_ctx'], 'v2_raw': V2 + ACC, 'v2_cal': ['c2_' + c for c in V2] + ACC,
       'v2_z': ['z2_' + c for c in V2] + ACC, 'v12_cal': FS['cal_ctx'] + ['c2_' + c for c in V2]}
d = detector_frame(W); log('dev', d.participant.nunique(), len(d), 'lockbox present:', bool(set(d.participant) & set(LOCK)))
prev = {k: prev_index(d, k) for k in (45,)}
C = [dict(fs=fs, model=m, k=k, rule='global') for fs in FS2 for m in ('xgb', 'et') for k in (15, 45)]
res = nested_lopo(d, C, FS2, prev, tag='M2_features'); res.to_csv('improve/res_M2.csv', index=False)
s = summarize(res, 'M2 features'); json.dump(s, open('improve/sum_M2.json', 'w'), indent=1); print(json.dumps(s, indent=1))
