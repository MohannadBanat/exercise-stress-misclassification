"""M1: baseline = the exploratory pipeline space, nested LOPO on the 24 development participants."""
import sys, json, pandas as pd
sys.path.insert(0, 'improve')
from framework import *; from features_v1 import add_v1
W, FS = add_v1(pd.read_csv('features/windows.csv'))
d = detector_frame(W); log('dev participants', d.participant.nunique(), 'windows', len(d), 'lockbox present:', bool(set(d.participant) & set(LOCK)))
prev = {k: prev_index(d, k) for k in (45,)}
C = [dict(fs=fs, model=m, k=k, rule='global') for fs in FS for m in ('xgb', 'lr', 'et') for k in (15, 45)]
res = nested_lopo(d, C, FS, prev, tag='M1_baseline'); res.to_csv('improve/res_M1.csv', index=False)
s = summarize(res, 'M1 baseline (exploratory pipeline space)'); json.dump(s, open('improve/sum_M1.json', 'w'), indent=1); print(json.dumps(s, indent=1))
