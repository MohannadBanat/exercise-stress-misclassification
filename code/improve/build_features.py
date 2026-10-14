"""Shared feature construction (v1 + v2, raw / calibrated / z-scored) for M3 and M4."""
import numpy as np, pandas as pd
from features_v1 import add_v1, ACC
def build():
    W = pd.read_csv('features/windows.csv'); F2 = pd.read_csv('features/windows_v2.csv', index_col=0).drop(columns=['eda_method'])
    W = W.join(F2); V2 = list(F2.columns)
    base = W[(W.protocol == 'STRESS') & (W.block == 'baseline')].groupby('participant')[V2]
    mu, sd = base.mean(), base.std()
    for c in V2:
        W['c2_' + c] = W[c] - W.participant.map(mu[c])
        W['z2_' + c] = (W[c] - W.participant.map(mu[c])) / W.participant.map(sd[c]).replace(0, np.nan)
    W, FS = add_v1(W)
    FS2 = {'v1_cal_ctx': FS['cal_ctx'], 'v2_z': ['z2_' + c for c in V2] + ACC, 'v2_raw': V2 + ACC,
           'v2_cal': ['c2_' + c for c in V2] + ACC, 'v12_cal': FS['cal_ctx'] + ['c2_' + c for c in V2]}
    return W, FS2, ACC
