"""Feature sets from the registered window features (v1): raw, personally calibrated, calibrated + causal context."""
import numpy as np, pandas as pd
EDA = ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']
BVP = ['bvp_pr_mean','bvp_pr_sd','bvp_sdnn','bvp_rmssd','bvp_amp_sd','bvp_coverage']
HR = ['hr_mean']; TEMP = ['temp_mean','temp_sd','temp_slope']
ACC = ['acc_mag_mean','acc_mag_sd','acc_x_mean','acc_x_sd','acc_y_mean','acc_y_sd','acc_z_mean','acc_z_sd','acc_dom_freq','acc_pow_frac_0p5_2']
PHYS = EDA + BVP + HR + TEMP
LEVEL = ['eda_tonic_mean','eda_tonic_min','eda_tonic_max','bvp_pr_mean','hr_mean','temp_mean','eda_phasic_area','eda_scr_count']
def add_v1(W):
    W = W.copy()
    base = W[(W.protocol == 'STRESS') & (W.block == 'baseline')].groupby('participant')[PHYS].mean()
    for c in PHYS: W['cal_' + c] = W[c] - W.participant.map(base[c])
    W = W.sort_values(['protocol', 'session', 'start_s']).reset_index(drop=True)
    D = np.full((len(W), len(LEVEL)), np.nan); X = W[LEVEL].values
    for _, g in W.groupby(['protocol', 'session'], sort=False):
        idx = g.index.values; t = g.start_s.values
        for j, i in enumerate(idx):
            m = idx[(t < t[j]) & (t >= t[j] - 120)]
            if len(m): D[i] = X[i] - np.nanmean(X[m], axis=0)
    for j, c in enumerate(LEVEL): W['d_' + c] = D[:, j]
    CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
    return W, {'raw': PHYS + ACC, 'cal': CAL + ACC, 'cal_ctx': CAL + DEL + ACC}
