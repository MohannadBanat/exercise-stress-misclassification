"""POST HOC analysis requested by Reviewer 1, point 4 (Sensors, October 2026). Not pre-specified; reported as post hoc.
Does the 5-min adaptive reference suppress strong arousal? Detection of stress windows by the reference detector and the
adaptive-reference detector (pre-specified selected M6 variant), binned on a common absolute scale of the calibrated
response (pulse-rate rise from the person's baseline, bpm; tonic EDA rise, uS): (a) grouped 5-fold CV on all 36 training
participants (stress windows; exercise windows shown for comparison); (b) WESAD TSST windows, detectors trained on all 36.
Each detector at its own threshold (0.80 training out-of-fold sensitivity), 45 s smoothing, as in M6 and the WESAD test.
Usage (from the work directory made by run_all.sh, or a copy of code/ with the data in place): python improve/review_r1_magnitude.py"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, log
from build_features import build
from features_v1 import PHYS, LEVEL, ACC
from run_m6 import add_adaptive
W, FS, _ = build(); AD5 = add_adaptive(W, PHYS, 5)
CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
d = add_flags(detector_frame(W, final=True)); prev = {45: prev_index(d, 45)}
Wt = pd.read_csv('features_wesad/windows_wesad.csv').sort_values(['session', 'start_s']).reset_index(drop=True)
base = Wt[Wt.block == 'baseline'].groupby('participant')[PHYS].mean()
for c in PHYS: Wt['cal_' + c] = Wt[c] - Wt.participant.map(base[c])
X = Wt[LEVEL].values; D = np.full(X.shape, np.nan)
for _, g in Wt.groupby(['protocol', 'session'], sort=False):
    idx = g.index.values; t = g.start_s.values
    for j, i in enumerate(idx):
        m = idx[(t < t[j]) & (t >= t[j] - 120)]
        if len(m): D[i] = X[i] - np.nanmean(X[m], axis=0)
for j, c in enumerate(LEVEL): Wt['d_' + c] = D[:, j]
add_adaptive(Wt, PHYS, 5); Wt['y'] = (Wt.block == 'tsst').astype(int); Wt['kind'] = np.where(Wt.y == 1, 'stress', 'rest'); Wt['rest_early'] = 0
prevt = {45: prev_index(Wt, 45)}
BINS = {'cal_bvp_pr_mean': [-np.inf, 10, 20, np.inf], 'cal_eda_tonic_mean': [-np.inf, 1, 3, np.inf]}
LAB = {'cal_bvp_pr_mean': ['<10 bpm', '10-20 bpm', '>=20 bpm'], 'cal_eda_tonic_mean': ['<1 uS', '1-3 uS', '>=3 uS']}
def table(frame, alarm, mask):
    out = {}
    for f, b in BINS.items():
        cut = pd.cut(frame[f].values, b, labels=LAB[f], right=False); r = {}
        for lab in LAB[f]:
            m = mask & (cut == lab)
            r[lab] = dict(alarm=round(float(alarm[m].mean()), 6) if m.any() else None, n_windows=int(m.sum()),
                          n_participants=len(set(frame.participant.values[m])))
        out[f] = r
    return out
out = {}
for name, cols in (('reference', CAL + DEL + ACC), ('adaptive_reference', CAL + DEL + AD5 + ACC)):
    po = smooth(oof(d, cols, ACC, 'et', 'single', 'all'), prev, d.index.values, 45); tau = threshold(po[d.y.values == 1])
    a = (po >= tau).astype(float)
    out[f'train_cv36_stress_{name}'] = table(d, a, d.kind.values == 'stress')
    out[f'train_cv36_exercise_{name}'] = table(d, a, d.kind.values == 'exercise')
    pt = smooth(train_predict(d, Wt, cols, ACC, 'et', 'single', 'all'), prevt, Wt.index.values, 45)
    out[f'wesad_tsst_{name}'] = table(Wt, (pt >= tau).astype(float), Wt.y.values == 1)
    for k in ('train_cv36_stress', 'train_cv36_exercise', 'wesad_tsst'): log(k, name, out[f'{k}_{name}'])
json.dump(out, open('improve/review/sum_r1_magnitude.json', 'w'), indent=1); log('saved')
