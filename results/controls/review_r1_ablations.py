"""POST HOC analyses requested by Reviewer 1 (Sensors, October 2026). Not pre-specified; reported as post hoc.
R1.6  Ablations of the exercise-robust detector (calibrated + context features, extremely randomized trees, stress vs all
      other windows incl. exercise, 45 s smoothing, threshold at 0.80 training out-of-fold sensitivity):
      full (physiology + accelerometer), physiology-only (accelerometer features removed), motion-only (accelerometer only).
      (a) grouped 5-fold CV on all 36 participants; (b) sealed test: trained on the 24 development participants, applied to
      the 12 sealed participants exactly as in M4; (c) WESAD: trained on all 36, applied to the 15 WESAD participants.
      For WESAD, the pre-specified selected variant (+ 5-min adaptive differences) is ablated in the same way.
R1.4  Does the 5-min adaptive reference suppress strong arousal? Stress-window detection of the reference and the
      adaptive-reference detectors by tertile of response magnitude (calibrated tonic EDA; calibrated pulse rate),
      grouped 5-fold CV on all 36, each at its own 0.80-sensitivity threshold (superseded in the manuscript by
      review_r1_magnitude.py, which bins on an absolute scale shared with WESAD; kept for transparency).
Intervals: participant bootstrap (10,000 resamples, seed 20261009); sealed-test rates reproduce the bootstrap draws of
framework2.summarize used by run_m4.py (unrounded), so the full-feature sealed row reproduces the sealed test.
Usage (from the work directory made by run_all.sh, or a copy of code/ with the data in place): python improve/review_r1_ablations.py"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, eval_candidate, LOCK, log
from build_features import build
from features_v1 import PHYS, LEVEL, ACC
from run_m6 import add_adaptive
from sklearn.metrics import roc_auc_score
MASTER = 20261009
def bci(v, B=10000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; rng = np.random.default_rng(MASTER)
    b = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return [round(float(v.mean()), 6), round(float(np.percentile(b, 2.5)), 6), round(float(np.percentile(b, 97.5)), 6), int(len(v))]
W, FS, ACCF = build(); AD5 = add_adaptive(W, PHYS, 5)
CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
FULL = FS['v1_cal_ctx']; assert set(FULL) == set(CAL + DEL + ACC)
VAR = {'full': CAL + DEL + ACC, 'physiology_only': CAL + DEL, 'motion_only': list(ACC)}
d = add_flags(detector_frame(W, final=True)); prev = {45: prev_index(d, 45)}; out = {}
def summarize_precise(res, seed=MASTER):
    """framework2.summarize (as used by run_m4.py) without rounding: same keys, order and random draws."""
    rng = np.random.default_rng(seed); o = {}
    for k in ['bds', 'sens', 'spec', 'far_rest', 'far_rest_all', 'far_ex', 'far_base', 'auroc']:
        if k in res:
            v = res[k].dropna().values; b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
            o[k] = [round(float(v.mean()), 6), round(float(np.percentile(b, 2.5)), 6), round(float(np.percentile(b, 97.5)), 6)]
    return o
def rates(g_kind, alarm, pids):
    df = pd.DataFrame(dict(k=g_kind, a=alarm, p=pids))
    f = lambda k: bci(df[df.k == k].groupby('p').a.mean())
    return dict(stress_sens=f('stress'), rest_far=f('rest'), exercise_far=f('exercise'))
# (a) grouped 5-fold CV on all 36
for name, cols in VAR.items():
    po = smooth(oof(d, cols, ACC, 'et', 'single', 'all'), prev, d.index.values, 45); tau = threshold(po[d.y.values == 1])
    r = rates(d.kind.values, (po >= tau).astype(float), d.participant.values)
    au = [roc_auc_score(g.y, po[np.searchsorted(d.index.values, g.index.values)]) for _, g in d[d.protocol == 'STRESS'].groupby('participant') if g.y.nunique() == 2]
    r['within_session_auroc'] = bci(au); out[f'cv36_{name}'] = r; log('cv36', name, r)
# (b) sealed test: dev 24 -> sealed 12, as in M4
dev = d[~d.participant.isin(LOCK)]; box = d[d.participant.isin(LOCK)]; assert box.participant.nunique() == 12
for name, cols in VAR.items():
    tau = threshold(smooth(oof(dev, cols, ACC, 'et', 'single', 'all'), prev, dev.index.values, 45)[dev.y.values == 1])
    p = train_predict(dev, box, cols, ACC, 'et', 'single', 'all'); r, ps = eval_candidate(box, p, prev, 45, 'global', tau=tau)
    pos = pd.Series(np.arange(len(box)), index=box.index.values)
    au = [roc_auc_score(g.y.values, ps[pos[g.index].values]) for _, g in box[box.protocol == 'STRESS'].groupby('participant') if g.y.nunique() == 2]
    s = summarize_precise(r)   # same bootstrap sequence as framework2.summarize in run_m4.py, without rounding
    out[f'sealed_{name}'] = dict(stress_sens=s['sens'], rest_far=s['far_rest'], exercise_far=s['far_ex'], baseline_far=s['far_base'],
                                 n=int(len(r)), n_exercise=int(r.far_ex.notna().sum()), within_session_auroc=bci(au))
    log('sealed', name, out[f'sealed_{name}'])
# (c) WESAD (frame built exactly as in run_wesad.py)
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
WV = {'reference_full': CAL + DEL + ACC, 'reference_physiology_only': CAL + DEL, 'motion_only': list(ACC),
      'selected_full': CAL + DEL + AD5 + ACC, 'selected_physiology_only': CAL + DEL + AD5}
for name, cols in WV.items():
    tau = threshold(smooth(oof(d, cols, ACC, 'et', 'single', 'all'), prev, d.index.values, 45)[d.y.values == 1])
    p = smooth(train_predict(d, Wt, cols, ACC, 'et', 'single', 'all'), prevt, Wt.index.values, 45); a = (p >= tau).astype(float)
    au = [roc_auc_score(g.y, p[g.index.values]) for _, g in Wt.groupby('participant')]
    tm = pd.Series(a[Wt.y.values == 1]).groupby(Wt.participant.values[Wt.y.values == 1]).mean()
    rm = pd.Series(a[Wt.y.values == 0]).groupby(Wt.participant.values[Wt.y.values == 0]).mean()
    out[f'wesad_{name}'] = dict(auroc=bci(au), tsst_alarm=bci(tm), rest_alarm=bci(rm)); log('wesad', name, out[f'wesad_{name}'])
# R1.4: detection by response magnitude, reference vs adaptive reference, grouped 5-fold CV on all 36
S = d.kind.values == 'stress'
for name, cols in (('reference', CAL + DEL + ACC), ('adaptive_reference', CAL + DEL + AD5 + ACC)):
    po = smooth(oof(d, cols, ACC, 'et', 'single', 'all'), prev, d.index.values, 45); tau = threshold(po[d.y.values == 1])
    for feat in ('cal_eda_tonic_mean', 'cal_bvp_pr_mean'):
        v = d[feat].values; ok = S & ~np.isnan(v); q = np.quantile(v[ok], [1 / 3, 2 / 3])
        terc = np.digitize(v, q)
        out[f'tertile_{name}_{feat}'] = {f'T{k + 1}': dict(detect=round(float((po[ok & (terc == k)] >= tau).mean()), 6), n=int((ok & (terc == k)).sum()),
                                          range=[round(float(v[ok & (terc == k)].min()), 2), round(float(v[ok & (terc == k)].max()), 2)]) for k in range(3)}
        log('tertile', name, feat, out[f'tertile_{name}_{feat}'])
json.dump(out, open('improve/review/sum_r1_ablations.json', 'w'), indent=1); log('saved')
