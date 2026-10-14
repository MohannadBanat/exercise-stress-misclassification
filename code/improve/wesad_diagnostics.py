"""POST HOC diagnostics after the one-time WESAD test (not part of the pre-specification; reported as post hoc).
D2 alarm rate in TSST windows by motion (still: ACC magnitude SD < 0.02 g, the registered stillness rule).
D3 a physiology-only detector like the registered E6 detector (XGBoost, raw physiological features, trained on the
   stress-session windows of all 36 participants, threshold at 0.80 training out-of-fold sensitivity), applied to WESAD.
D4 the final pipeline without accelerometer features (calibrated + context + 5-min adaptive physiological features,
   extremely randomized trees, stress vs all other windows incl. exercise, 45 s smoothing), applied to WESAD and
   evaluated on the training data by grouped 5-fold CV (exercise false alarms).
D5 the final pipeline with ACC but without adaptive features is the pre-specified reference (already reported)."""
import sys, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from run_m6 import add_adaptive, alarm_states
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, fit, log
from build_features import build
from features_v1 import PHYS, LEVEL, ACC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
MASTER = 20261009; rng = np.random.default_rng(MASTER)
def bci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))]
R = pickle.load(open('improve/wesad_all.pkl', 'rb'))
W, FS, ACCF = build(); AD5 = add_adaptive(W, PHYS, 5); CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
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
prevt = {45: prev_index(Wt, 45)}; out = {}
def wesad_eval(p, tau, name):
    Wt['_p'] = p; Wt['_a'] = (p >= tau).astype(int)
    au = [roc_auc_score(g.y, g._p) for _, g in Wt.groupby('participant')]
    st = Wt.acc_mag_sd < 0.02
    r = dict(auroc=bci(au), tsst_alarm=bci(Wt[Wt.y == 1].groupby('participant')._a.mean()), rest_alarm=bci(Wt[Wt.y == 0].groupby('participant')._a.mean()),
             tsst_alarm_still=bci(Wt[(Wt.y == 1) & st].groupby('participant')._a.mean()), tsst_alarm_moving=bci(Wt[(Wt.y == 1) & ~st].groupby('participant')._a.mean()),
             share_tsst_moving=float((~st[Wt.y == 1]).mean()), share_rest_moving=float((~st[Wt.y == 0]).mean()))
    out[name] = r; log(name, {k: (round(v[0], 3) if isinstance(v, list) else round(v, 3)) for k, v in r.items()})
# D2: the two pre-specified detectors, re-scored by motion (same models refitted deterministically)
for name, (fs, logic) in (('selected_refad5', ('refad5', 'state')), ('reference_ref', ('ref', 'state'))):
    cols = (CAL + DEL + AD5 + ACC) if fs == 'refad5' else (CAL + DEL + ACC)
    p = smooth(train_predict(d, Wt, cols, ACCF, 'et', 'single', 'all'), prevt, Wt.index.values, 45)
    wesad_eval(p, R['tau'][fs], 'D2_' + name)
# D3: E6-style physiology-only detector (XGBoost, nested grid as registered, stress-session windows only)
GRID = [dict(max_depth=a, n_estimators=n, learning_rate=l) for a in (2, 3) for n in (100, 300) for l in (0.05, 0.1)]
def mk(p): return XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=2, verbosity=0, random_state=20261008)
def cw(y):
    u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}; return np.array([w[v] for v in y])
S = d[d.protocol == 'STRESS']; Xs, ys, gs = S[PHYS].values, S.y.values, S.participant.values; folds = list(GroupKFold(5).split(Xs, ys, gs)); best, bs = None, -1
for p in GRID:
    pr = np.empty(len(ys), int)
    for a, b in folds: m = mk(p); m.fit(Xs[a], ys[a], sample_weight=cw(ys[a])); pr[b] = m.predict(Xs[b])
    s_ = balanced_accuracy_score(ys, pr)
    if s_ > bs: best, bs = p, s_
oo = np.zeros(len(ys))
for a, b in folds: m = mk(best); m.fit(Xs[a], ys[a], sample_weight=cw(ys[a])); oo[b] = m.predict_proba(Xs[b])[:, 1]
tau3 = float(np.quantile(oo[ys == 1], 0.20, method='lower')); m = mk(best); m.fit(Xs, ys, sample_weight=cw(ys))
wesad_eval(m.predict_proba(Wt[PHYS].values)[:, 1], tau3, 'D3_physiology_only_E6_style')
# D4: final pipeline without ACC features
cols4 = CAL + DEL + AD5
po = smooth(oof(d, cols4, ACCF, 'et', 'single', 'all'), prev, d.index.values, 45); tau4 = threshold(po[d.y.values == 1], sens=0.80)
dd = d.assign(_a=(po >= tau4).astype(int)); out['D4_training_cv'] = dict(exercise_alarm=bci(dd[dd.kind == 'exercise'].groupby('participant')._a.mean()),
    rest_alarm=bci(dd[dd.kind == 'rest'].groupby('participant')._a.mean()), stress_alarm=bci(dd[dd.kind == 'stress'].groupby('participant')._a.mean()))
po5 = smooth(oof(d, CAL + DEL + AD5 + ACC, ACCF, 'et', 'single', 'all'), prev, d.index.values, 45); tau5 = threshold(po5[d.y.values == 1], sens=0.80)
d5 = d.assign(_a=(po5 >= tau5).astype(int)); out['D4_training_cv_with_acc'] = dict(exercise_alarm=bci(d5[d5.kind == 'exercise'].groupby('participant')._a.mean()),
    rest_alarm=bci(d5[d5.kind == 'rest'].groupby('participant')._a.mean()), stress_alarm=bci(d5[d5.kind == 'stress'].groupby('participant')._a.mean()))
log('D4 training CV without ACC', out['D4_training_cv']); log('D4 training CV with ACC', out['D4_training_cv_with_acc'])
p4 = smooth(train_predict(d, Wt, cols4, ACCF, 'et', 'single', 'all'), prevt, Wt.index.values, 45)
wesad_eval(p4, tau4, 'D4_no_accelerometer')
json.dump(out, open('improve/sum_WESAD_diagnostics.json', 'w'), indent=1); log('saved')
