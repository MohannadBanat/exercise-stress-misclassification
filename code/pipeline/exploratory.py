"""EXPLORATORY (not registered): best participant-independent models with resources at hand.
Every candidate pipeline is deployable on a new person: (i) calibration uses only that person's
3-min resting baseline from the stress session; (ii) context features use only past windows (causal);
(iii) probability smoothing uses only past windows. The pipeline is chosen INSIDE each training fold
(grouped 5-fold CV on training participants); the held-out participant never influences the choice.
Usage: python -I exploratory.py <features_dir> <out_dir> --osf URL"""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('feat'); ap.add_argument('out'); ap.add_argument('--osf', required=True)
ap.add_argument('--only', default='stress_rest,four,deploy'); ap.add_argument('--jobs', type=int, default=2)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
MASTER = 20261008; N_BOOT = 10000
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

EDA = ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']
BVP = ['bvp_pr_mean','bvp_pr_sd','bvp_sdnn','bvp_rmssd','bvp_amp_sd','bvp_coverage']
HR = ['hr_mean']; TEMP = ['temp_mean','temp_sd','temp_slope']
ACC = ['acc_mag_mean','acc_mag_sd','acc_x_mean','acc_x_sd','acc_y_mean','acc_y_sd','acc_z_mean','acc_z_sd','acc_dom_freq','acc_pow_frac_0p5_2']
PHYS = EDA + BVP + HR + TEMP; ALL = PHYS + ACC
LEVEL = ['eda_tonic_mean','eda_tonic_min','eda_tonic_max','bvp_pr_mean','hr_mean','temp_mean','eda_phasic_area','eda_scr_count']

W = pd.read_csv(os.path.join(A.feat, 'windows.csv'))
for c in ['y_stress_rest', 'y_aer_anaer', 'y_four']: W[c] = W[c].fillna('')
W = W.sort_values(['protocol', 'session', 'start_s']).reset_index(drop=True)

# (i) personal calibration: subtract the participant's STRESS-session baseline mean (label-free at deployment: a 3-min rest at set-up)
base = W[(W.protocol == 'STRESS') & (W.block == 'baseline')].groupby('participant')[PHYS].mean()
for c in PHYS: W['cal_' + c] = W[c] - W.participant.map(base[c])
CAL = ['cal_' + c for c in PHYS]
# (ii) causal context: difference from the mean of the same session's windows that started in the previous 120 s
def causal_delta(df, cols, horizon=120.0):
    out = np.full((len(df), len(cols)), np.nan); X = df[cols].values; t = df.start_s.values
    for i in range(len(df)):
        m = (t < t[i]) & (t >= t[i] - horizon)
        if m.any(): out[i] = X[i] - np.nanmean(X[m], axis=0)
    return out
D = []
for _, g in W.groupby(['protocol', 'session'], sort=False):
    D.append(pd.DataFrame(causal_delta(g, LEVEL), index=g.index, columns=['d_' + c for c in LEVEL]))
W = W.join(pd.concat(D)); DEL = ['d_' + c for c in LEVEL]

FEATSETS = {'raw': ALL, 'cal': CAL + ACC, 'cal_ctx': CAL + DEL + ACC}
FEATSETS_PHYS = {'raw': PHYS, 'cal': CAL, 'cal_ctx': CAL + DEL}
def model(kind):
    if kind == 'xgb': return XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, verbosity=0, random_state=MASTER)
    if kind == 'lr': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000))
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced', n_jobs=1, random_state=MASTER))
def cw(y):
    u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}; return np.array([w[v] for v in y])
def fit(kind, X, y):
    m = model(kind)
    if kind == 'xgb': m.fit(X, y, sample_weight=cw(y))
    else: m.fit(X, y)
    return m
def smooth(df_idx, P, k):
    """causal smoothing: mean of the current and previous k-1 windows of the same session (by start time)."""
    if k == 1: return P
    S = P.copy(); d = W.loc[df_idx, ['session', 'protocol', 'start_s']].reset_index(drop=True)
    for _, g in d.groupby(['protocol', 'session']):
        ii = g.sort_values('start_s').index.values; t = g.loc[ii, 'start_s'].values
        for j, i in enumerate(ii):
            m = ii[(t <= t[j]) & (t > t[j] - 15 * k)]; S[i] = P[m].mean(0)
    return S
CONFIGS = [(fs, mk, k) for fs in ('raw', 'cal', 'cal_ctx') for mk in ('xgb', 'lr', 'et') for k in (1, 3)]

def choose(d, y, g, sets):
    """inner grouped 5-fold CV over pipelines; returns best (featureset, model, smoothing)."""
    folds = list(GroupKFold(5).split(d, y, g)); scores = {}
    for fs in ('raw', 'cal', 'cal_ctx'):
        X = d[sets[fs]].values
        for mk in ('xgb', 'lr', 'et'):
            P = np.zeros((len(y), len(np.unique(y))))
            for tr, te in folds: P[te] = fit(mk, X[tr], y[tr]).predict_proba(X[te])
            for k in (1, 3):
                Ps = smooth(d.index.values, P, k); scores[(fs, mk, k)] = balanced_accuracy_score(y, Ps.argmax(1))
    return max(scores, key=scores.get), scores

def run_task(name, d, ycol, sets):
    cls = sorted(d[ycol].unique()); y = np.array([cls.index(v) for v in d[ycol]]); g = d.participant.values
    def one(pid):
        tr = g != pid; te = ~tr
        (fs, mk, k), sc = choose(d[tr], y[tr], g[tr], sets)
        m = fit(mk, d[tr][sets[fs]].values, y[tr]); P = m.predict_proba(d[te][sets[fs]].values)
        Pall = np.zeros((len(y), len(cls))); Pall[te] = P
        P = smooth(d.index.values, Pall, k)[te]
        # registered-style reference: raw features, XGBoost, no smoothing
        Pr = fit('xgb', d[tr][sets['raw']].values, y[tr]).predict_proba(d[te][sets['raw']].values)
        return pid, P, Pr, (fs, mk, k)
    res = Parallel(n_jobs=A.jobs)(delayed(one)(pid) for pid in np.unique(g))
    P = np.zeros((len(y), len(cls))); Pr = np.zeros_like(P); chosen = {}
    for pid, p, pr, c in res: P[g == pid] = p; Pr[g == pid] = pr; chosen[pid] = '|'.join(map(str, c))
    pred, predr = P.argmax(1), Pr.argmax(1)
    def ppba(pr):
        return pd.Series({pid: np.mean([np.mean(pr[(g == pid) & (y == c)] == c) for c in np.unique(y[g == pid])]) for pid in np.unique(g)})
    a, b = ppba(pred), ppba(predr); dlt = (a - b).values
    rng = np.random.default_rng(MASTER + 7); bm = dlt[rng.integers(0, len(dlt), (N_BOOT, len(dlt)))].mean(1)
    rb = a.values[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1)
    out = dict(task=name, n_participants=int(len(a)), n_windows=int(len(y)), classes=cls,
               improved=dict(pooled_ba=balanced_accuracy_score(y, pred), macro_f1=f1_score(y, pred, average='macro'), mean_participant_ba=float(a.mean()),
                             ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]),
               reference=dict(pooled_ba=balanced_accuracy_score(y, predr), macro_f1=f1_score(y, predr, average='macro'), mean_participant_ba=float(b.mean())),
               gain=dict(estimate=float(dlt.mean()), ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]),
               chosen_pipelines=pd.Series(chosen).value_counts().to_dict())
    if len(cls) == 2:
        out['improved']['auroc'] = roc_auc_score(y, P[:, 1]); out['reference']['auroc'] = roc_auc_score(y, Pr[:, 1])
    pd.DataFrame({'ba_improved': a, 'ba_reference': b, 'chosen': pd.Series(chosen)}).to_csv(os.path.join(A.out, f'X_{name}_perparticipant.csv'))
    return out

R = {}
rp = os.path.join(A.out, 'exploratory.json')
if os.path.exists(rp): R = json.load(open(rp))
if 'stress_rest' in A.only:
    log('stress_rest'); R['stress_rest'] = run_task('stress_rest', W[W.y_stress_rest != ''], 'y_stress_rest', FEATSETS)
    json.dump(R, open(rp, 'w'), indent=1, default=float); log(json.dumps(R['stress_rest'], default=float)[:600])
if 'four' in A.only:
    log('four'); R['four'] = run_task('four', W[W.y_four != ''], 'y_four', FEATSETS)
    json.dump(R, open(rp, 'w'), indent=1, default=float); log(json.dumps(R['four'], default=float)[:600])
if 'deploy' in A.only:
    # exercise-robust stress detector: stress vs everything else (rest + all exercise phases), physiology + motion,
    # pipeline chosen inside training folds; report stress sensitivity and exercise false-alarm rate at the E6 threshold rule
    log('deploy')
    d = W[(W.y_stress_rest != '') | ((W.protocol != 'STRESS') & (W.block != 'baseline'))].copy()
    d['y_det'] = np.where(d.y_stress_rest == 'stress', 'stress', 'other')
    cls = ['other', 'stress']; y = (d.y_det == 'stress').astype(int).values; g = d.participant.values
    test_parts = sorted(set(d[d.protocol != 'STRESS'].participant))
    def one(pid):
        tr = g != pid; te = ~tr
        (fs, mk, k), _ = choose(d[tr], y[tr], g[tr], FEATSETS)
        X = d[FEATSETS[fs]].values
        oof = np.zeros(tr.sum()); ytr = y[tr]; Xtr = X[tr]
        for a, b in GroupKFold(5).split(Xtr, ytr, g[tr]): oof[b] = fit(mk, Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        oofs = smooth(d.index.values[tr], np.c_[1 - oof, oof], k)[:, 1]
        tau = float(np.quantile(oofs[ytr == 1], 0.20, method='lower'))
        m = fit(mk, Xtr, ytr); Pall = np.zeros((len(y), 2)); Pall[te] = m.predict_proba(X[te]); p = smooth(d.index.values, Pall, k)[te][:, 1]
        dd = d[te]; ex = (dd.protocol != 'STRESS').values; st = (dd.y_det == 'stress').values; rs = (dd.y_stress_rest == 'rest').values
        return dict(participant=pid, pipeline=f'{fs}|{mk}|{k}', tau=tau, sens=float(np.mean(p[st] >= tau)), far_ex=float(np.mean(p[ex] >= tau)),
                    far_rest=float(np.mean(p[rs] >= tau)), auroc=float(roc_auc_score(st[st | rs], p[st | rs])) if st.any() and rs.any() else np.nan)
    res = pd.DataFrame(Parallel(n_jobs=A.jobs)(delayed(one)(pid) for pid in test_parts)); res.to_csv(os.path.join(A.out, 'X_deploy_perparticipant.csv'), index=False)
    rng = np.random.default_rng(MASTER + 8)
    def ci(v): v = v.dropna().values; b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    R['deploy'] = dict(n=int(len(res)), sens=float(res.sens.mean()), sens_ci=ci(res.sens), far_ex=float(res.far_ex.mean()), far_ex_ci=ci(res.far_ex),
                       far_rest=float(res.far_rest.mean()), far_rest_ci=ci(res.far_rest), auroc_stress_vs_rest=float(res.auroc.mean()),
                       chosen_pipelines=res.pipeline.value_counts().to_dict())
    json.dump(R, open(rp, 'w'), indent=1, default=float); log(json.dumps(R['deploy'], default=float))
log('done')
