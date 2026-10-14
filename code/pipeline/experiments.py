"""Experiments E1-E6 of the registered analysis plan (AbuOwida_Banat_AnalysisPlan_v4).
SAFETY: real labels are used only with --osf <registration URL>. Without it, every class label is
randomly permuted (--null mode), which tests the code and must give chance-level results.
Usage: python -I experiments.py <features_dir> <out_dir> [--osf URL] [--only E1,E2,...] [--quick]"""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, wilcoxon
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser(); ap.add_argument('feat'); ap.add_argument('out')
ap.add_argument('--osf', default=None); ap.add_argument('--only', default='E1,E2,E3,E4,E5,E6')
ap.add_argument('--quick', action='store_true', help='reduced repeats for code testing only')
ap.add_argument('--jobs', type=int, default=2)
ap.add_argument('--smoke-parts', type=int, default=0, help='code test only: limit E6 to N test participants')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
NULL = A.osf is None
MASTER = 20261008
rng0 = np.random.default_rng(MASTER)
N_BOOT = 2000 if A.quick else 10000
N_PERM = 200 if A.quick else 10000
E1_SEEDS = 5 if A.quick else (25 if NULL else 100)   # shuffled-label control: 25 seeds (Table A3)

EDA = ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']
BVP = ['bvp_pr_mean','bvp_pr_sd','bvp_sdnn','bvp_rmssd','bvp_amp_sd','bvp_coverage']
HR = ['hr_mean']; TEMP = ['temp_mean','temp_sd','temp_slope']
ACC = ['acc_mag_mean','acc_mag_sd','acc_x_mean','acc_x_sd','acc_y_mean','acc_y_sd','acc_z_mean','acc_z_sd','acc_dom_freq','acc_pow_frac_0p5_2']
PHYS = EDA + BVP + HR + TEMP; ALL = PHYS + ACC
assert len(ALL) == 29

XGB_GRID = [dict(max_depth=d, n_estimators=n, learning_rate=l) for d in (2, 3) for n in (100, 300) for l in (0.05, 0.1)]
LR_GRID = [dict(C=c) for c in (0.01, 0.1, 1, 10)]

def make(kind, p):
    if kind == 'xgb':
        return XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, verbosity=0, random_state=MASTER)
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=p['C'], class_weight='balanced', max_iter=2000))

def cw(y):
    u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}
    return np.array([w[v] for v in y])

def fit(kind, p, X, y):
    m = make(kind, p)
    if kind == 'xgb': m.fit(X, y, sample_weight=cw(y))
    else: m.fit(X, y)
    return m

def select(kind, X, y, g, n_folds=5):
    """Inner grouped CV on training participants only; returns best params by balanced accuracy."""
    grid = XGB_GRID if kind == 'xgb' else LR_GRID
    if len(np.unique(g)) < n_folds: n_folds = len(np.unique(g))
    folds = list(GroupKFold(n_folds).split(X, y, g)); best, bs = None, -1
    for p in grid:
        pred = np.empty(len(y), dtype=y.dtype)
        for tr, te in folds:
            if len(np.unique(y[tr])) < len(np.unique(y)): pred[te] = np.bincount(y[tr]).argmax(); continue
            pred[te] = fit(kind, p, X[tr], y[tr]).predict(X[te])
        s = balanced_accuracy_score(y, pred)
        if s > bs: best, bs = p, s
    return best

def lopo(kind, X, y, g, return_proba=False, jobs=A.jobs):
    """Nested leave-one-participant-out: returns out-of-fold predictions (and probabilities)."""
    parts = np.unique(g)
    def one(pid):
        te = g == pid; tr = ~te
        p = select(kind, X[tr], y[tr], g[tr]); m = fit(kind, p, X[tr], y[tr])
        return pid, m.predict(X[te]), m.predict_proba(X[te]), p
    res = Parallel(n_jobs=jobs)(delayed(one)(pid) for pid in parts)
    pred = np.empty(len(y), dtype=y.dtype); proba = np.zeros((len(y), len(np.unique(y)))); params = {}
    for pid, pr, pb, p in res:
        pred[g == pid] = pr; proba[g == pid] = pb; params[pid] = p
    return (pred, proba, params) if return_proba else pred

def pp_ba(y, pred, g):
    """Per-participant balanced accuracy (mean recall over the classes the participant has)."""
    out = {}
    for pid in np.unique(g):
        m = g == pid; yy, pp = y[m], pred[m]
        out[pid] = np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])
    return pd.Series(out)

def boot_mean(d, rng, n=N_BOOT):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), (n, len(d))); return d[idx].mean(1)

def summarize_diff(d, rng, null_value=0.0, direction='greater'):
    """Mean of per-participant differences, 95% percentile CI, one-sided bootstrap p for H1: mean > null (or < null)."""
    bm = boot_mean(d, rng); est = float(np.nanmean(d))
    p = float(np.mean(bm <= null_value)) if direction == 'greater' else float(np.mean(bm >= null_value))
    p = max(p, 1.0 / len(bm))
    try: w = float(wilcoxon(np.asarray(d)[~np.isnan(d)] - null_value, alternative=direction).pvalue)
    except Exception: w = np.nan
    return dict(estimate=est, ci_low=float(np.percentile(bm, 2.5)), ci_high=float(np.percentile(bm, 97.5)), p=p, p_wilcoxon=w, n=int(np.sum(~np.isnan(d))))

def null_labels(y, rng):
    return rng.permutation(y) if NULL else y

def enc(y):
    cls = sorted(np.unique(y)); return np.array([cls.index(v) for v in y]), cls

# ------------------------------------------------------------------ data
W = pd.read_csv(os.path.join(A.feat, 'windows.csv'), keep_default_na=True)
B = pd.read_csv(os.path.join(A.feat, 'e1_blocks.csv'))
for c in ['y_stress_rest', 'y_aer_anaer', 'y_four']: W[c] = W[c].fillna('')
R = {'mode': 'NULL (labels permuted; code test only)' if NULL else 'REAL', 'osf': A.osf, 'quick': A.quick,
     'n_boot': N_BOOT, 'n_perm': N_PERM, 'e1_seeds': E1_SEEDS}
TESTS = {}
_prev = os.path.join(A.out, 'results.json')
if os.path.exists(_prev):
    _old = json.load(open(_prev)); TESTS.update(_old.pop('confirmatory', {}))
    for k, v in _old.items():
        if k not in R: R[k] = v
    assert _old.get('mode') == R['mode'], 'refusing to mix NULL and REAL results'
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def save(): json.dump(dict(R, confirmatory=TESTS), open(os.path.join(A.out, 'results.json'), 'w'), indent=1, default=float)

TASKS_E1 = {'stress_rest': ['stress', 'rest'], 'aer_anaer': ['aerobic', 'anaerobic'], 'four': ['rest', 'stress', 'aerobic', 'anaerobic']}
B1 = B[~((B.protocol == 'STRESS') & B.participant.isin(['f07', 'f13']))].copy()   # exclusions of the original study

def corr_filter(X, thr=0.8):
    keep = []
    C = pd.DataFrame(X).corr().abs().values
    for j in range(X.shape[1]):
        if all(not (C[j, k] > thr) for k in keep): keep.append(j)
    return keep

def oversample(X, y, rng):
    u, c = np.unique(y, return_counts=True); idx = []
    for k, n in zip(u, c):
        ii = np.where(y == k)[0]; idx += list(ii) + list(rng.choice(ii, c.max() - n, replace=True))
    idx = np.array(idx); return X[idx], y[idx]

# ------------------------------------------------------------------ E1 reproduction
def e1_run(task, variant, seed, kind='xgb'):
    rng = np.random.default_rng(MASTER + seed)
    d = B1[B1.e1_class.isin(TASKS_E1[task])]
    X = d[ALL].values; y, _ = enc(null_labels(d.e1_class.values, rng))
    if variant == 'leaky':   # filter and oversample on all samples, then split
        keep = corr_filter(X); X = X[:, keep]; X, y = oversample(X, y, rng)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    else:                    # split first; filter and oversample inside training only
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        keep = corr_filter(Xtr); Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    grid = XGB_GRID if kind == 'xgb' else LR_GRID; skf = StratifiedKFold(10, shuffle=True, random_state=seed); best, bs = None, -1
    for p in grid:
        pr = np.empty(len(ytr), dtype=int)
        for a, b in skf.split(Xtr, ytr):
            xa, ya = (Xtr[a], ytr[a]) if variant == 'leaky' else oversample(Xtr[a], ytr[a], rng)
            pr[b] = fit(kind, p, xa, ya).predict(Xtr[b])
        s = np.mean(pr == ytr)
        if s > bs: best, bs = p, s
    xa, ya = (Xtr, ytr) if variant == 'leaky' else oversample(Xtr, ytr, rng)
    pred = fit(kind, best, xa, ya).predict(Xte)
    return dict(task=task, variant=variant, seed=seed, model=kind, accuracy=float(np.mean(pred == yte)), balanced_accuracy=float(balanced_accuracy_score(yte, pred)))

if 'E1' in A.only:
    log('E1 start')
    ck = os.path.join(A.out, 'E1_runs.csv')
    df = pd.read_csv(ck) if os.path.exists(ck) else pd.DataFrame(columns=['task', 'variant', 'seed'])
    todo = [(t, v, s) for t in TASKS_E1 for v in ('leaky', 'clean') for s in range(E1_SEEDS)
            if not ((df.task == t) & (df.variant == v) & (df.seed == s)).any()]
    for i in range(0, len(todo), 20):   # checkpoint every 20 runs
        rows = Parallel(n_jobs=A.jobs)(delayed(e1_run)(*a) for a in todo[i:i + 20])
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True); df.to_csv(ck, index=False)
        log(f'E1 {len(df)}/{len(TASKS_E1) * 2 * E1_SEEDS} runs')
    pub = {'stress_rest': 0.93, 'aer_anaer': 0.91, 'four': 0.84}
    summ = df.groupby(['task', 'variant']).accuracy.agg(['mean', 'std', 'min', 'max']).reset_index()
    summ['published'] = summ.task.map(pub); summ['within_0.05'] = (summ['mean'] - summ.published).abs() <= 0.05
    R['E1'] = summ.to_dict('records')
    R['E1_reproduced'] = bool(any(summ[summ.variant == v]['within_0.05'].all() for v in ('leaky', 'clean')))
    save(); log('E1 done'); print(summ.to_string())

# ------------------------------------------------------------------ E2 participant-level, paired with sample-level CV
if 'E2' in A.only:
    log('E2 start'); R['E2'] = {}
    for h, task in zip(['H1a', 'H1b', 'H1c'], TASKS_E1):
        rng = np.random.default_rng(MASTER + 2)
        d = B1[B1.e1_class.isin(TASKS_E1[task])]
        X = d[ALL].values; g = d.participant.values; y, cls = enc(null_labels(d.e1_class.values, rng))
        res = {}
        for kind in ('xgb', 'lr'):
            p_lopo = lopo(kind, X, y, g)
            skf = StratifiedKFold(10, shuffle=True, random_state=MASTER); p_samp = np.empty(len(y), dtype=int)
            gidx = np.zeros(len(y), int)  # sample-level CV: hyperparameters by inner stratified CV (not grouped), as in the original
            for tr, te in skf.split(X, y):
                best, bs = None, -1
                for p in (XGB_GRID if kind == 'xgb' else LR_GRID):
                    pr = np.empty(len(tr), int)
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=MASTER).split(X[tr], y[tr]):
                        pr[b] = fit(kind, p, X[tr][a], y[tr][a]).predict(X[tr][b])
                    s = balanced_accuracy_score(y[tr], pr)
                    if s > bs: best, bs = p, s
                p_samp[te] = fit(kind, best, X[tr], y[tr]).predict(X[te])
            ba_l, ba_s = pp_ba(y, p_lopo, g), pp_ba(y, p_samp, g)
            diff = (ba_s - ba_l).values
            res[kind] = dict(pooled_ba_lopo=float(balanced_accuracy_score(y, p_lopo)), pooled_ba_sample=float(balanced_accuracy_score(y, p_samp)),
                             macro_f1_lopo=float(f1_score(y, p_lopo, average='macro')), macro_f1_sample=float(f1_score(y, p_samp, average='macro')),
                             paired=summarize_diff(diff, np.random.default_rng(MASTER + 21)))
            pd.DataFrame({'ba_lopo': ba_l, 'ba_sample': ba_s}).to_csv(os.path.join(A.out, f'E2_{task}_{kind}_perparticipant.csv'))
        R['E2'][task] = res; TESTS[h] = dict(res['xgb']['paired'], test='mean(BA_sample - BA_LOPO) > 0')
        save(); log('E2', task, json.dumps(res['xgb']['paired']))

# ------------------------------------------------------------------ E3 shortcut baselines (window level, LOPO)
def win_task(col):
    d = W[W[col] != ''].copy(); return d

if 'E3' in A.only:
    log('E3 start'); R['E3'] = {}
    d = win_task('y_four'); rng = np.random.default_rng(MASTER + 3)
    y, cls = enc(null_labels(d.y_four.values, rng)); g = d.participant.values
    d['cohort'] = (d.version == 'V2').astype(int)
    sets = {'full': ALL, 'acc_only': ACC, 'time_only': ['t_session'], 'block_position': ['block_pos', 'cohort']}
    preds = {}
    for name, cols in sets.items():
        preds[name] = lopo('xgb', d[cols].values.astype(float), y, g)
        R['E3'][f'four_{name}'] = dict(pooled_ba=float(balanced_accuracy_score(y, preds[name])))
        log('E3 four', name, R['E3'][f'four_{name}'])
    pd.DataFrame({k: pp_ba(y, v, g) for k, v in preds.items()}).to_csv(os.path.join(A.out, 'E3_four_perparticipant.csv'))
    ba_full, ba_acc = pp_ba(y, preds['full'], g), pp_ba(y, preds['acc_only'], g)
    TESTS['H2'] = dict(summarize_diff((ba_full - ba_acc).values, np.random.default_rng(MASTER + 31), null_value=0.05, direction='less'),
                       test='non-inferiority: mean(BA_full - BA_ACC) < 0.05')
    # H3: time-only, stress vs rest
    d2 = win_task('y_stress_rest'); y2, _ = enc(null_labels(d2.y_stress_rest.values, rng)); g2 = d2.participant.values
    p2 = lopo('xgb', d2[['t_session']].values.astype(float), y2, g2)
    ba2 = pp_ba(y2, p2, g2); ba2.to_frame('time_only').to_csv(os.path.join(A.out, 'E3_stress_rest_time_perparticipant.csv')); R['E3']['stress_rest_time_only'] = dict(pooled_ba=float(balanced_accuracy_score(y2, p2)))
    TESTS['H3'] = dict(summarize_diff(ba2.values, np.random.default_rng(MASTER + 32), null_value=0.5, direction='greater'), test='mean BA(time-only) > 0.50')
    save(); log('E3 done', TESTS['H2'], TESTS['H3'])

# ------------------------------------------------------------------ E4 pre-task baselines (V2)
if 'E4' in A.only:
    log('E4 start')
    d = W[(W.version == 'V2') & (W.block == 'baseline') & (W.still == 1)].copy()
    ok = d.groupby('participant').protocol.transform(lambda s: s.nunique() >= 2 and 'STRESS' in set(s)); d = d[ok]
    g = d.participant.values; X = d[PHYS].values
    def labels_for(perm_rng=None):
        lab = np.where(d.protocol.values == 'STRESS', 1, 0)
        if perm_rng is not None:   # within-participant permutation of session labels
            lab = lab.copy()
            for pid in np.unique(g):
                m = g == pid; sess = d.protocol.values[m]; us = np.unique(sess); perm = dict(zip(us, perm_rng.permutation(us)))
                lab[m] = np.where(np.array([perm[s] for s in sess]) == 'STRESS', 1, 0)
        return lab
    rng = np.random.default_rng(MASTER + 4)
    y = labels_for(rng if NULL else None)
    pred, proba, params = lopo('xgb', X, y, g, return_proba=True)
    obs = float(np.mean(pp_ba(y, pred, g)))
    # permutation null with the hyperparameters selected in the observed run (fixed per held-out participant)
    def perm_once(i):
        r = np.random.default_rng(MASTER + 4000 + i); yp = labels_for(r); pr = np.empty(len(yp), int)
        for pid in np.unique(g):
            te = g == pid; tr = ~te
            pr[te] = fit('xgb', params[pid], X[tr], yp[tr]).predict(X[te]) if len(np.unique(yp[tr])) > 1 else yp[tr][0]
        return float(np.mean(pp_ba(yp, pr, g)))
    nulls = np.array(Parallel(n_jobs=A.jobs)(delayed(perm_once)(i) for i in range(N_PERM)))
    p = float((1 + np.sum(nulls >= obs)) / (1 + len(nulls)))
    R['E4'] = dict(n_participants=int(len(np.unique(g))), n_windows=int(len(y)), mean_participant_ba=obs, pooled_ba=float(balanced_accuracy_score(y, pred)),
                   null_mean=float(nulls.mean()), null_95=float(np.percentile(nulls, 95)))
    TESTS['H4'] = dict(estimate=obs, p=p, n=int(len(np.unique(g))), test='permutation: BA(pre-task baselines) > null')
    # X1 exploratory control: aerobic vs anaerobic baselines (concurrent groups)
    dx = W[(W.version == 'V2') & (W.block == 'baseline') & (W.still == 1) & (W.protocol != 'STRESS')]
    dx = dx[dx.groupby('participant').protocol.transform('nunique') == 2]
    if len(dx):
        yx = null_labels((dx.protocol == 'AEROBIC').astype(int).values, rng)
        px = lopo('xgb', dx[PHYS].values, yx, dx.participant.values)
        R['E4']['X1_aer_vs_anaer_baseline_ba'] = float(np.mean(pp_ba(yx, px, dx.participant.values))); R['E4']['X1_n'] = int(dx.participant.nunique())
    save(); log('E4 done', R['E4'], TESTS['H4'])

# ------------------------------------------------------------------ E5 cross-cohort transfer (stress vs rest)
if 'E5' in A.only:
    log('E5 start')
    d = win_task('y_stress_rest'); rng = np.random.default_rng(MASTER + 5)
    y, _ = enc(null_labels(d.y_stress_rest.values, rng)); g = d.participant.values; v = d.version.values; X = d[ALL].values
    within = np.empty(len(y), int); cross = np.empty(len(y), int)
    for coh in ('V1', 'V2'):
        m = v == coh
        within[m] = lopo('xgb', X[m], y[m], g[m])
        p = select('xgb', X[~m], y[~m], g[~m]); cross[m] = fit('xgb', p, X[~m], y[~m]).predict(X[m])
    pd.DataFrame({'within': pp_ba(y, within, g), 'cross': pp_ba(y, cross, g), 'cohort': pd.Series(dict(zip(g, v)))}).to_csv(os.path.join(A.out, 'E5_perparticipant.csv'))
    dif = (pp_ba(y, within, g) - pp_ba(y, cross, g)).values
    R['E5'] = dict(pooled_ba_within=float(balanced_accuracy_score(y, within)), pooled_ba_cross=float(balanced_accuracy_score(y, cross)),
                   by_cohort={c: dict(within=float(balanced_accuracy_score(y[v == c], within[v == c])), cross=float(balanced_accuracy_score(y[v == c], cross[v == c]))) for c in ('V1', 'V2')})
    TESTS['H5'] = dict(summarize_diff(dif, np.random.default_rng(MASTER + 51)), test='mean(BA_within - BA_cross) > 0')
    save(); log('E5 done', R['E5'], TESTS['H5'])

# ------------------------------------------------------------------ E6 deployment: stress detector on exercise
if 'E6' in A.only:
    log('E6 start')
    S = W[W.y_stress_rest != ''].copy(); rng = np.random.default_rng(MASTER + 6)
    S['y'] = (null_labels(S.y_stress_rest.values, rng) == 'stress').astype(int)
    EX = W[(W.protocol != 'STRESS') & (W.block != 'baseline')].copy()
    test_parts = sorted(set(EX.participant))
    if A.smoke_parts:
        assert NULL, '--smoke-parts is for null-mode code tests only'; test_parts = test_parts[:A.smoke_parts]
    def detector(pid, aware):
        tr = S[S.participant != pid]; Xtr, ytr, gtr = tr[PHYS].values, tr.y.values, tr.participant.values
        if aware:   # exercise windows of training participants added as non-stress
            ex = EX[EX.participant != pid]
            Xtr = np.vstack([Xtr, ex[PHYS].values]); ytr = np.concatenate([ytr, np.zeros(len(ex), int)]); gtr = np.concatenate([gtr, ex.participant.values])
        p = select('xgb', Xtr, ytr, gtr)
        # threshold: training out-of-fold stress probabilities, sensitivity >= 0.80 (Equation 2)
        oof = np.zeros(len(ytr))
        for a, b in GroupKFold(5).split(Xtr, ytr, gtr): oof[b] = fit('xgb', p, Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        tau = float(np.quantile(oof[ytr == 1], 0.20, method='lower'))
        m = fit('xgb', p, Xtr, ytr)
        te_s = S[S.participant == pid]; te_e = EX[EX.participant == pid]
        ps = m.predict_proba(te_s[PHYS].values)[:, 1] if len(te_s) else np.array([])
        pe = m.predict_proba(te_e[PHYS].values)[:, 1]
        rest = te_s.y.values == 0; st = te_s.y.values == 1
        return dict(participant=pid, aware=aware, tau=tau, far_ex=float(np.mean(pe >= tau)), far_rest=float(np.mean(ps[rest] >= tau)) if rest.any() else np.nan,
                    sens=float(np.mean(ps[st] >= tau)) if st.any() else np.nan, n_ex=int(len(pe)),
                    alarms=list(zip(te_e.index.tolist(), (pe >= tau).astype(int).tolist())))
    res = Parallel(n_jobs=A.jobs)(delayed(detector)(pid, aw) for pid in test_parts for aw in (False, True))
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'alarms'} for r in res]); df.to_csv(os.path.join(A.out, 'E6_perparticipant.csv'), index=False)
    base = df[~df.aware].set_index('participant'); aw = df[df.aware].set_index('participant')
    TESTS['H6'] = dict(summarize_diff((base.far_ex - base.far_rest).values, np.random.default_rng(MASTER + 61)), test='mean(FAR_exercise - FAR_rest) > 0')
    # H7: GEE of alarms on intensity
    import statsmodels.api as sm, statsmodels.formula.api as smf
    al = pd.DataFrame([(i, a) for r in res if not r['aware'] for i, a in r['alarms']], columns=['idx', 'alarm']).set_index('idx')
    G = EX.join(al, how='inner'); G = G[G.intensity.notna()]; G['v2'] = (G.version == 'V2').astype(int)
    form = 'alarm ~ intensity + v2' if G.v2.nunique() > 1 else 'alarm ~ intensity'
    gee = smf.gee(form, groups='participant', data=G, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    z = float(gee.params['intensity'] / gee.bse['intensity'])
    TESTS['H7'] = dict(estimate=float(gee.params['intensity']), ci_low=float(gee.conf_int().loc['intensity', 0]), ci_high=float(gee.conf_int().loc['intensity', 1]),
                       p=float(1 - norm.cdf(z)), n=int(G.participant.nunique()), test='GEE gamma1 > 0 (log-odds per intensity level)')
    R['E6'] = dict(far_by_intensity=G.groupby('intensity').alarm.mean().to_dict(), far_ex_mean=float(base.far_ex.mean()), far_rest_mean=float(base.far_rest.mean()),
                   sens_mean=float(base.sens.mean()), aware_far_ex_mean=float(aw.far_ex.mean()), aware_sens_mean=float(aw.sens.mean()),
                   alarms_per_hour_ex=float(base.far_ex.mean() * 3600 / 15))
    # H8: intersection-union: (a) ratio of mean FAR <= 0.5, (b) sensitivity loss <= 0.05
    pids = np.array(base.index); r8 = np.random.default_rng(MASTER + 81); idx = r8.integers(0, len(pids), (N_BOOT, len(pids)))
    fb, fa = base.far_ex.values, aw.loc[pids].far_ex.values
    ratio = fa[idx].mean(1) / np.maximum(fb[idx].mean(1), 1e-9); pa = float(max(np.mean(ratio >= 0.5), 1 / N_BOOT))
    dl = (base.sens - aw.loc[pids].sens).values; bl = boot_mean(dl, r8); pb = float(max(np.mean(bl >= 0.05), 1 / N_BOOT))
    TESTS['H8'] = dict(estimate=float(fa.mean() / max(fb.mean(), 1e-9)), sens_loss=float(np.nanmean(dl)), p=max(pa, pb), p_ratio=pa, p_sens=pb, n=int(len(pids)),
                       test='FAR ratio (aware/base) < 0.5 AND sensitivity loss < 0.05')
    save(); log('E6 done', R['E6'], TESTS['H6'], TESTS['H7'], TESTS['H8'])

# ------------------------------------------------------------------ Holm-Bonferroni across the confirmatory family
order = ['H1a', 'H1b', 'H1c', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']
have = [h for h in order if h in TESTS]
if have:
    ps = np.array([TESTS[h]['p'] if np.isfinite(TESTS[h]['p']) else 1.0 for h in have]); o = np.argsort(ps); m = len(order); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    for h, a in zip(have, adj): TESTS[h]['p_holm'] = float(a); TESTS[h]['reject_H0'] = bool(a <= 0.05)
    R['holm_family_size'] = m; R['tests_complete'] = len(have) == m
save(); log('ALL DONE ->', os.path.join(A.out, 'results.json'))
