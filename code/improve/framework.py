"""Development framework for the exercise-robust stress detector.
- Lockbox participants (improve/lockbox.json) are removed on load unless final=True.
- Candidates = (feature set, model, smoothing, threshold rule). The candidate is chosen inside each outer
  training fold by grouped 5-fold CV on the training participants only (nested LOPO).
- Objective (per held-out participant): specificity at the training-derived 0.80-sensitivity threshold,
  averaged over rest and exercise windows: SPEC = 1 - mean(FAR_rest, FAR_exercise).
Every evaluated configuration is appended to improve/trial_log.csv (multiplicity record)."""
import os, json, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
MASTER = 20261009
LOCK = json.load(open(os.path.join(os.path.dirname(__file__), 'lockbox.json')))['lockbox']
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

# ------------------------------------------------------------------ data
def detector_frame(W, final=False):
    """Stress-session windows (stress vs rest) + exercise windows (warm-up to post-exercise rest) as non-stress."""
    W = W.copy()
    for c in ['y_stress_rest']: W[c] = W[c].fillna('')
    d = W[(W.y_stress_rest != '') | ((W.protocol != 'STRESS') & (W.block != 'baseline'))].copy()
    d['y'] = (d.y_stress_rest == 'stress').astype(int)
    d['kind'] = np.where(d.protocol != 'STRESS', 'exercise', np.where(d.y == 1, 'stress', 'rest'))
    d['is_base'] = ((d.protocol == 'STRESS') & (d.block == 'baseline')).astype(int)
    if not final: d = d[~d.participant.isin(LOCK)]
    return d.sort_values(['protocol', 'session', 'start_s']).reset_index(drop=True)

def prev_index(d, horizon):
    """For each row, the indices of rows of the same session whose start lies in (t - horizon, t] (causal)."""
    out = [None] * len(d)
    for _, g in d.groupby(['protocol', 'session'], sort=False):
        idx = g.index.values; t = g.start_s.values
        for j in range(len(idx)):
            out[idx[j]] = idx[(t <= t[j]) & (t > t[j] - horizon)]
    return out

# ------------------------------------------------------------------ models
def model(kind):
    if kind == 'xgb': return XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, verbosity=0, random_state=MASTER)
    if kind == 'xgb_deep': return XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6, min_child_weight=5, tree_method='hist', n_jobs=1, verbosity=0, random_state=MASTER)
    if kind == 'lr': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000))
    if kind == 'et': return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced', n_jobs=1, random_state=MASTER))
    raise ValueError(kind)
def cw(y):
    u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}; return np.array([w[v] for v in y])
def fit(kind, X, y):
    m = model(kind)
    if kind.startswith('xgb'): m.fit(X, y, sample_weight=cw(y))
    else: m.fit(X, y)
    return m

# ------------------------------------------------------------------ decision layer
def smooth(p, prev_idx, rows, k_seconds):
    """causal mean of probabilities over the previous k_seconds (same session); rows = global row ids of p."""
    if k_seconds <= 15: return p
    pos = {r: i for i, r in enumerate(rows)}; out = p.copy()
    for i, r in enumerate(rows):
        ii = [pos[j] for j in prev_idx[k_seconds][r] if j in pos]
        if ii: out[i] = p[ii].mean()
    return out

def threshold(p_stress_train, sens=0.80):
    return float(np.quantile(p_stress_train, 1 - sens, method='lower'))

def participant_scores(d_te, p, tau):
    al = p >= tau; k = d_te.kind.values
    f = lambda m: float(al[m].mean()) if m.any() else np.nan
    sens, far_rest, far_ex = f(k == 'stress'), f(k == 'rest'), f(k == 'exercise')
    far_base = f((k == 'rest') & (d_te.is_base.values == 1))
    spec = 1 - np.nanmean([far_rest, far_ex])
    return dict(sens=sens, far_rest=far_rest, far_ex=far_ex, far_base=far_base, spec=spec)

# ------------------------------------------------------------------ nested evaluation
def oof_proba(d, feats, kind, n_folds=5):
    X, y, g = d[feats].values, d.y.values, d.participant.values; p = np.zeros(len(y))
    for a, b in GroupKFold(n_folds).split(X, y, g): p[b] = fit(kind, X[a], y[a]).predict_proba(X[b])[:, 1]
    return p

def inner_score(d, p, prev, rows, k, rule):
    """score a candidate on inner out-of-fold probabilities: threshold from the pooled inner stress windows."""
    ps = smooth(p, prev, rows, k)
    tau = threshold(ps[d.y.values == 1]); pos = pd.Series(np.arange(len(rows)), index=rows); sc = []
    for _, g in d.groupby('participant'):
        pg = ps[pos[g.index].values]
        sc.append(participant_scores(g, pg, tau if rule == 'global' else personal_tau(g, pg, tau))['spec'])
    return float(np.nanmean(sc))

def personal_tau(g, p, tau_global, q=0.95):
    """personal threshold: max of the global threshold and the 95th percentile of the person's own baseline-window scores
    (label-free calibration: 3 min of rest at set-up)."""
    b = g.is_base.values == 1
    return max(tau_global, float(np.quantile(p[b], q))) if b.any() else tau_global

def nested_lopo(d, candidates, featuresets, prev, tag, jobs=2, log_path='improve/trial_log.csv'):
    """candidates: list of dicts {fs, model, k, rule}. Returns per-participant results and chosen candidates."""
    parts = sorted(d.participant.unique())
    def one(pid):
        tr = d[d.participant != pid]; te = d[d.participant == pid]
        rows_tr = tr.index.values
        # inner selection
        cache = {}; scores = {}
        for c in candidates:
            key = (c['fs'], c['model'])
            if key not in cache: cache[key] = oof_proba(tr, featuresets[c['fs']], c['model'])
            scores[json.dumps(c, sort_keys=True)] = inner_score(tr, cache[key], prev, rows_tr, c['k'], c['rule'])
        best = json.loads(max(scores, key=scores.get))
        # outer fit
        feats = featuresets[best['fs']]
        p_oof = smooth(cache[(best['fs'], best['model'])], prev, rows_tr, best['k'])
        tau = threshold(p_oof[tr.y.values == 1])
        m = fit(best['model'], tr[feats].values, tr.y.values)
        p_te = smooth(m.predict_proba(te[feats].values)[:, 1], prev, te.index.values, best['k'])
        t_use = tau if best['rule'] == 'global' else personal_tau(te, p_te, tau)
        r = participant_scores(te, p_te, t_use); r.update(participant=pid, tau=t_use, chosen=json.dumps(best, sort_keys=True))
        if te.y.sum() > 0 and (te.y == 0).any(): r['auroc'] = float(roc_auc_score(te.y.values, p_te))
        return r, scores
    out = Parallel(n_jobs=jobs)(delayed(one)(pid) for pid in parts)
    res = pd.DataFrame([o[0] for o in out])
    # multiplicity log: every candidate's mean inner score across outer folds
    allsc = pd.DataFrame([o[1] for o in out]).mean()
    lg = pd.DataFrame({'tag': tag, 'candidate': allsc.index, 'mean_inner_score': allsc.values, 'time': time.strftime('%Y-%m-%d %H:%M'), 'score': 'specificity'})
    lg.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    return res

def summarize(res, label, rng=np.random.default_rng(MASTER)):
    def ci(v):
        v = v.dropna().values; b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        return f'{v.mean():.3f} ({np.percentile(b, 2.5):.3f}–{np.percentile(b, 97.5):.3f})'
    s = {k: ci(res[k]) for k in ['spec', 'sens', 'far_rest', 'far_ex', 'far_base', 'auroc'] if k in res}
    s['n'] = len(res); s['chosen'] = res.chosen.value_counts().head(3).to_dict(); s['label'] = label
    return s
