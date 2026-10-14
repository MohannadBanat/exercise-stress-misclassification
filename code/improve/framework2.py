"""M3 framework: decision-level candidates for the exercise-robust stress detector.
Candidate = (feature set, model, architecture, training labels, smoothing, threshold rule).
- architecture 'single': one model, stress vs everything else (motion features included).
- architecture 'twostage': stage 1 = logistic regression on accelerometer features, P(exercise);
  stage 2 = model trained on stress-session windows only (stress vs rest); p = p2 * (1 - p1).
- labels 'all' or 'clean': 'clean' drops the first half of each post-task rest block from TRAINING only.
- rule 'global': threshold at 0.80 sensitivity on pooled training out-of-fold stress windows;
  'personal': max(global, 95th percentile of the person's own baseline-window scores).
Selection and reporting score: balanced detection score BDS = (sensitivity + specificity) / 2, where
specificity = 1 - mean(FAR on non-baseline rest windows, FAR on exercise windows). Baseline windows are excluded from
the rest false-alarm rate because the personal rule uses them for calibration."""
import os, json, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from framework import fit, smooth, threshold, prev_index, detector_frame, LOCK, MASTER, log

def add_flags(d):
    d = d.copy()
    blk = d.groupby(['protocol', 'session', 'block']).start_s
    lo, hi = blk.transform('min'), blk.transform('max')
    d['rest_early'] = (d.block.isin(['rest1', 'rest2']) & (d.start_s < (lo + hi) / 2)).astype(int)
    return d

def motion_model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def train_predict(tr, te, feats, acc, kind, arch, labels):
    """fit on tr (DataFrame) and return stress probabilities for te."""
    keep = np.ones(len(tr), bool) if labels == 'all' else (tr.rest_early.values == 0)
    t = tr[keep]
    if arch == 'single':
        return fit(kind, t[feats].values, t.y.values).predict_proba(te[feats].values)[:, 1]
    m1 = motion_model().fit(t[acc].values, (t.kind == 'exercise').astype(int).values)
    s = t[t.protocol == 'STRESS']
    m2 = fit(kind, s[feats].values, s.y.values)
    return m2.predict_proba(te[feats].values)[:, 1] * (1 - m1.predict_proba(te[acc].values)[:, 1])

def personal_tau(g, p, tau_global, q=0.95):
    b = g.is_base.values == 1
    return max(tau_global, float(np.quantile(p[b], q))) if b.sum() >= 3 else tau_global

def scores(g, p, tau):
    al = p >= tau; k = g.kind.values; nb = g.is_base.values == 0
    f = lambda m: float(al[m].mean()) if m.any() else np.nan
    sens, far_rest_nb, far_rest_all, far_ex, far_base = f(k == 'stress'), f((k == 'rest') & nb), f(k == 'rest'), f(k == 'exercise'), f((k == 'rest') & ~nb)
    spec = 1 - np.nanmean([far_rest_nb, far_ex])
    return dict(sens=sens, far_rest=far_rest_nb, far_rest_all=far_rest_all, far_ex=far_ex, far_base=far_base, spec=spec, bds=(sens + spec) / 2)

def eval_candidate(d, p, prev, k, rule, tau=None):
    """apply smoothing and threshold rule; tau computed from d's own stress windows if not given (inner scoring)."""
    rows = d.index.values; ps = smooth(p, prev, rows, k)
    if tau is None: tau = threshold(ps[d.y.values == 1])
    pos = pd.Series(np.arange(len(rows)), index=rows); out = []
    for pid, g in d.groupby('participant'):
        pg = ps[pos[g.index].values]; t = tau if rule == 'global' else personal_tau(g, pg, tau)
        r = scores(g, pg, t); r['participant'] = pid; r['tau'] = t; out.append(r)
    return pd.DataFrame(out), ps

def oof(d, feats, acc, kind, arch, labels, n=5):
    p = np.zeros(len(d)); g = d.participant.values
    for a, b in GroupKFold(n).split(d, d.y, g):
        p[b] = train_predict(d.iloc[a], d.iloc[b], feats, acc, kind, arch, labels)
    return p

def nested(d, C, FS, acc, prev, tag, reference, jobs=2, log_path='improve/trial_log.csv'):
    parts = sorted(d.participant.unique())
    def one(pid):
        tr, te = d[d.participant != pid], d[d.participant == pid]
        cache, sc = {}, {}
        for c in C:
            key = (c['fs'], c['model'], c['arch'], c['labels'])
            if key not in cache: cache[key] = oof(tr, FS[c['fs']], acc, c['model'], c['arch'], c['labels'])
            sc[json.dumps(c, sort_keys=True)] = float(eval_candidate(tr, cache[key], prev, c['k'], c['rule'])[0].bds.mean())
        out = {}
        for name, c in (('selected', json.loads(max(sc, key=sc.get))), ('reference', reference)):
            key = (c['fs'], c['model'], c['arch'], c['labels'])
            if key not in cache: cache[key] = oof(tr, FS[c['fs']], acc, c['model'], c['arch'], c['labels'])
            tau = threshold(smooth(cache[key], prev, tr.index.values, c['k'])[tr.y.values == 1])
            p = train_predict(tr, te, FS[c['fs']], acc, c['model'], c['arch'], c['labels'])
            r, ps = eval_candidate(te, p, prev, c['k'], c['rule'], tau=tau)
            r = r.iloc[0].to_dict(); r['chosen'] = json.dumps(c, sort_keys=True)
            if te.y.sum() and (te.y == 0).any(): r['auroc'] = float(roc_auc_score(te.y.values, ps))
            out[name] = r
        return out, sc
    res = Parallel(n_jobs=jobs)(delayed(one)(pid) for pid in parts)
    sel = pd.DataFrame([r[0]['selected'] for r in res]); ref = pd.DataFrame([r[0]['reference'] for r in res])
    allsc = pd.DataFrame([r[1] for r in res]).mean()
    pd.DataFrame({'tag': tag, 'candidate': allsc.index, 'mean_inner_score': allsc.values, 'time': time.strftime('%Y-%m-%d %H:%M'), 'score': 'BDS'}).to_csv(
        log_path, mode='a', header=not os.path.exists(log_path), index=False)
    return sel, ref

def summarize(res, label, seed=MASTER):
    rng = np.random.default_rng(seed)
    def ci(v):
        v = v.dropna().values; b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        return [round(float(v.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    s = {k: ci(res[k]) for k in ['bds', 'sens', 'spec', 'far_rest', 'far_rest_all', 'far_ex', 'far_base', 'auroc'] if k in res}
    s['n'] = len(res); s['chosen'] = res.chosen.value_counts().head(4).to_dict(); s['label'] = label
    return s

def paired_gain(a, b, key='bds', seed=MASTER):
    rng = np.random.default_rng(seed); d = (a.set_index('participant')[key] - b.set_index('participant')[key]).dropna().values
    bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(gain=round(float(d.mean()), 3), ci=[round(float(np.percentile(bm, 2.5)), 3), round(float(np.percentile(bm, 97.5)), 3)], p_le0=float(max(np.mean(bm <= 0), 1e-4)))
