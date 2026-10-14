"""Post hoc controls added after internal review of manuscript v10 (all exploratory, labelled post hoc in the paper).
S1  final detector (REF pipeline): outer LOPO scores on all 36 participants (same folds/seeds as M5), window-level false-alarm
    rate by session position, detector vs time-only model (within-session and time-matched AUROC), and a circular-shift null
    for the episode metrics.
S2  window-level stress vs rest under the registered model (XGBoost, nested tuning, LOPO): all features, physiology only,
    accelerometer only (time only = registered H3).
S3  registered E6 physiology-only detector re-run with identical code/seeds to obtain window scores, then scored with the
    episode rules of Section 3.4 (events per hour).
Usage (from code/):  python improve/review_controls.py S1 S2 S3"""
import sys, os, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from run_m5_episode import build_streams, events_for_stream, score_stream, evaluate, G_PRIMARY
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, fit as fw_fit, log
from build_features import build
OUT = 'improve/review'; os.makedirs(OUT, exist_ok=True)
MASTER = 20261009; RB = np.random.default_rng(MASTER)
def boot(v, n=10000, seed=MASTER):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; r = np.random.default_rng(seed); b = v[r.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))]
def save(key, obj):
    f = f'{OUT}/sum_review.json'; S = json.load(open(f)) if os.path.exists(f) else {}; S[key] = obj; json.dump(S, open(f, 'w'), indent=1, default=float)

W, FS, ACCF = build(); d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
streams = build_streams(d, seg); feats = FS['v1_cal_ctx']; parts = sorted(d.participant.unique())
R5 = {r['pid']: r for r in pickle.load(open('improve/m5_all.pkl', 'rb'))}

# =========================================================== S1
if 'S1' in sys.argv:
    prev = {45: prev_index(d, 45)}
    f1 = f'{OUT}/outer_scores.npy'
    if os.path.exists(f1): S_out = np.load(f1)
    else:
        def one(pid):
            tr, te = d[d.participant != pid], d[d.participant == pid]
            return te.index.values, smooth(train_predict(tr, te, feats, ACCF, 'et', 'single', 'all'), prev, te.index.values, 45)
        S_out = np.full(len(d), np.nan)
        for idx, p in Parallel(n_jobs=2)(delayed(one)(pid) for pid in parts): S_out[idx] = p
        np.save(f1, S_out)
    # consistency with M5 (reference rule)
    for pid in parts[:3] + parts[-3:]:
        a = evaluate(streams, {pid}, S_out, R5[pid]['taus'][0.8], 1).iloc[0]; b = R5[pid]['grid'][(0.8, 1)]
        assert a.det == b['det'] and a.fa_rest == b['fa_rest'] and a.fa_ex == b['fa_ex'], (pid, a.to_dict(), b)
    log('S1 outer scores reproduce M5')
    tau = {pid: R5[pid]['taus'][0.8] for pid in parts}
    d['al'] = [S_out[i] >= tau[p] for i, p in zip(d.index, d.participant)]
    # window FAR by position (per participant means)
    st = d[d.protocol == 'STRESS'].copy()
    blk = st.groupby(['session', 'block']).start_s; lo, hi = blk.transform('min'), blk.transform('max')
    st['third'] = np.where(st.block.isin(['rest1', 'rest2']), np.clip(((st.start_s - lo) / (hi - lo + 1e-9) * 3).astype(int), 0, 2), -1)
    pos = {}
    pos['baseline'] = boot(st[st.is_base == 1].groupby('participant').al.mean())
    for k, nm in enumerate(['rest_first_third', 'rest_middle_third', 'rest_last_third']): pos[nm] = boot(st[st.third == k].groupby('participant').al.mean())
    pos['rest_nonbaseline'] = boot(st[(st.kind == 'rest') & (st.is_base == 0)].groupby('participant').al.mean())
    pos['stress'] = boot(st[st.kind == 'stress'].groupby('participant').al.mean())
    pos['exercise'] = boot(d[d.kind == 'exercise'].groupby('participant').al.mean())
    save('S1_window_far_by_position', pos); log('S1 positions', {k: round(v[0], 2) for k, v in pos.items()})
    # time-only model vs detector, within the stress session
    st['v2'] = (st.version == 'V2').astype(int)
    def tonly(pid):
        tr, te = st[st.participant != pid], st[st.participant == pid]
        m = fw_fit('et', tr[['t_session', 'v2']].values, tr.y.values)
        return te.index.values, m.predict_proba(te[['t_session', 'v2']].values)[:, 1]
    T = pd.Series(np.nan, index=st.index)
    for idx, p in Parallel(n_jobs=2)(delayed(tonly)(pid) for pid in parts): T.loc[idx] = p
    st['p_det'] = S_out[st.index.values]; st['p_time'] = T
    rows = []
    for pid, g in st.groupby('participant'):
        s_t = g.start_s[g.kind == 'stress'].values
        near = (g.kind == 'stress') | ((g.kind == 'rest') & g.start_s.apply(lambda t: np.min(np.abs(s_t - t)) <= 180).values)
        nb = g.is_base == 0
        r = dict(participant=pid)
        for nm, m in (('all', np.ones(len(g), bool)), ('nobase', nb.values), ('matched', near.values)):
            gg = g[m]
            if gg.y.nunique() == 2:
                r[f'det_{nm}'] = roc_auc_score(gg.y, gg.p_det); r[f'time_{nm}'] = roc_auc_score(gg.y, gg.p_time)
        rows.append(r)
    A = pd.DataFrame(rows); A.to_csv(f'{OUT}/auroc_det_vs_time.csv', index=False)
    out = {}
    for nm in ('all', 'nobase', 'matched'):
        out[f'det_{nm}'] = boot(A[f'det_{nm}']); out[f'time_{nm}'] = boot(A[f'time_{nm}']); out[f'diff_{nm}'] = boot(A[f'det_{nm}'] - A[f'time_{nm}'])
    save('S1_auroc_det_vs_time', out); log('S1 auroc', {k: [round(x, 3) for x in v[:3]] for k, v in out.items()})
    # circular-shift null for episode detection (reference rule and per-fold selected rule)
    rng = np.random.default_rng(MASTER + 5); NREP = 1000
    sess = {S_['pid']: S_ for k_, S_ in streams.items() if k_[0] == 'STRESS'}; assert len(sess) == 36
    def det_count(S, sc, tau_, N, strict):
        o = score_stream(S, sc, tau_, N, G_PRIMARY, strict); return o['det'], o['n_ep']
    res = {}
    for name, rule in (('reference', lambda p: (0.8, 1)), ('selected', lambda p: R5[p]['best'])):
        for strict in (False, True):
            obs = sum(det_count(sess[p], S_out[sess[p]['rows']], R5[p]['taus'][rule(p)[0]], rule(p)[1], strict)[0] for p in parts)
            null = np.zeros(NREP)
            for k in range(NREP):
                tot = 0
                for p in parts:
                    S = sess[p]; sc = S_out[S['rows']]; sh = rng.integers(1, len(sc))
                    tot += det_count(S, np.roll(sc, sh), R5[p]['taus'][rule(p)[0]], rule(p)[1], strict)[0]
                null[k] = tot / 108
            res[f'{name}_{"strict" if strict else "overlap"}'] = dict(observed=obs / 108, null_mean=float(null.mean()), null_p2_5=float(np.percentile(null, 2.5)),
                                                                     null_p97_5=float(np.percentile(null, 97.5)), p_upper=float((1 + (null >= obs / 108).sum()) / (NREP + 1)))
            log('S1 null', name, strict, res[f'{name}_{"strict" if strict else "overlap"}'])
    save('S1_episode_null', res)

# =========================================================== S2 (registered model style)
XGB_GRID = [dict(max_depth=dd, n_estimators=n, learning_rate=l) for dd in (2, 3) for n in (100, 300) for l in (0.05, 0.1)]
def xmake(p): return XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, verbosity=0, random_state=20261008)
def xcw(y):
    u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}; return np.array([w[v] for v in y])
def xfit(p, X, y): m = xmake(p); m.fit(X, y, sample_weight=xcw(y)); return m
def xselect(X, y, g):
    folds = list(GroupKFold(5).split(X, y, g)); best, bs = None, -1
    for p in XGB_GRID:
        pred = np.empty(len(y), dtype=y.dtype)
        for a, b in folds: pred[b] = xfit(p, X[a], y[a]).predict(X[b])
        s = balanced_accuracy_score(y, pred)
        if s > bs: best, bs = p, s
    return best
def xlopo(X, y, g):
    def one(pid):
        te = g == pid; tr = ~te; p = xselect(X[tr], y[tr], g[tr]); return pid, xfit(p, X[tr], y[tr]).predict(X[te])
    pred = np.empty(len(y), dtype=y.dtype)
    for pid, pr in Parallel(n_jobs=2)(delayed(one)(pid) for pid in np.unique(g)): pred[g == pid] = pr
    return pred
def pp_ba(y, pred, g):
    return pd.Series({pid: np.mean([np.mean(pred[(g == pid) & (y == c)] == c) for c in np.unique(y[g == pid])]) for pid in np.unique(g)})
EDA = ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']
BVP = ['bvp_pr_mean','bvp_pr_sd','bvp_sdnn','bvp_rmssd','bvp_amp_sd','bvp_coverage']; HRc = ['hr_mean']; TEMP = ['temp_mean','temp_sd','temp_slope']
ACC = ['acc_mag_mean','acc_mag_sd','acc_x_mean','acc_x_sd','acc_y_mean','acc_y_sd','acc_z_mean','acc_z_sd','acc_dom_freq','acc_pow_frac_0p5_2']
PHYS = EDA + BVP + HRc + TEMP
if 'S2' in sys.argv:
    Wr = pd.read_csv('features/windows.csv'); Wr['y_stress_rest'] = Wr.y_stress_rest.fillna('')
    sr = Wr[Wr.y_stress_rest != '']; y = (sr.y_stress_rest == 'stress').astype(int).values; g = sr.participant.values
    out = {}
    for nm, cols in (('acc_only', ACC), ('phys_only', PHYS), ('all', PHYS + ACC)):
        pr = xlopo(sr[cols].values.astype(float), y, g); ba = pp_ba(y, pr, g); ba.to_csv(f'{OUT}/S2_{nm}.csv'); out[nm] = boot(ba.values); log('S2', nm, out[nm])
    t = pd.read_csv('results_real/E3_stress_rest_time_perparticipant.csv', index_col=0).time_only; out['time_only_registered'] = boot(t.values)
    a = pd.read_csv(f'{OUT}/S2_acc_only.csv', index_col=0).iloc[:, 0]; ph = pd.read_csv(f'{OUT}/S2_phys_only.csv', index_col=0).iloc[:, 0]
    out['phys_minus_time'] = boot((ph - t.reindex(ph.index)).values); out['phys_minus_acc'] = boot((ph - a).values)
    save('S2_stress_rest_window', out)

# =========================================================== S3 registered E6 detector, event level
if 'S3' in sys.argv:
    Wr = pd.read_csv('features/windows.csv');
    for c in ['y_stress_rest']: Wr[c] = Wr[c].fillna('')
    S = Wr[Wr.y_stress_rest != ''].copy(); S['y'] = (S.y_stress_rest == 'stress').astype(int)
    EX = Wr[(Wr.protocol != 'STRESS') & (Wr.block != 'baseline')].copy(); test_parts = sorted(set(EX.participant))
    def det(pid):
        tr = S[S.participant != pid]; Xtr, ytr, gtr = tr[PHYS].values, tr.y.values, tr.participant.values
        p = xselect(Xtr, ytr, gtr); oof = np.zeros(len(ytr))
        for a, b in GroupKFold(5).split(Xtr, ytr, gtr): oof[b] = xfit(p, Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        tau = float(np.quantile(oof[ytr == 1], 0.20, method='lower')); m = xfit(p, Xtr, ytr)
        te_s, te_e = S[S.participant == pid], EX[EX.participant == pid]
        return pid, tau, te_s.index.values, m.predict_proba(te_s[PHYS].values)[:, 1], te_e.index.values, m.predict_proba(te_e[PHYS].values)[:, 1]
    f3 = f'{OUT}/e6_scores.pkl'
    if os.path.exists(f3): RES = pickle.load(open(f3, 'rb'))
    else:
        RES = Parallel(n_jobs=2)(delayed(det)(pid) for pid in test_parts); pickle.dump(RES, open(f3, 'wb'))
    e6 = pd.read_csv('results_real/E6_perparticipant.csv'); e6 = e6[~e6.aware].set_index('participant')
    key = d.set_index(['protocol', 'session', 'start_s']).index
    Sx = np.full(len(d), np.nan); tau = {}; chk = []
    for pid, t, i_s, p_s, i_e, p_e in RES:
        tau[pid] = t; chk.append(abs(np.mean(p_e >= t) - e6.loc[pid, 'far_ex']))
        for ii, pp in ((i_s, p_s), (i_e, p_e)):
            k = pd.MultiIndex.from_frame(Wr.loc[ii, ['protocol', 'session', 'start_s']]); loc = key.get_indexer(k); ok = loc >= 0
            Sx[loc[ok]] = pp[ok]
    log('S3 reproduction max |far_ex diff|', max(chk))
    rows = [evaluate(streams, {pid}, Sx, tau[pid], 1).iloc[0].to_dict() for pid in test_parts]
    df = pd.DataFrame(rows); df.drop(columns='lat').to_csv(f'{OUT}/S3_e6_events.csv', index=False)
    tot = df[['n_ep', 'det', 'fa_rest', 'fa_ex', 'fa_base', 'h_rest', 'h_ex', 'h_base', 'al_rest', 'n_rest']].sum()
    idx = np.random.default_rng(MASTER).integers(0, len(df), (10000, len(df))); X = df[['det', 'n_ep', 'fa_rest', 'h_rest', 'fa_ex', 'h_ex']].values.astype(float)
    B = X[idx].sum(1)
    ci = lambda num, den: [float(tot[num] / tot[den]), float(np.percentile(B[:, ['det', 'n_ep', 'fa_rest', 'h_rest', 'fa_ex', 'h_ex'].index(num)] / B[:, ['det', 'n_ep', 'fa_rest', 'h_rest', 'fa_ex', 'h_ex'].index(den)], 2.5)),
                           float(np.percentile(B[:, ['det', 'n_ep', 'fa_rest', 'h_rest', 'fa_ex', 'h_ex'].index(num)] / B[:, ['det', 'n_ep', 'fa_rest', 'h_rest', 'fa_ex', 'h_ex'].index(den)], 97.5))]
    out = dict(n=len(df), reproduction_max_abs_diff=float(max(chk)), sens=ci('det', 'n_ep'), fa_rest=ci('fa_rest', 'h_rest'), fa_ex=ci('fa_ex', 'h_ex'),
               n_fa_ex=int(tot.fa_ex), h_ex=float(tot.h_ex), alarm_time_ex=float(np.nanmean([np.mean(Sx[S_['rows']] >= tau[S_['pid']]) for k_, S_ in streams.items() if k_[0] != 'STRESS' and S_['pid'] in tau])))
    save('S3_e6_event_level', out); log('S3', out)
