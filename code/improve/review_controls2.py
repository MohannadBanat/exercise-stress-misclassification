"""Second round of post hoc controls (after the second internal review). All post hoc.
S4  registered E6 detector: false-alarm rate by exercise phase (incl. cool-down, post-exercise rest, and the exercise-session
    baselines of V2), GEE of alarms on intensity adjusted for elapsed time, bias-reduced (small-sample) sandwich SEs, and a
    within-session sprint vs following cool-down comparison.
S5  registered E6 detector with a cStress-style accelerometer gate (alarms suppressed when the accelerometer magnitude SD
    exceeds the 95th percentile of the training participants' stress-session windows).
S6  time+motion comparator for the final detector: same pipeline (ET, 45 s smoothing, 0.80 threshold from training OOF) on
    the ten accelerometer features plus elapsed time and cohort; LOPO on 36.
S7  sealed participants only, with the frozen sealed-test model (fitted on the 24 development participants).
Usage (from code/): python improve/review_controls2.py S4 S5 S6 S7"""
import sys, os, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from run_m5_episode import build_streams, score_stream, evaluate, G_PRIMARY
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, fit as fw_fit, log, LOCK
from build_features import build
import importlib.util
spec = importlib.util.spec_from_file_location('rc', 'improve/review_controls.py')
OUT = 'improve/review'; MASTER = 20261009
def boot(v, n=10000, seed=MASTER):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; r = np.random.default_rng(seed); b = v[r.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))]
def save(key, obj):
    f = f'{OUT}/sum_review.json'; S = json.load(open(f)) if os.path.exists(f) else {}; S[key] = obj; json.dump(S, open(f, 'w'), indent=1, default=float)
def pooled_ep(df):
    s = df[['n_ep', 'det', 'fa_rest', 'fa_ex', 'fa_base', 'h_rest', 'h_ex', 'h_base', 'al_rest', 'n_rest']].astype(float)
    ix = np.random.default_rng(MASTER).integers(0, len(s), (10000, len(s))); B = s.values[ix].sum(1); T = s.sum()
    col = list(s.columns); r = lambda a, b: [float(T[a] / T[b]), float(np.percentile(B[:, col.index(a)] / B[:, col.index(b)], 2.5)), float(np.percentile(B[:, col.index(a)] / B[:, col.index(b)], 97.5))]
    out = dict(sens=r('det', 'n_ep'), fa_rest=r('fa_rest', 'h_rest'), alarm_rest=r('al_rest', 'n_rest'), n=len(s))
    if T['h_ex'] > 0: out['fa_ex'] = r('fa_ex', 'h_ex')
    if T['h_base'] > 0: out['fa_base'] = r('fa_base', 'h_base')
    return out

W, FS, ACCF = build(); d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
streams = build_streams(d, seg); parts = sorted(d.participant.unique())
Wr = pd.read_csv('features/windows.csv'); Wr['y_stress_rest'] = Wr.y_stress_rest.fillna('')
EDA = ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']
BVP = ['bvp_pr_mean','bvp_pr_sd','bvp_sdnn','bvp_rmssd','bvp_amp_sd','bvp_coverage']; HRc = ['hr_mean']; TEMP = ['temp_mean','temp_sd','temp_slope']
ACC = ['acc_mag_mean','acc_mag_sd','acc_x_mean','acc_x_sd','acc_y_mean','acc_y_sd','acc_z_mean','acc_z_sd','acc_dom_freq','acc_pow_frac_0p5_2']
PHYS = EDA + BVP + HRc + TEMP
key = d.set_index(['protocol', 'session', 'start_s']).index
def to_d(idx_w, vals, arr):
    loc = key.get_indexer(pd.MultiIndex.from_frame(Wr.loc[idx_w, ['protocol', 'session', 'start_s']])); ok = loc >= 0; arr[loc[ok]] = np.asarray(vals)[ok]

# ------------------------------------------------ E6 detector re-run with exercise baselines scored too (identical models)
if any(k in sys.argv for k in ('S4', 'S5')):
    from xgboost import XGBClassifier
    from sklearn.metrics import balanced_accuracy_score
    S = Wr[Wr.y_stress_rest != ''].copy(); S['y'] = (S.y_stress_rest == 'stress').astype(int)
    EXALL = Wr[Wr.protocol != 'STRESS'].copy(); test_parts = sorted(set(Wr[(Wr.protocol != 'STRESS') & (Wr.block != 'baseline')].participant))
    GRID = [dict(max_depth=a, n_estimators=n, learning_rate=l) for a in (2, 3) for n in (100, 300) for l in (0.05, 0.1)]
    def mk(p): return XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, verbosity=0, random_state=20261008)
    def cw(y):
        u, c = np.unique(y, return_counts=True); w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}; return np.array([w[v] for v in y])
    def xf(p, X, y): m = mk(p); m.fit(X, y, sample_weight=cw(y)); return m
    def sel(X, y, g):
        f = list(GroupKFold(5).split(X, y, g)); best, bs = None, -1
        for p in GRID:
            pr = np.empty(len(y), int)
            for a, b in f: pr[b] = xf(p, X[a], y[a]).predict(X[b])
            s_ = balanced_accuracy_score(y, pr)
            if s_ > bs: best, bs = p, s_
        return best
    f4 = f'{OUT}/e6_scores_all.pkl'
    if os.path.exists(f4): E6 = pickle.load(open(f4, 'rb'))
    else:
        def det(pid):
            tr = S[S.participant != pid]; Xtr, ytr, gtr = tr[PHYS].values, tr.y.values, tr.participant.values
            p = sel(Xtr, ytr, gtr); o = np.zeros(len(ytr))
            for a, b in GroupKFold(5).split(Xtr, ytr, gtr): o[b] = xf(p, Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
            tau = float(np.quantile(o[ytr == 1], 0.20, method='lower')); m = xf(p, Xtr, ytr)
            te_s, te_e = S[S.participant == pid], EXALL[EXALL.participant == pid]
            theta = float(np.quantile(tr.acc_mag_sd.dropna(), 0.95))
            return dict(pid=pid, tau=tau, theta=theta, i_s=te_s.index.values, p_s=m.predict_proba(te_s[PHYS].values)[:, 1], i_e=te_e.index.values, p_e=m.predict_proba(te_e[PHYS].values)[:, 1])
        E6 = Parallel(n_jobs=2)(delayed(det)(pid) for pid in test_parts); pickle.dump(E6, open(f4, 'wb'))
    old = {r[0]: r for r in pickle.load(open(f'{OUT}/e6_scores.pkl', 'rb'))}
    for r in E6:   # identical to the first re-run (and to the registered E6 results)
        o = old[r['pid']]; m = pd.Series(r['p_e'], index=r['i_e']).loc[o[4]].values; assert np.allclose(m, o[5]) and abs(r['tau'] - o[1]) < 1e-12
    log('E6 re-run identical')

if 'S4' in sys.argv:
    rows = []
    for r in E6:
        e = EXALL.loc[r['i_e']].copy(); e['al'] = (r['p_e'] >= r['tau']).astype(int); e['participant'] = r['pid']; rows.append(e)
    E = pd.concat(rows)
    E['phase'] = np.select([E.block == 'baseline', E.block == 'warmup', E.block.str.startswith('stage'), E.block.str.startswith('sprint'), E.block.str.startswith('cooldown'), E.block == 'rest'],
                           ['baseline', 'warm-up', 'aerobic stage', 'sprint', 'cool-down', 'post-exercise rest'], 'other')
    ph = {p: boot(g.groupby('participant').al.mean()) for p, g in E.groupby('phase')}
    log('S4 FAR by phase', {k: [round(x, 2) for x in v[:3]] + [v[3]] for k, v in ph.items()})
    import statsmodels.api as sm, statsmodels.formula.api as smf
    G = E[E.intensity.notna()].copy(); G['v2'] = (G.version == 'V2').astype(int); G['tmin'] = G.t_session / 60
    out = dict(far_by_phase=ph)
    for nm, form in (('registered', 'al ~ intensity + v2'), ('time_adjusted', 'al ~ intensity + tmin + v2')):
        for cov in ('robust', 'bias_reduced'):
            f = smf.gee(form, groups='participant', data=G, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type=cov)
            ci = f.conf_int().loc['intensity']; out[f'gee_{nm}_{cov}'] = dict(or_=float(np.exp(f.params['intensity'])), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])), p=float(f.pvalues['intensity']))
            if 'tmin' in f.params: out[f'gee_{nm}_{cov}']['or_tmin'] = float(np.exp(f.params['tmin']))
    log('S4 GEE', {k: v for k, v in out.items() if k.startswith('gee')})
    # sprint vs the cool-down that follows it, within participant (anaerobic sessions): intensity falls while time advances
    an = E[E.protocol == 'ANAEROBIC']
    pp = an.groupby(['participant', 'phase']).al.mean().unstack()
    dd = (pp['sprint'] - pp['cool-down']).dropna(); out['sprint_minus_cooldown'] = boot(dd.values)
    out['n_windows_by_phase'] = E.phase.value_counts().to_dict()
    out['share_exercise_windows_cooldown_or_rest'] = float(E[E.phase != 'baseline'].phase.isin(['cool-down', 'post-exercise rest']).mean())
    save('S4_e6_by_phase', out); log('S4', out['sprint_minus_cooldown'], out['share_exercise_windows_cooldown_or_rest'])

if 'S5' in sys.argv:
    Sx = np.full(len(d), np.nan); Sg = np.full(len(d), np.nan); res = []
    for r in E6:
        for ii, pp_ in ((r['i_s'], r['p_s']), (r['i_e'], r['p_e'])):
            acc = Wr.loc[ii, 'acc_mag_sd'].values; gated = np.where(np.nan_to_num(acc, nan=0) > r['theta'], 0.0, pp_)
            to_d(ii, pp_, Sx); to_d(ii, gated, Sg)
    win = []
    for r in E6:
        g = d[d.participant == r['pid']]; a0 = Sx[g.index] >= r['tau']; a1 = Sg[g.index] >= r['tau']
        k = g.kind.values; nb = g.is_base.values == 0
        win.append(dict(participant=r['pid'], sens0=a0[k == 'stress'].mean(), sens1=a1[k == 'stress'].mean(), rest0=a0[(k == 'rest') & nb].mean(), rest1=a1[(k == 'rest') & nb].mean(),
                        ex0=a0[k == 'exercise'].mean(), ex1=a1[k == 'exercise'].mean()))
    Wd = pd.DataFrame(win); out = {c: boot(Wd[c]) for c in Wd.columns if c != 'participant'}
    ep0 = pd.DataFrame([evaluate(streams, {r['pid']}, Sx, r['tau'], 1).iloc[0].to_dict() for r in E6])
    ep1 = pd.DataFrame([evaluate(streams, {r['pid']}, Sg, r['tau'], 1).iloc[0].to_dict() for r in E6])
    out['episodes_ungated'] = pooled_ep(ep0); out['episodes_gated'] = pooled_ep(ep1)
    out['theta_mean'] = float(np.mean([r['theta'] for r in E6]))
    save('S5_e6_motion_gate', out); log('S5', {k: [round(x, 3) for x in v[:3]] for k, v in out.items() if isinstance(v, list)}, out['episodes_gated'])

# ------------------------------------------------ S6 time + motion comparator for the final detector
if 'S6' in sys.argv:
    prev = {45: prev_index(d, 45)}; d['v2'] = (d.version == 'V2').astype(int); TM = ACC + ['t_session', 'v2']
    f6 = f'{OUT}/tm_scores.pkl'
    if os.path.exists(f6): S_tm, tau_tm = pickle.load(open(f6, 'rb'))
    else:
        def one(pid):
            tr, te = d[d.participant != pid], d[d.participant == pid]
            po = smooth(oof(tr, TM, ACCF, 'et', 'single', 'all'), prev, tr.index.values, 45); tau = threshold(po[tr.y.values == 1])
            return pid, tau, te.index.values, smooth(train_predict(tr, te, TM, ACCF, 'et', 'single', 'all'), prev, te.index.values, 45)
        S_tm = np.full(len(d), np.nan); tau_tm = {}
        for pid, tau, idx, p in Parallel(n_jobs=2)(delayed(one)(pid) for pid in parts): S_tm[idx] = p; tau_tm[pid] = tau
        pickle.dump((S_tm, tau_tm), open(f6, 'wb'))
    S_det = np.load(f'{OUT}/outer_scores.npy'); R5 = {r['pid']: r for r in pickle.load(open('improve/m5_all.pkl', 'rb'))}
    tau_det = {p: R5[p]['taus'][0.8] for p in parts}
    rows = []
    for pid in parts:
        g = d[d.participant == pid]; k = g.kind.values; nb = g.is_base.values == 0; st = g.protocol.values == 'STRESS'
        r = dict(participant=pid)
        for nm, Sc, t in (('det', S_det, tau_det[pid]), ('tm', S_tm, tau_tm[pid])):
            a = Sc[g.index] >= t
            r[f'{nm}_sens'] = a[k == 'stress'].mean(); r[f'{nm}_rest'] = a[(k == 'rest') & nb].mean(); r[f'{nm}_base'] = a[(k == 'rest') & ~nb].mean()
            r[f'{nm}_ex'] = a[k == 'exercise'].mean() if (k == 'exercise').any() else np.nan
            r[f'{nm}_auc'] = roc_auc_score(g.y.values[st], Sc[g.index][st])
        rows.append(r)
    D = pd.DataFrame(rows); out = {c: boot(D[c]) for c in D.columns if c != 'participant'}
    out['diff_auc'] = boot(D.det_auc - D.tm_auc); out['diff_rest'] = boot(D.det_rest - D.tm_rest); out['diff_ex'] = boot(D.det_ex - D.tm_ex); out['diff_sens'] = boot(D.det_sens - D.tm_sens)
    ep = pd.DataFrame([evaluate(streams, {p}, S_tm, tau_tm[p], 1).iloc[0].to_dict() for p in parts]); out['episodes_tm'] = pooled_ep(ep)
    save('S6_time_motion', out); log('S6', {k: [round(x, 3) for x in v[:3]] for k, v in out.items() if isinstance(v, list)}, out['episodes_tm'])

# ------------------------------------------------ S7 sealed participants only, frozen sealed-test model
if 'S7' in sys.argv:
    prev = {45: prev_index(d, 45)}; FE = FS['v1_cal_ctx']
    dev, box = d[~d.participant.isin(LOCK)], d[d.participant.isin(LOCK)]
    tau = threshold(smooth(oof(dev, FE, ACCF, 'et', 'single', 'all'), prev, dev.index.values, 45)[dev.y.values == 1])
    Sb = np.full(len(d), np.nan); Sb[box.index.values] = smooth(train_predict(dev, box, FE, ACCF, 'et', 'single', 'all'), prev, box.index.values, 45)
    m4 = pd.read_csv('improve/res_M4_selected.csv').set_index('participant')
    chk = []
    for pid, g in box.groupby('participant'):
        a = Sb[g.index] >= tau; chk.append(abs(a[g.kind.values == 'stress'].mean() - m4.loc[pid, 'sens']))
    log('S7 reproduces M4 sensitivity, max diff', max(chk))
    ep = pd.DataFrame([evaluate(streams, {p}, Sb, tau, 1).iloc[0].to_dict() for p in sorted(LOCK)]); out = dict(episodes=pooled_ep(ep), reproduction_max_diff=float(max(chk)), n=len(LOCK))
    rng = np.random.default_rng(MASTER + 7); sess = {S_['pid']: S_ for k_, S_ in streams.items() if k_[0] == 'STRESS' and S_['pid'] in LOCK}
    obs = sum(score_stream(sess[p], Sb[sess[p]['rows']], tau, 1, G_PRIMARY, False)['det'] for p in sess) / (3 * len(sess)); null = []
    for _ in range(1000):
        null.append(sum(score_stream(S_, np.roll(Sb[S_['rows']], rng.integers(1, len(S_['rows']))), tau, 1, G_PRIMARY, False)['det'] for S_ in sess.values()) / (3 * len(sess)))
    null = np.array(null); out['null'] = dict(observed=obs, mean=float(null.mean()), lo=float(np.percentile(null, 2.5)), hi=float(np.percentile(null, 97.5)), n_ge=int((null >= obs).sum()))
    st = box[box.protocol == 'STRESS'].copy(); st['v2'] = (st.version == 'V2').astype(int); dv = dev[dev.protocol == 'STRESS'].copy(); dv['v2'] = (dv.version == 'V2').astype(int)
    mt = fw_fit('et', dv[['t_session', 'v2']].values, dv.y.values); pt = mt.predict_proba(st[['t_session', 'v2']].values)[:, 1]
    st['p_det'] = Sb[st.index]; st['p_t'] = pt
    au = pd.DataFrame([dict(participant=p, det=roc_auc_score(g.y, g.p_det), time=roc_auc_score(g.y, g.p_t)) for p, g in st.groupby('participant')])
    out['auc_det'] = boot(au.det); out['auc_time'] = boot(au.time); out['auc_diff'] = boot(au.det - au.time)
    save('S7_sealed_only', out); log('S7', out)

# ------------------------------------------------ S4b seated pre-task baselines: before exercise vs before the stress tasks (same V2 people)
if 'S4b' in sys.argv:
    E6 = pickle.load(open(f'{OUT}/e6_scores_all.pkl', 'rb')); rows = []
    for r in E6:
        s_, e_ = Wr.loc[r['i_s']], Wr.loc[r['i_e']]
        sb = r['p_s'][(s_.block == 'baseline').values] >= r['tau']; eb = r['p_e'][(e_.block == 'baseline').values] >= r['tau']
        if len(eb) and len(sb): rows.append(dict(pid=r['pid'], stress_base=sb.mean(), ex_base=eb.mean()))
    D = pd.DataFrame(rows); rng = np.random.default_rng(20261009); n = len(D); dd = (D.ex_base - D.stress_base).values
    b = dd[rng.integers(0, n, (10000, n))].mean(1)
    out = dict(n=n, stress_base=float(D.stress_base.mean()), ex_base=float(D.ex_base.mean()), diff=[float(dd.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
               stress_base_ci=[float(np.percentile(D.stress_base.values[rng.integers(0, n, (10000, n))].mean(1), q)) for q in (2.5, 97.5)],
               ex_base_ci=[float(np.percentile(D.ex_base.values[rng.integers(0, n, (10000, n))].mean(1), q)) for q in (2.5, 97.5)])
    save('S4b_pretask_baselines', out); log('S4b', out)
