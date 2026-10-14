"""ONE-TIME external test on WESAD, exactly as fixed in improve/m6_prespec.md and improve/wesad_addendum.md.
Usage (from code/): python improve/run_wesad.py --i-understand-this-runs-once"""
import sys, os, json, hashlib, pickle, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
if '--i-understand-this-runs-once' not in sys.argv: raise SystemExit('refusing: pass --i-understand-this-runs-once')
MARK = 'improve/WESAD_OPENED'
if os.path.exists(MARK): raise SystemExit('WESAD test already run on ' + open(MARK).read())
for doc in ('m6_prespec', 'wesad_addendum'):
    h = hashlib.sha256(open(f'improve/{doc}.md', 'rb').read()).hexdigest()
    assert h == open(f'improve/{doc}.sha256').read().split()[0], doc + ' changed since it was hashed'
from run_m6 import CANDS, REF, FSN, strata, participant_metrics, select, add_adaptive, alarm_states, events, score
from run_m5_episode import build_streams
from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, log
from build_features import build
from features_v1 import PHYS, LEVEL, ACC
from sklearn.metrics import roc_auc_score
MASTER = 20261009

# ------------------------------------------------ training side (all 36)
W, FS, ACCF = build(); AD5, AD10 = add_adaptive(W, PHYS, 5), add_adaptive(W, PHYS, 10)
CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
FSETS = {'ref': CAL + DEL + ACC, 'ad5': AD5 + DEL + ACC, 'ad10': AD10 + DEL + ACC, 'refad5': CAL + DEL + AD5 + ACC}
d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
prev = {45: prev_index(d, 45)}; streams = build_streams(d, seg)
for S in streams.values(): S['strat'] = strata(S)
by_pid = {}
for S in streams.values(): by_pid.setdefault(S['pid'], []).append(S)
parts = sorted(d.participant.unique()); assert len(parts) == 36
S_oof, tau = {}, {}
for fs, cols in FSETS.items():
    po = smooth(oof(d, cols, ACCF, 'et', 'single', 'all'), prev, d.index.values, 45)
    S_oof[fs] = np.full(len(d), np.nan); S_oof[fs][d.index.values] = po; tau[fs] = threshold(po[d.y.values == 1], sens=0.80)
inner = {c: [participant_metrics(by_pid[p], S_oof[c[0]], tau[c[0]], c[1], 100, p) for p in parts] for c in CANDS}
best, P = select(inner); log('selected on all 36:', best, {k: round(v, 3) for k, v in P[best].items()}, 'reference', {k: round(v, 3) for k, v in P[REF].items()})
open(MARK, 'w').write(time.strftime('%Y-%m-%dT%H:%M:%S') + ' selected=' + '|'.join(best))

# ------------------------------------------------ WESAD frame
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
assert add_adaptive(Wt, PHYS, 5) == AD5
Wt['y'] = (Wt.block == 'tsst').astype(int); Wt['kind'] = np.where(Wt.y == 1, 'stress', 'rest'); Wt['is_base'] = (Wt.block == 'baseline').astype(int)
Wt['rest_early'] = 0; prevt = {45: prev_index(Wt, 45)}
segw = pd.read_csv('data/segments_wesad.csv')
ST = {}
for s, g in Wt.groupby('participant'):
    g = g.sort_values('start_s'); sr = segw[(segw.participant == s) & (segw.block == 'tsst')].iloc[0]
    S = dict(pid=s, rows=g.index.values, start=g.start_s.values.astype(float), dec=g.end_s.values.astype(float), epi=np.where(g.block.values == 'tsst', 0, -1),
             kind=g.kind.values, base=g.is_base.values.astype(bool), episodes=[(0, 'tsst', float(sr.start_s), float(sr.end_s))])
    S['strat'] = strata(S); ST[s] = S
subs = sorted(ST, key=lambda x: int(x[1:]))

# ------------------------------------------------ apply once: selected and reference detectors
OUTR = {}
for name, c in (('selected', best), ('reference', REF)):
    cols = FSETS[c[0]]
    p = smooth(train_predict(d, Wt, cols, ACCF, 'et', 'single', 'all'), prevt, Wt.index.values, 45)
    rows = []
    for s in subs:
        S = ST[s]; sc = p[S['rows']]
        o = participant_metrics([S], p, tau[c[0]], c[1], 1000, s); o['participant'] = s
        al = alarm_states(S['dec'], sc, tau[c[0]], c[1]); blk = Wt.loc[S['rows'], 'block'].values; ok = S['strat'] != 'grace'
        for b in ('baseline', 'amusement', 'medi1', 'medi2', 'tsst'): o[f'al_{b}'] = int(al[(blk == b) & ok].sum()); o[f'n_{b}'] = int(((blk == b) & ok).sum())
        o['auroc'] = float(roc_auc_score(Wt.loc[S['rows'], 'y'].values, sc)); rows.append(o)
    OUTR[name] = pd.DataFrame(rows); OUTR[name]['config'] = '|'.join(c); OUTR[name]['tau'] = tau[c[0]]
    OUTR[name].to_csv(f'improve/res_WESAD_{name}.csv', index=False)
pickle.dump(dict(best=best, inner=P, tau=tau, res=OUTR), open('improve/wesad_all.pkl', 'wb'))

# ------------------------------------------------ summary
n = len(subs); ix = np.random.default_rng(MASTER).integers(0, n, (10000, n))
def M(df):
    c = {k: df[k].values.astype(float) for k in df.columns if k.startswith(('al_', 'n_', 'det_', 'null_', 'fa_'))}
    return c
def stats(c, sel):
    s = {k: v[sel].sum(-1) for k, v in c.items()}; h = lambda k: s['n_' + k] * 15 / 3600
    return dict(P1=(s['al_rest'] + s['al_base']) / (s['n_rest'] + s['n_base']), det_st=s['det_st'] / s['n_ep'], null_st=s['null_st'] / s['n_ep'],
                excess_st=(s['det_st'] - s['null_st']) / s['n_ep'], det_ov=s['det_ov'] / s['n_ep'], null_ov=s['null_ov'] / s['n_ep'],
                baseline=s['al_baseline'] / s['n_baseline'], amusement=s['al_amusement'] / s['n_amusement'], meditation=(s['al_medi1'] + s['al_medi2']) / (s['n_medi1'] + s['n_medi2']),
                tsst_windows=s['al_tsst'] / s['n_tsst'], fa_h_rest=(s['fa_rest'] + s['fa_base']) / ((s['n_rest'] + s['n_base']) * 15 / 3600))
out = dict(selected_config='|'.join(best), training_inner=P[best], training_inner_reference=P[REF], n=n)
for name in ('selected', 'reference'):
    c = M(OUTR[name]); pt = stats(c, np.arange(n)); bs = stats({k: v[ix] for k, v in c.items()}, slice(None))
    out[name] = {k: [float(pt[k]), float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))] for k in pt}
    a = OUTR[name].auroc.values; b = a[ix].mean(1); out[name]['auroc'] = [float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
cs, cr = M(OUTR['selected']), M(OUTR['reference'])
ps, pr = stats(cs, np.arange(n)), stats(cr, np.arange(n)); bs_, br_ = stats({k: v[ix] for k, v in cs.items()}, slice(None)), stats({k: v[ix] for k, v in cr.items()}, slice(None))
out['diff'] = {k: [float(ps[k] - pr[k]), float(np.percentile(bs_[k] - br_[k], 2.5)), float(np.percentile(bs_[k] - br_[k], 97.5))] for k in ps}
da = OUTR['selected'].auroc.values - OUTR['reference'].auroc.values; b = da[ix].mean(1); out['diff']['auroc'] = [float(da.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
a_ok = out['diff']['P1'][2] < 0; b_ok = out['selected']['excess_st'][1] > 0
out['replication'] = 'replicated' if (a_ok and b_ok) else ('partly replicated' if (a_ok or b_ok) else 'not replicated')
out['replication_parts'] = dict(P1_diff_ci_below_0=bool(a_ok), selected_strict_excess_ci_above_0=bool(b_ok))
json.dump(out, open('improve/sum_WESAD.json', 'w'), indent=1)
f = lambda v: f'{v[0]:.3f} ({v[1]:.3f} to {v[2]:.3f})'
print('selected config', out['selected_config'])
for k in out['selected']: print(f'{k:13s} sel {f(out["selected"][k])}  ref {f(out["reference"][k])}  diff {f(out["diff"][k])}')
print('REPLICATION:', out['replication'], out['replication_parts'])
