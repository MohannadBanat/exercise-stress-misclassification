"""M6: reducing false alarms at rest, exactly as fixed in improve/m6_prespec.md (SHA-256 in improve/m6_prespec.sha256).
Exploratory; nested LOPO on all 36 participants. Usage: python improve/run_m6.py [--test]"""
import sys, os, json, hashlib, pickle, time, zlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from run_m5_episode import build_streams, G_PRIMARY, MERGE_S, ADJ_S, STEP_S
OUT = 'improve/m6_folds'; MASTER = 20261009
FSN = ('ref', 'ad5', 'ad10', 'refad5'); LOGIC = ('state', 'hold2', 'hold4'); HOLD = {'hold2': 120.0, 'hold4': 240.0}
CANDS = [(f, l) for f in FSN for l in LOGIC]; REF = ('ref', 'state')
N_IN, N_OUT = 100, 1000

# ------------------------------------------------------------------ alarm logic
def alarm_states(dec, s, tau, logic):
    above = s >= tau
    if logic == 'state': return above.copy()
    H = HOLD[logic]; alarm = np.zeros(len(s), bool); armed, until = True, -np.inf
    for i in range(len(s)):
        if dec[i] <= until:
            alarm[i] = True
            if not above[i]: armed = True
            continue
        if above[i] and armed: alarm[i] = True; until = dec[i] + H; armed = False
        elif not above[i]: armed = True
    return alarm

def events(start, dec, epi, alarm):
    runs = []
    for i in np.flatnonzero(alarm):
        adj = i > 0 and alarm[i - 1] and ((start[i] - start[i - 1] <= ADJ_S) or (epi[i] >= 0 and epi[i] == epi[i - 1]))
        if runs and adj: runs[-1][1] = i
        else: runs.append([i, i])
    ev = []
    for a, b in runs:
        if ev and dec[a] - ev[-1][3] < MERGE_S: ev[-1][1], ev[-1][3] = b, dec[b]
        else: ev.append([a, b, dec[a], dec[b]])
    return ev

def strata(S, G=G_PRIMARY):
    dec, kind = S['dec'], S['kind']; grace = np.zeros(len(dec), bool)
    for _, _, a, b in S['episodes']: grace |= (dec > b) & (dec <= b + G) & (kind == 'rest')
    return np.where(kind == 'exercise', 'ex', np.where(kind == 'stress', 'stress', np.where(S['base'], 'base', np.where(grace, 'grace', 'rest'))))

def detect(S, ev, G=G_PRIMARY):
    ov = st = 0
    for _, _, a, b in S['episodes']:
        ov += any(e[2] <= b + G and e[3] >= a for e in ev); st += any(a <= e[2] <= b + G for e in ev)
    return ov, st

def score(S, s, tau, logic, strat):
    al = alarm_states(S['dec'], s, tau, logic); ev = events(S['start'], S['dec'], S['epi'], al)
    o = {f'al_{k}': int(al[strat == k].sum()) for k in ('rest', 'base', 'ex', 'stress')}
    o.update({f'n_{k}': int((strat == k).sum()) for k in ('rest', 'base', 'ex', 'stress')})
    ov, st = detect(S, ev); o.update(n_ep=len(S['episodes']), det_ov=ov, det_st=st, fa_rest=0, fa_base=0, fa_ex=0)
    for e in ev:
        if any(e[2] <= b + G_PRIMARY and e[3] >= a for _, _, a, b in S['episodes']): continue
        k = strat[e[0]]
        if k in ('rest', 'base', 'ex'): o['fa_' + k] += 1
    return o

def null_det(S, s, tau, logic, n, seed):
    rng = np.random.default_rng(seed); ov = st = 0.0
    for _ in range(n):
        sh = np.roll(s, rng.integers(1, len(s))); ev = events(S['start'], S['dec'], S['epi'], alarm_states(S['dec'], sh, tau, logic))
        a, b = detect(S, ev); ov += a; st += b
    return ov / n, st / n

def participant_metrics(streams_p, Sfull, tau, logic, n_null, pid):
    acc = None
    for S in streams_p:
        s = Sfull[S['rows']]; o = score(S, s, tau, logic, S['strat'])
        if S['episodes']:
            o['null_ov'], o['null_st'] = null_det(S, s, tau, logic, n_null, zlib.crc32(pid.encode()) + MASTER)
        else: o['null_ov'] = o['null_st'] = 0.0
        acc = o if acc is None else {k: acc[k] + o[k] for k in acc}
    return acc

def pooled(rows):
    T = pd.DataFrame(rows).sum(numeric_only=True)
    return dict(P1=(T.al_rest + T.al_base) / (T.n_rest + T.n_base), P2=(T.det_st - T.null_st) / T.n_ep, P3=T.al_ex / T.n_ex)

def select(inner):
    """inner: {cand: list of per-participant dicts}. Pre-specified rule."""
    P = {c: pooled(v) for c, v in inner.items()}; r = P[REF]
    ok = [c for c in CANDS if P[c]['P2'] >= r['P2'] - 0.05 and P[c]['P3'] <= r['P3'] + 0.02]
    return min(ok, key=lambda c: (P[c]['P1'], CANDS.index(c))), P

# ------------------------------------------------------------------ adaptive features
def add_adaptive(W, cols, minutes):
    Wsec = minutes * 60.0; X = W[cols].values.astype(float); A = np.full(X.shape, np.nan)
    for _, g in W.groupby(['protocol', 'session'], sort=False):
        idx = g.index.values; t = g.start_s.values
        for j, i in enumerate(idx):
            m = idx[(t >= t[j] - Wsec) & (t <= t[j] - 30)]
            if len(m) >= 4: A[i] = X[i] - np.nanmedian(X[m], axis=0)
    names = [f'ad{minutes}_{c}' for c in cols]
    for k, nme in enumerate(names): W[nme] = A[:, k]
    return names

def unit_tests():
    dec = np.arange(1, 21) * 15.0 + 30; s = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1], float)
    a = alarm_states(dec, s, 0.5, 'hold2')
    # onset at i=1 (dec 60), hold until 180 -> i=1..9; i=10,11 above but disarmed -> off; i=12 below (re-arm); i=13 onset, hold to dec 255+120=375 -> i=13..19
    assert a.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], a.astype(int).tolist()
    assert alarm_states(dec, s, 0.5, 'state').tolist() == (s >= 0.5).tolist()
    s2 = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], float)   # dip during hold re-arms; new onset right after hold ends
    a2 = alarm_states(dec, s2, 0.5, 'hold2'); assert a2[:10].all() and a2[9] and a2[10], a2.astype(int).tolist()
    print('unit tests passed')

if __name__ == '__main__':
    unit_tests()
    if '--test' in sys.argv: raise SystemExit
    from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict, oof, threshold, log
    from build_features import build
    from features_v1 import PHYS, LEVEL, ACC
    from sklearn.metrics import roc_auc_score
    h = hashlib.sha256(open('improve/m6_prespec.md', 'rb').read()).hexdigest()
    assert h == open('improve/m6_prespec.sha256').read().split()[0], 'pre-specification changed since it was hashed'
    W, FS, ACCF = build()
    AD5, AD10 = add_adaptive(W, PHYS, 5), add_adaptive(W, PHYS, 10)
    CAL = ['cal_' + c for c in PHYS]; DEL = ['d_' + c for c in LEVEL]
    FSETS = {'ref': CAL + DEL + ACC, 'ad5': AD5 + DEL + ACC, 'ad10': AD10 + DEL + ACC, 'refad5': CAL + DEL + AD5 + ACC}
    assert FSETS['ref'] == FS['v1_cal_ctx']
    d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
    prev = {45: prev_index(d, 45)}; streams = build_streams(d, seg)
    for S in streams.values(): S['strat'] = strata(S)
    by_pid = {}
    for S in streams.values(): by_pid.setdefault(S['pid'], []).append(S)
    parts = sorted(d.participant.unique()); os.makedirs(OUT, exist_ok=True)
    log('participants', len(parts), 'feature sets', {k: len(v) for k, v in FSETS.items()})

    def one(pid):
        f = f'{OUT}/{pid}.pkl'
        if os.path.exists(f): return pickle.load(open(f, 'rb'))
        tr, te = d[d.participant != pid], d[d.participant == pid]; trp = sorted(tr.participant.unique())
        S_in, S_te, tau, auc = {}, {}, {}, {}
        for fs, cols in FSETS.items():
            po = smooth(oof(tr, cols, ACCF, 'et', 'single', 'all'), prev, tr.index.values, 45)
            S_in[fs] = np.full(len(d), np.nan); S_in[fs][tr.index.values] = po; tau[fs] = threshold(po[tr.y.values == 1], sens=0.80)
            pt = smooth(train_predict(tr, te, cols, ACCF, 'et', 'single', 'all'), prev, te.index.values, 45)
            S_te[fs] = np.full(len(d), np.nan); S_te[fs][te.index.values] = pt
            st = te[te.protocol == 'STRESS']; auc[fs] = float(roc_auc_score(st.y.values, S_te[fs][st.index.values]))
        inner = {c: [participant_metrics(by_pid[p], S_in[c[0]], tau[c[0]], c[1], N_IN, p) for p in trp] for c in CANDS}
        best, P = select(inner)
        outer = {c: participant_metrics(by_pid[pid], S_te[c[0]], tau[c[0]], c[1], N_OUT, pid) for c in CANDS}
        r = dict(pid=pid, best=best, inner={f'{a}|{b}': v for (a, b), v in P.items()}, outer={f'{a}|{b}': v for (a, b), v in outer.items()}, tau=tau, auc=auc)
        pickle.dump(r, open(f, 'wb')); log('done', pid, 'selected', best, {k: round(v, 3) for k, v in P[best].items()}, 'ref', {k: round(v, 3) for k, v in P[REF].items()})
        return r

    from joblib import Parallel, delayed
    R = Parallel(n_jobs=2)(delayed(one)(pid) for pid in parts)
    pickle.dump(R, open('improve/m6_all.pkl', 'wb'))
    inner = pd.DataFrame([{k: v['P1'] for k, v in r['inner'].items()} for r in R]).mean()
    p2 = pd.DataFrame([{k: v['P2'] for k, v in r['inner'].items()} for r in R]).mean(); p3 = pd.DataFrame([{k: v['P3'] for k, v in r['inner'].items()} for r in R]).mean()
    lg = pd.DataFrame({'tag': 'M6_rest', 'candidate': [json.dumps(dict(fs=k.split('|')[0], logic=k.split('|')[1])) for k in inner.index], 'mean_inner_score': inner.values,
                       'time': time.strftime('%Y-%m-%d %H:%M'), 'score': [f'P1 rest time in alarm (inner; P2 {p2[k]:.3f}, P3 {p3[k]:.3f})' for k in inner.index]})
    lg.to_csv('improve/trial_log.csv', mode='a', header=False, index=False)
    log('all folds done; run summarize_m6.py')
