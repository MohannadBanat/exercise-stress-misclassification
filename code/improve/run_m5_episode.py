"""M5: episode-level evaluation of the final stress detector, exactly as fixed in improve/episode_prespec.md
(SHA-256 recorded in improve/episode_prespec.sha256). Exploratory; nested LOPO on all 36 participants.
Usage: python run_m5_episode.py [--test]   (--test runs the alarm-logic unit checks only)"""
import sys, os, json, hashlib, pickle, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')

G_PRIMARY, MERGE_S, ADJ_S, STEP_S = 60.0, 120.0, 45.0, 15.0
QS, NS = (0.5, 0.6, 0.7, 0.8, 0.9), (1, 2, 3, 4, 6, 8)
REF_RULE = (0.8, 1)
SENS_MIN = 0.90
OUT = 'improve/m5_folds'

# ------------------------------------------------------------------ alarm logic (one session stream)
def events_for_stream(start, dec, epi, above, N):
    """start/dec: window start and decision (end) times, sorted; epi: episode id (-1 if none); above: s >= tau.
    Returns (alarm_state bool array, list of events [first_idx, last_idx, first_dec, last_dec])."""
    n = len(start); c = np.zeros(n, int); alarm = np.zeros(n, bool)
    for i in range(n):
        if not above[i]: continue
        adj = i > 0 and ((start[i] - start[i - 1] <= ADJ_S) or (epi[i] >= 0 and epi[i] == epi[i - 1]))
        c[i] = c[i - 1] + 1 if (adj and c[i - 1] > 0) else 1
        alarm[i] = c[i] >= N
    runs = []
    for i in np.flatnonzero(alarm):
        adj = i > 0 and alarm[i - 1] and ((start[i] - start[i - 1] <= ADJ_S) or (epi[i] >= 0 and epi[i] == epi[i - 1]))
        if runs and adj: runs[-1][1] = i
        else: runs.append([i, i])
    ev = []
    for a, b in runs:
        if ev and dec[a] - ev[-1][3] < MERGE_S: ev[-1][1], ev[-1][3] = b, dec[b]
        else: ev.append([a, b, dec[a], dec[b]])
    return alarm, ev

def score_stream(S, scores, tau, N, G, strict):
    """S: precomputed stream dict for one session; returns counts dict."""
    above = scores >= tau
    alarm, ev = events_for_stream(S['start'], S['dec'], S['epi'], above, N)
    eps = S['episodes']  # list of (ep_id, type, start, end)
    dec = S['dec']
    grace = np.zeros(len(dec), bool)
    for _, _, a, b in eps: grace |= (dec > b) & (dec <= b + G) & (S['kind'] == 'rest')
    stratum = np.where(S['kind'] == 'exercise', 'ex', np.where(S['kind'] == 'stress', 'stress', np.where(S['base'], 'base', np.where(grace, 'grace', 'rest'))))
    out = dict(n_ep=len(eps), det=0, fa_rest=0, fa_ex=0, fa_base=0,
               h_rest=(stratum == 'rest').sum() * STEP_S / 3600, h_ex=(stratum == 'ex').sum() * STEP_S / 3600, h_base=(stratum == 'base').sum() * STEP_S / 3600,
               al_rest=int(alarm[stratum == 'rest'].sum()), n_rest=int((stratum == 'rest').sum()), lat=[], det_type={})
    hit_ev = np.zeros(len(ev), bool)
    for eid, typ, a, b in eps:
        if strict: ov = [k for k, e in enumerate(ev) if a <= e[2] <= b + G]
        else: ov = [k for k, e in enumerate(ev) if e[2] <= b + G and e[3] >= a]
        d = len(ov) > 0; out['det'] += d; out['det_type'][typ] = out['det_type'].get(typ, 0) + d
        out['det_type'][typ + '_n'] = out['det_type'].get(typ + '_n', 0) + 1
        if d:
            hit_ev[ov] = True; out['lat'].append(min(max(ev[k][2], a) for k in ov) - a)
    for k, e in enumerate(ev):
        if hit_ev[k]: continue
        if strict and any(a <= e[2] <= b + G for _, _, a, b in eps): continue
        s = stratum[e[0]]
        if s == 'rest': out['fa_rest'] += 1
        elif s == 'ex': out['fa_ex'] += 1
        elif s == 'base': out['fa_base'] += 1
        else: raise RuntimeError(f'unexpected false-alarm stratum {s}')
    return out

# ------------------------------------------------------------------ streams
def build_streams(d, seg):
    streams = {}
    for (pr, se), g in d.groupby(['protocol', 'session']):
        g = g.sort_values('start_s'); pid = g.participant.iloc[0]
        epi = np.full(len(g), -1); eps = []
        if pr == 'STRESS':
            sg = seg[(seg.protocol == 'STRESS') & (seg.participant == pid)].sort_values('start_s')
            cur = []
            for _, r in sg.iterrows():
                if r.block in ('stroop', 'tmct', 'real_opinion', 'opposite_opinion', 'subtract'): cur.append(r)
                elif cur: eps.append(cur); cur = []
            if cur: eps.append(cur)
            E = []
            for k, blocks in enumerate(eps):
                names = [b.block for b in blocks]
                typ = 'stroop' if names == ['stroop'] else 'tmct' if names == ['tmct'] else 'subtract' if names == ['subtract'] else 'opinion_cluster'
                a, b = float(blocks[0].start_s), float(blocks[-1].end_s)
                m = g.block.isin(names).values & (g.kind.values == 'stress'); epi[m] = k
                assert m.sum() >= 1, (pid, names)
                E.append((k, typ, a, b))
            eps = E
            assert len(eps) == 3, (pid, eps)
        streams[(pr, se)] = dict(pid=pid, rows=g.index.values, start=g.start_s.values.astype(float), dec=g.end_s.values.astype(float),
                                 epi=epi, kind=g.kind.values, base=g.is_base.values.astype(bool), episodes=eps)
    return streams

def evaluate(streams, pids, Sfull, tau, N, G=G_PRIMARY, strict=False):
    """per-participant counts for participants pids, scores taken from Sfull (array over all rows of d)."""
    acc = {}
    for key, S in streams.items():
        if S['pid'] not in pids: continue
        o = score_stream(S, Sfull[S['rows']], tau, N, G, strict); a = acc.setdefault(S['pid'], dict(participant=S['pid']))
        for k, v in o.items():
            if k == 'lat': a['lat'] = a.get('lat', []) + v
            elif k == 'det_type':
                for t, x in v.items(): a['dt_' + t] = a.get('dt_' + t, 0) + x
            else: a[k] = a.get(k, 0) + v
    return pd.DataFrame(list(acc.values()))

def pooled(df):
    s = df[['n_ep', 'det', 'fa_rest', 'fa_ex', 'fa_base', 'h_rest', 'h_ex', 'h_base']].sum()
    fr, fe = s.fa_rest / s.h_rest, s.fa_ex / s.h_ex
    return dict(sens=s.det / s.n_ep, fa_rest=fr, fa_ex=fe, fa_comb=(fr + fe) / 2, fa_base=s.fa_base / s.h_base)

def select(df_by_rule):
    """df_by_rule: {(q, N): per-participant df on training OOF}. Pre-specified objective."""
    P = {r: pooled(df) for r, df in df_by_rule.items()}
    ok = [r for r in P if P[r]['sens'] >= SENS_MIN]
    if ok: best = min(ok, key=lambda r: (P[r]['fa_comb'], r[1], -r[0]))
    else: best = max(P, key=lambda r: (P[r]['sens'], -P[r]['fa_comb'], -r[1]))
    return best, P

# ------------------------------------------------------------------ unit checks for the alarm logic
def unit_tests():
    st = np.arange(12) * 15.0; de = st + 30; ep = np.full(12, -1)
    al, ev = events_for_stream(st, de, ep, np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], bool), 2)
    assert al.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], al
    assert len(ev) == 2 and ev[0][:2] == [2, 3] and ev[1][:2] == [11, 11], ev       # 195 - 75 = 120 s: not merged (needs < 120)
    al, ev = events_for_stream(st, de, ep, np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool), 1)
    assert len(ev) == 1 and ev[0][2] == 30 and ev[0][3] == 90, ev                  # gaps < 120 s merge
    st2 = np.array([0, 15, 100, 200, 215.]); ep2 = np.array([-1, 0, 0, -1, -1])
    al, ev = events_for_stream(st2, st2 + 30, ep2, np.ones(5, bool), 3)
    assert al.tolist() == [0, 0, 1, 0, 0], al                                     # same-episode gap bridged; 100 -> 200 resets
    S = dict(start=np.array([0, 15, 30, 45, 60, 75.]), dec=np.array([30, 45, 60, 75, 90, 105.]), epi=np.array([0, 0, -1, -1, -1, -1]),
             kind=np.array(['stress', 'stress', 'rest', 'rest', 'rest', 'rest']), base=np.zeros(6, bool), episodes=[(0, 'tmct', 0., 45.)])
    o = score_stream(S, np.array([0, 0, 0, 0, 0, 1.]), 0.5, 1, 30, False)          # grace = decisions 60, 75; alarm at 105 -> FA
    assert o['det'] == 0 and o['fa_rest'] == 1 and o['n_rest'] == 2, o
    o = score_stream(S, np.array([0, 0, 0, 0, 0, 1.]), 0.5, 1, 60, False)          # with G = 60 the same alarm is in grace -> hit
    assert o['det'] == 1 and o['fa_rest'] == 0 and o['n_rest'] == 0, o
    S2 = dict(start=np.array([0, 40, 55.]), dec=np.array([30, 70, 85.]), epi=np.array([-1, 0, 0]), kind=np.array(['rest', 'stress', 'stress']),
              base=np.zeros(3, bool), episodes=[(0, 'tmct', 40., 85.)])
    o = score_stream(S2, np.ones(3), 0.5, 1, 0, False); assert o['det'] == 1 and o['fa_rest'] == 0 and o['lat'] == [0], o
    o = score_stream(S2, np.ones(3), 0.5, 1, 0, True); assert o['det'] == 0 and o['fa_rest'] == 1, o   # strict: onset before episode
    print('unit tests passed')

if __name__ == '__main__':
    if '--test' in sys.argv: unit_tests(); raise SystemExit
    unit_tests()
    from framework2 import *; from build_features import build
    h = hashlib.sha256(open('improve/episode_prespec.md', 'rb').read()).hexdigest()
    assert h == open('improve/episode_prespec.sha256').read().split()[0], 'pre-specification changed since it was hashed'
    W, FS, ACC = build(); d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
    prev = {45: prev_index(d, 45)}; streams = build_streams(d, seg); feats = FS['v1_cal_ctx']
    parts = sorted(d.participant.unique()); os.makedirs(OUT, exist_ok=True)
    log('participants', len(parts), 'episodes', sum(len(S['episodes']) for S in streams.values()))

    def one(pid):
        f = f'{OUT}/{pid}.pkl'
        if os.path.exists(f): return pickle.load(open(f, 'rb'))
        tr, te = d[d.participant != pid], d[d.participant == pid]
        p_oof = smooth(oof(tr, feats, ACC, 'et', 'single', 'all'), prev, tr.index.values, 45)
        S_tr = np.full(len(d), np.nan); S_tr[tr.index.values] = p_oof
        taus = {q: threshold(p_oof[tr.y.values == 1], sens=q) for q in QS}
        trp = set(tr.participant.unique())
        inner = {(q, N): evaluate(streams, trp, S_tr, taus[q], N) for q in QS for N in NS}
        best, P = select(inner)
        p_te = smooth(train_predict(tr, te, feats, ACC, 'et', 'single', 'all'), prev, te.index.values, 45)
        S_te = np.full(len(d), np.nan); S_te[te.index.values] = p_te
        grid = {(q, N): evaluate(streams, {pid}, S_te, taus[q], N).iloc[0].to_dict() for q in QS for N in NS}
        sa = {}
        for name, (q, N) in (('selected', best), ('reference', REF_RULE)):
            sa[name + '_G0'] = evaluate(streams, {pid}, S_te, taus[q], N, G=0.0).iloc[0].to_dict()
            sa[name + '_strict'] = evaluate(streams, {pid}, S_te, taus[q], N, strict=True).iloc[0].to_dict()
        r = dict(pid=pid, best=best, taus=taus, inner={f'{q}|{N}': v for (q, N), v in P.items()}, grid=grid, sa=sa)
        pickle.dump(r, open(f, 'wb')); log('done', pid, 'selected', best, 'inner', {k: round(v, 3) for k, v in P[best].items()})
        return r

    from joblib import Parallel, delayed
    R = Parallel(n_jobs=2)(delayed(one)(pid) for pid in parts)
    pickle.dump(R, open('improve/m5_all.pkl', 'wb'))
    # multiplicity record
    inner = pd.DataFrame([{k: v['fa_comb'] for k, v in r['inner'].items()} for r in R]).mean()
    sens_in = pd.DataFrame([{k: v['sens'] for k, v in r['inner'].items()} for r in R]).mean()
    lg = pd.DataFrame({'tag': 'M5_episode', 'candidate': [json.dumps(dict(q=float(k.split('|')[0]), N=int(k.split('|')[1]))) for k in inner.index],
                       'mean_inner_score': inner.values, 'time': time.strftime('%Y-%m-%d %H:%M'), 'score': 'FA/h_comb (inner; sens ' + sens_in.round(3).astype(str).values + ')'})
    lg.to_csv('improve/trial_log.csv', mode='a', header=False, index=False)
    log('all folds done; run summarize_m5.py')
