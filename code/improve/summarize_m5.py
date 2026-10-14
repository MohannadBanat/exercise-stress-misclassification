"""Summarise M5 (episode-level evaluation) exactly as pre-specified in improve/episode_prespec.md."""
import pickle, json, numpy as np, pandas as pd
MASTER = 20261009; B = 10000
R = pickle.load(open('improve/m5_all.pkl', 'rb')); pids = [r['pid'] for r in R]; n = len(R)
idx = np.random.default_rng(MASTER).integers(0, n, (B, n))
COLS = ['n_ep', 'det', 'fa_rest', 'fa_ex', 'fa_base', 'h_rest', 'h_ex', 'h_base', 'al_rest', 'n_rest']

def frame(rows):
    df = pd.DataFrame(rows)
    for c in COLS: df[c] = df[c].astype(float)
    return df

def metrics(M):
    """M: (..., n_participants, len(COLS)) summed over participants -> dict of arrays."""
    s = {c: M[..., i] for i, c in enumerate(COLS)}
    fr, fe = s['fa_rest'] / s['h_rest'], s['fa_ex'] / s['h_ex']
    return dict(sens=s['det'] / s['n_ep'], fa_rest=fr, fa_ex=fe, fa_comb=(fr + fe) / 2, fa_base=s['fa_base'] / s['h_base'], alarm_rest=s['al_rest'] / s['n_rest'])

def boot(df):
    X = df[COLS].values; point = metrics(X.sum(0)); bs = metrics(X[idx].sum(1))
    return {k: [float(point[k]), float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))] for k in point}, bs

def paired(dfa, dfb):
    Xa, Xb = dfa[COLS].values, dfb[COLS].values
    pa, pb = metrics(Xa.sum(0)), metrics(Xb.sum(0)); ba, bb = metrics(Xa[idx].sum(1)), metrics(Xb[idx].sum(1))
    return {k: [float(pa[k] - pb[k]), float(np.percentile(ba[k] - bb[k], 2.5)), float(np.percentile(ba[k] - bb[k], 97.5))] for k in pa}

def extra(df):
    lat = np.concatenate([np.asarray(l, float) for l in df.lat if len(l)]) if df.lat.map(len).sum() else np.array([np.nan])
    types = {}
    for t in ('stroop', 'tmct', 'opinion_cluster', 'subtract'):
        if f'dt_{t}_n' in df: types[t] = [int(df[f'dt_{t}'].fillna(0).sum()), int(df[f'dt_{t}_n'].fillna(0).sum())]
    pp = df.assign(r=df.fa_rest / df.h_rest, e=df.fa_ex / df.h_ex.replace(0, np.nan))
    return dict(latency_median=float(np.median(lat)), latency_q1=float(np.percentile(lat, 25)), latency_q3=float(np.percentile(lat, 75)),
                by_type=types, perparticipant_mean_fa_rest=float(pp.r.mean()), perparticipant_mean_fa_ex=float(pp.e.mean()),
                n_participants_ex=int((df.h_ex > 0).sum()), hours_rest=float(df.h_rest.sum()), hours_ex=float(df.h_ex.sum()), hours_base=float(df.h_base.sum()),
                n_false_rest=int(df.fa_rest.sum()), n_false_ex=int(df.fa_ex.sum()), n_false_base=int(df.fa_base.sum()), n_detected=int(df.det.sum()), n_episodes=int(df.n_ep.sum()))

sel_rules = [r['best'] for r in R]
sel = frame([r['grid'][r['best']] for r in R]); ref = frame([r['grid'][(0.8, 1)] for r in R])
out = dict(n=n, selected_rules=pd.Series([f'q={q}, N={N}' for q, N in sel_rules]).value_counts().to_dict())
out['selected'], _ = boot(sel); out['reference'], _ = boot(ref)
out['selected_extra'], out['reference_extra'] = extra(sel), extra(ref)
out['diff_selected_minus_reference'] = paired(sel, ref)
d = out['diff_selected_minus_reference']
out['decision'] = dict(fa_comb_ci_below_0=bool(d['fa_comb'][2] < 0), sens_noninferior_margin_0_10=bool(d['sens'][1] > -0.10),
                       claim_allowed=bool(d['fa_comb'][2] < 0 and d['sens'][1] > -0.10))
for v in ('G0', 'strict'):
    for nm in ('selected', 'reference'):
        df = frame([r['sa'][f'{nm}_{v}'] for r in R]); out[f'{nm}_{v}'], _ = boot(df); out[f'{nm}_{v}_extra'] = extra(df)
# descriptive operating curve on outer predictions (not used for any choice)
curve = []
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    for N in (1, 2, 3, 4, 6, 8):
        df = frame([r['grid'][(q, N)] for r in R]); m = metrics(df[COLS].values.sum(0))
        curve.append(dict(q=q, N=N, **{k: float(v) for k, v in m.items()}, latency_median=extra(df)['latency_median']))
out['curve_descriptive'] = curve
# inner (training) view: how often did any rule with N > 1 reach the sensitivity floor, and selection margins
inner = pd.DataFrame([{k: v['sens'] for k, v in r['inner'].items()} for r in R])
out['inner_share_rules_meeting_floor_by_N'] = {N: float((inner[[c for c in inner if c.endswith(f'|{N}')]] >= 0.90).any(axis=1).mean()) for N in (1, 2, 3, 4, 6, 8)}
out['inner_mean'] = {k: dict(sens=float(np.mean([r['inner'][k]['sens'] for r in R])), fa_comb=float(np.mean([r['inner'][k]['fa_comb'] for r in R]))) for k in R[0]['inner']}
out['taus_mean'] = {str(q): float(np.mean([r['taus'][q] for r in R])) for q in (0.5, 0.6, 0.7, 0.8, 0.9)}
json.dump(out, open('improve/sum_M5.json', 'w'), indent=1)
sel.assign(rule=[f'{q}|{N}' for q, N in sel_rules]).drop(columns='lat').to_csv('improve/res_M5_selected.csv', index=False)
ref.drop(columns='lat').to_csv('improve/res_M5_reference.csv', index=False)
pr = lambda x: f'{x[0]:.2f} ({x[1]:.2f}–{x[2]:.2f})'
print('selected rules', out['selected_rules'])
for nm in ('selected', 'reference', 'selected_G0', 'reference_G0', 'selected_strict', 'reference_strict'):
    print(f'{nm:18s}', {k: pr(v) for k, v in out[nm].items()})
print('diff', {k: pr(v) for k, v in d.items()}); print('decision', out['decision'])
print('sel extra', out['selected_extra']); print('ref extra', out['reference_extra'])
print(pd.DataFrame(curve).round(3).to_string())
print('inner floor by N', out['inner_share_rules_meeting_floor_by_N'])
