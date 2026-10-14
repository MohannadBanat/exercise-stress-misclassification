"""Summarise M6 exactly as pre-specified in improve/m6_prespec.md."""
import pickle, json, numpy as np, pandas as pd
MASTER = 20261009; B = 10000
R = pickle.load(open('improve/m6_all.pkl', 'rb')); n = len(R)
idx = np.random.default_rng(MASTER).integers(0, n, (B, n))
COLS = ['al_rest', 'n_rest', 'al_base', 'n_base', 'al_ex', 'n_ex', 'al_stress', 'n_stress', 'n_ep', 'det_ov', 'det_st', 'null_ov', 'null_st', 'fa_rest', 'fa_base', 'fa_ex']
def mat(rows): return pd.DataFrame(rows)[COLS].astype(float).values
def metrics(S):
    c = {k: S[..., i] for i, k in enumerate(COLS)}; h = lambda k: c['n_' + k] * 15 / 3600
    return dict(P1=(c['al_rest'] + c['al_base']) / (c['n_rest'] + c['n_base']), P2=(c['det_st'] - c['null_st']) / c['n_ep'], P3=c['al_ex'] / c['n_ex'],
                rest_excl_base=c['al_rest'] / c['n_rest'], base=c['al_base'] / c['n_base'], stress_windows=c['al_stress'] / c['n_stress'],
                det_ov=c['det_ov'] / c['n_ep'], null_ov=c['null_ov'] / c['n_ep'], det_st=c['det_st'] / c['n_ep'], null_st=c['null_st'] / c['n_ep'],
                excess_ov=(c['det_ov'] - c['null_ov']) / c['n_ep'], fa_h_rest=c['fa_rest'] / h('rest'), fa_h_base=c['fa_base'] / h('base'), fa_h_ex=c['fa_ex'] / h('ex'))
def boot(X):
    p, b = metrics(X.sum(0)), metrics(X[idx].sum(1))
    return {k: [float(p[k]), float(np.percentile(b[k], 2.5)), float(np.percentile(b[k], 97.5))] for k in p}
def paired(Xa, Xb):
    p = {k: metrics(Xa.sum(0))[k] - metrics(Xb.sum(0))[k] for k in metrics(Xa.sum(0))}
    ba, bb = metrics(Xa[idx].sum(1)), metrics(Xb[idx].sum(1))
    return {k: [float(p[k]), float(np.percentile(ba[k] - bb[k], 2.5)), float(np.percentile(ba[k] - bb[k], 97.5))] for k in p}
CANDS = list(R[0]['outer'].keys())
X = {c: mat([r['outer'][c] for r in R]) for c in CANDS}
Xsel = mat([r['outer']['|'.join(r['best'])] for r in R]); Xref = X['ref|state']
out = dict(n=n, selected_counts=pd.Series(['|'.join(r['best']) for r in R]).value_counts().to_dict())
out['reference'] = boot(Xref); out['selected'] = boot(Xsel); out['diff_selected_minus_reference'] = d = paired(Xsel, Xref)
out['decision'] = dict(a_P1_ci_below_0=bool(d['P1'][2] < 0), b_P2_noninferior=bool(d['P2'][1] > -0.05), c_P3_guard=bool(d['P3'][2] < 0.05))
out['decision']['claim_allowed'] = all(out['decision'].values())
out['candidates_descriptive'] = {c: boot(X[c]) for c in CANDS}
out['candidates_diff_vs_reference_descriptive'] = {c: paired(X[c], Xref) for c in CANDS if c != 'ref|state'}
A = pd.DataFrame([r['auc'] for r in R]); rng = np.random.default_rng(MASTER)
def bci(v):
    v = np.asarray(v, float); b = v[rng.integers(0, len(v), (B, len(v)))].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out['auroc_within_session'] = {k: bci(A[k]) for k in A}; out['auroc_diff_vs_ref'] = {k: bci(A[k] - A['ref']) for k in A if k != 'ref'}
json.dump(out, open('improve/sum_M6.json', 'w'), indent=1)
f = lambda v: f'{v[0]:.3f} ({v[1]:.3f} to {v[2]:.3f})'
print('selected', out['selected_counts'])
for k in ('P1', 'P2', 'P3', 'rest_excl_base', 'base', 'stress_windows', 'det_st', 'null_st', 'det_ov', 'null_ov', 'fa_h_rest', 'fa_h_base', 'fa_h_ex'):
    print(f'{k:15s} ref {f(out["reference"][k])}   sel {f(out["selected"][k])}   diff {f(d[k])}')
print('decision', out['decision'])
print('AUROC', {k: f(v) for k, v in out['auroc_within_session'].items()}, {k: f(v) for k, v in out['auroc_diff_vs_ref'].items()})
print(pd.DataFrame({c: {k: round(v[0], 3) for k, v in out['candidates_descriptive'][c].items() if k in ('P1', 'P2', 'P3', 'det_st', 'stress_windows', 'fa_h_rest')} for c in CANDS}).T.to_string())
