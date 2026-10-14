"""Compare the outputs of a fresh run (current directory, made by run_all.sh) with the archived outputs in the repository.
Numbers must agree to 1e-9; timestamps are ignored. Usage (from the work directory): python verify.py <repository root>"""
import os, sys, json, glob, numpy as np, pandas as pd
ROOT = sys.argv[1]
P = [(f'data/{n}.csv', f'segment_index/{n}.csv') for n in ('segments', 'session_times', 'concurrent_groups', 'same_day_pairs', 'tag_qc', 'inventory')]
P += [('improve/lockbox.json', 'prespecifications/lockbox.json'), ('improve/trial_log.csv', 'logs/trial_log.csv'), ('build/vals.json', 'results/vals.json')]
P += [(f'results_real/{os.path.basename(f)}', f'results/registered/{os.path.basename(f)}') for f in sorted(glob.glob(f'{ROOT}/results/registered/*'))]
P += [(f'results_x/{os.path.basename(f)}', f'results/exploratory/{os.path.basename(f)}') for f in sorted(glob.glob(f'{ROOT}/results/exploratory/*'))]
P += [(f'improve/{os.path.basename(f)}', f'results/detector/{os.path.basename(f)}') for f in sorted(glob.glob(f'{ROOT}/results/detector/*'))
      if os.path.basename(f) != 'diagnosis.json']
P += [('improve/review/sum_review.json', 'results/controls/sum_review.json')]
if os.path.exists('wesad_e4'): P += [('data/segments_wesad.csv', 'segment_index/segments_wesad.csv'), ('wesad_e4/sync_wesad.csv', 'segment_index/sync_wesad.csv')]
else: P = [p for p in P if 'WESAD' not in p[0]]
def flat(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from flat(v, f'{p}[{i}]')
    else: yield p, x
def cmp_json(a, b):
    fa, fb = dict(flat(json.load(open(a)))), dict(flat(json.load(open(b)))); out = []
    for k in sorted((set(fa) | set(fb))):
        if k.endswith(('.time', 'time', '.sealed', 'mode')): continue
        x, y = fa.get(k, '<missing>'), fb.get(k, '<missing>')
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not (np.isclose(x, y, rtol=0, atol=1e-9) or (np.isnan(x) and np.isnan(y))): out.append((k, x, y))
        elif x != y: out.append((k, x, y))
    return out
def cmp_csv(a, b):
    x, y = pd.read_csv(a), pd.read_csv(b)
    if 'time' in x.columns: x, y = x.drop(columns='time'), y.drop(columns='time')
    if x.shape != y.shape: return [('shape', x.shape, y.shape)]
    num = x.select_dtypes('number').columns; out = []
    if len(num):
        d = (x[num].astype(float) - y[num].astype(float)).abs().max(); out += [(c, f'max |diff| {v:.3g}') for c, v in d[d > 1e-9].items()]
    out += [(c, 'text differs') for c in x.columns.difference(num) if not x[c].fillna('').astype(str).equals(y[c].fillna('').astype(str))]
    return out
bad = 0
for a, b in P:
    if not os.path.exists(a): print(f'{"MISSING IN RUN":16s} {a}'); bad += 1; continue
    d = cmp_json(a, f'{ROOT}/{b}') if a.endswith('.json') else cmp_csv(a, f'{ROOT}/{b}')
    print(f'{"IDENTICAL" if not d else str(len(d)) + " differences":16s} {b}' + ('' if not d else f'\n{"":16s}   {d[:5]}')); bad += bool(d)
v = json.load(open(f'{ROOT}/results/vals.json')); print(f'\n{len(P)} files compared, {bad} not identical; manuscript values archived: {len(v)}')
sys.exit(1 if bad else 0)
