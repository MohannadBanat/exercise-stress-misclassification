"""WESAD window features with the UNCHANGED feature functions of pipeline/features.py (executed, not re-implemented).
Usage (from code/): python improve/wesad_features.py <WESAD_E4 dir> <out.csv>"""
import sys, os, numpy as np, pandas as pd
SRC, OUTF = sys.argv[1], sys.argv[2]
code = open('pipeline/features.py').read()
code = code[:code.index('segs = pd.read_csv(SEG)')]
code = code.replace('ROOT, SEG, TIMES, GROUPS, OUT = sys.argv[1:6]', 'ROOT = SEG = TIMES = GROUPS = OUT = None').replace('os.makedirs(OUT, exist_ok=True)', '')
NS = {}; exec(compile(code, 'pipeline/features.py', 'exec'), NS)
WIN, STEP = NS['WIN'], NS['STEP']; process_session, feats = NS['process_session'], NS['feats']
segs = pd.read_csv(os.path.join(SRC, 'segments_wesad.csv'))
rows = []
for s, sg in segs.groupby('participant', sort=False):
    P = process_session('WESAD', 'WESAD', s, os.path.join(SRC, s)); rec_end = len(P['eda']) / P['fe']
    for r in sg.sort_values('start_s').itertuples():
        a, b0 = max(r.start_s, 0.0), min(r.end_s, rec_end)
        while a + WIN <= b0 + 1e-9:
            row = dict(protocol='WESAD', version='WESAD', participant=s, session=s, block=r.block, start_s=a, end_s=a + WIN, t_session=a)
            row.update(feats(P, a, a + WIN)); rows.append(row); a += STEP
    print(s, sum(1 for x in rows if x['participant'] == s), 'windows', flush=True)
W = pd.DataFrame(rows); W.to_csv(OUTF, index=False)
print(W.shape); print(W.groupby('block').size()); print(W.isna().mean()[W.isna().mean() > 0].round(3).to_string())
