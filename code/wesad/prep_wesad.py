"""WESAD preparation (no model, no outcome): extract each subject's raw Empatica E4 CSVs, align the WESAD label
stream (700 Hz, synchronised with the wrist signals in S#.pkl) to the raw E4 clock by exact matching of the wrist
EDA samples, and write the condition blocks in seconds from the raw E4 recording start."""
import zipfile, io, pickle, sys, os, numpy as np, pandas as pd, gc
SRC = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
Z = zipfile.ZipFile(SRC)
subs = sorted({n.split('/')[1] for n in Z.namelist() if n.count('/') >= 2 and n.split('/')[1].startswith('S')}, key=lambda s: int(s[1:]))
only = sys.argv[3].split(',') if len(sys.argv) > 3 else subs
NAMES = {1: 'baseline', 2: 'tsst', 3: 'amusement', 4: 'medi'}
segs, sync = [], []
for s in only:
    d = os.path.join(OUT, s); os.makedirs(d, exist_ok=True)
    inner = zipfile.ZipFile(io.BytesIO(Z.read(f'WESAD/{s}/{s}_E4_Data.zip')))
    for n in inner.namelist():
        if n.endswith('.csv'): open(os.path.join(d, n), 'wb').write(inner.read(n))
    for n in (f'{s}_quest.csv', f'{s}_readme.txt'): open(os.path.join(d, n), 'wb').write(Z.read(f'WESAD/{s}/{n}'))
    raw = pd.read_csv(os.path.join(d, 'EDA.csv'), header=None).values[:, 0]; t0, fs = raw[0], raw[1]; raw = raw[2:]
    with Z.open(f'WESAD/{s}/{s}.pkl') as f: P = pickle.load(f, encoding='latin1')
    eda = np.asarray(P['signal']['wrist']['EDA']).ravel(); lab = np.asarray(P['label']).ravel(); del P; gc.collect()
    L = 40; pat = eda[:L]; cand = [k for k in range(len(raw) - L) if np.allclose(raw[k:k + L], pat, atol=1e-6)]
    k = cand[0] if cand else -1
    nmatch = int(np.isclose(raw[k:k + len(eda)], eda[:len(raw) - k], atol=1e-6).sum()) if k >= 0 else 0
    sync.append(dict(subject=s, e4_start_unix=t0, eda_fs=fs, offset_samples=k, offset_s=k / fs if k >= 0 else np.nan, n_candidates=len(cand),
                     pkl_eda_n=len(eda), matched=nmatch, raw_eda_n=len(raw), label_n=len(lab), label_dur_s=len(lab) / 700, eda_dur_s=len(eda) / 4))
    # label runs -> blocks (>= 60 s), times on the raw E4 clock
    ch = np.flatnonzero(np.diff(lab)) + 1; st = np.r_[0, ch]; en = np.r_[ch, len(lab)]; medi = 0
    for a, b in zip(st, en):
        v = int(lab[a])
        if v in NAMES and (b - a) / 700 >= 60:
            nm = NAMES[v]
            if v == 4: medi += 1; nm = f'medi{medi}'
            segs.append(dict(protocol='WESAD', participant=s, session=s, block=nm, label=v, start_s=k / fs + a / 700, end_s=k / fs + b / 700, duration_s=(b - a) / 700))
    print(s, sync[-1], flush=True)
pd.DataFrame(segs).to_csv(os.path.join(OUT, f'segments_wesad{"_" + only[0] if len(only) == 1 else ""}.csv'), index=False)
pd.DataFrame(sync).to_csv(os.path.join(OUT, f'sync_wesad{"_" + only[0] if len(only) == 1 else ""}.csv'), index=False)
