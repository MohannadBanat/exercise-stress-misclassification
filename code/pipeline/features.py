"""Window- and block-level feature extraction exactly as specified in the analysis plan (Sections 3.2-3.4,
Appendix A, Table A1). No model is trained and no metric is computed here.
Usage: python -I features.py <dataset_root> <segments.csv> <session_times.csv> <concurrent_groups.csv> <outdir>"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

ROOT, SEG, TIMES, GROUPS, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
WIN, STEP = 30.0, 15.0
MIN_SHORT = 25.0  # plan v4, Section 3.4

# ---------- session-level decisions (Table 2)
S02_TRUNC_S = 1548.0          # documented duplication onset: ACC row 49,545 / BVP 99,091 / EDA,TEMP 6,195 -> 1548 s
PPG_TEMP_INVALID = {('STRESS', 'f07')}
ORDER = {('STRESS','V1'):['baseline','stroop','rest1','tmct','rest2','real_opinion','opposite_opinion','subtract'],
 ('STRESS','V2'):['baseline','tmct','rest1','real_opinion','opposite_opinion','rest2','subtract'],
 ('AEROBIC','V1'):['warmup']+[f'stage_{r}rpm' for r in [60,70,75,80,85,90,95,100,105,110]]+['cooldown','rest'],
 ('AEROBIC','V2'):['baseline','warmup']+[f'stage_{r}rpm' for r in [70,75,80,85]]+['stage_90_95rpm','cooldown','rest'],
 ('ANAEROBIC','V1'):['warmup']+sum([[f'sprint{i}',f'cooldown{i}'] for i in range(1,5)],[])+['rest'],
 ('ANAEROBIC','V2'):['baseline','warmup']+sum([[f'sprint{i}',f'cooldown{i}'] for i in range(1,5)],[])+['rest']}
STRESS_BLOCKS = {'stroop','tmct','real_opinion','opposite_opinion','subtract'}
REST_BLOCKS = {'baseline','rest1','rest2'}
AER_STAGES = {'stage_70rpm','stage_75rpm','stage_80rpm','stage_85rpm'}

def intensity(prot, ver, blk):
    """Ordinal exercise intensity (Section 3.8): 0 warm-up; 1 aerobic 60-75; 2 aerobic 80-95 (V2: 80-85);
    3 aerobic 100-110 (V1) or 90/95 (V2); 4 sprint. Cool-down, rest, baseline -> NaN (analysed separately)."""
    if blk == 'warmup': return 0
    if blk.startswith('sprint'): return 4
    if blk.startswith('stage'):
        r = blk.replace('stage_', '').replace('rpm', '')
        if r == '90_95': return 3
        r = int(r)
        return 1 if r <= 75 else (2 if r <= 95 else 3)
    return np.nan

def read_sig(d, name):
    f = os.path.join(d, name + '.csv')
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None, None
    h = pd.read_csv(f, header=None, nrows=2)
    x = pd.read_csv(f, header=None, skiprows=2).values.astype(float)
    return x, float(h.iloc[1, 0])

def sos(kind, fc, fs, order=4): return butter(order, fc, btype=kind, fs=fs, output='sos')

def slope(y, fs):
    if len(y) < 2 or np.all(np.isnan(y)): return np.nan
    t = np.arange(len(y)) / fs; ok = ~np.isnan(y)
    return np.polyfit(t[ok], y[ok], 1)[0] if ok.sum() >= 2 else np.nan

def process_session(prot, ver, sess, d):
    sig = {}
    for n in ['EDA', 'BVP', 'TEMP', 'ACC', 'HR']:
        x, fs = read_sig(d, n)
        if x is not None and prot == 'STRESS' and sess == 'S02':
            x = x[: int(S02_TRUNC_S * fs)]
        sig[n] = (x, fs)
    # EDA
    eda, fe = sig['EDA']; eda = eda[:, 0]
    eda_f = sosfiltfilt(sos('low', 1.0, fe), eda)
    tonic = sosfiltfilt(sos('low', 0.05, fe, 2), eda_f); phasic = eda_f - tonic
    pk, pr = find_peaks(phasic, prominence=0.01, distance=int(fe))
    scr_t, scr_a = pk / fe, pr['prominences']
    # BVP -> pulse intervals
    bvp, fb = sig['BVP']; bvp = bvp[:, 0]
    bf = sosfiltfilt(sos('band', [0.5, 5.0], fb, 2), bvp)
    seg = int(10 * fb); scale = np.array([np.median(np.abs(bf[i:i + seg])) for i in range(0, len(bf), seg)])
    prom_thr = np.repeat(scale, seg)[:len(bf)] * 0.5
    peaks, pp = find_peaks(bf, distance=int(0.33 * fb), prominence=(None, None))
    peaks = peaks[pp['prominences'] >= prom_thr[peaks]]
    pt = peaks / fb; pamp = bf[peaks]
    ibi = np.diff(pt); ibi_end = pt[1:]; amp_end = pamp[1:]
    ok = (ibi >= 0.33) & (ibi <= 1.5)
    med = pd.Series(np.where(ok, ibi, np.nan)).rolling(11, center=True, min_periods=3).median().values
    ok &= np.abs(ibi - med) <= 0.2 * med
    # TEMP
    temp, ft = sig['TEMP']; temp = temp[:, 0].copy(); temp[(temp < 20) | (temp > 40)] = np.nan
    if (prot, sess) in PPG_TEMP_INVALID: temp[:] = np.nan
    # ACC (1/64 g)
    acc, fa = sig['ACC']; acc = acc / 64.0; mag = np.linalg.norm(acc, axis=1)
    hr, fh = sig['HR']; hr = hr[:, 0] if hr is not None else None
    return dict(eda=eda, eda_f=eda_f, tonic=tonic, phasic=phasic, fe=fe, scr_t=scr_t, scr_a=scr_a,
                ibi=ibi, ibi_end=ibi_end, ibi_ok=ok, amp_end=amp_end, temp=temp, ft=ft, acc=acc, mag=mag, fa=fa,
                hr=hr, fh=fh, ppg_invalid=(prot, sess) in PPG_TEMP_INVALID)

def feats(P, a, b):
    """Features for the interval [a, b) seconds (Table A1)."""
    f = {}
    fe = P['fe']; i0, i1 = int(round(a * fe)), int(round(b * fe))
    raw = P['eda'][i0:i1]; ton = P['tonic'][i0:i1]; ph = P['phasic'][i0:i1]
    f['eda_valid'] = int(len(raw) > 0 and np.mean(raw < 0.05) <= 0.5)
    if f['eda_valid']:
        f.update(eda_tonic_mean=ton.mean(), eda_tonic_sd=ton.std(ddof=1), eda_tonic_slope=slope(ton, fe),
                 eda_tonic_min=ton.min(), eda_tonic_max=ton.max(), eda_tonic_range=ton.max() - ton.min())
        m = (P['scr_t'] >= a) & (P['scr_t'] < b)
        f['eda_scr_count'] = int(m.sum()); f['eda_scr_amp_mean'] = P['scr_a'][m].mean() if m.any() else 0.0
        f['eda_phasic_area'] = np.clip(ph, 0, None).sum() / fe
    else:
        for k in ['eda_tonic_mean','eda_tonic_sd','eda_tonic_slope','eda_tonic_min','eda_tonic_max','eda_tonic_range','eda_scr_count','eda_scr_amp_mean','eda_phasic_area']: f[k] = np.nan
    # BVP pulse
    m = (P['ibi_end'] >= a) & (P['ibi_end'] < b) & P['ibi_ok']
    iv = P['ibi'][m]; cov = iv.sum() / (b - a)
    f['bvp_coverage'] = np.nan if P['ppg_invalid'] else min(cov, 1.0)
    pr_ok = (not P['ppg_invalid']) and cov >= 0.5; hrv_ok = (not P['ppg_invalid']) and cov >= 0.7
    f['bvp_pr_mean'] = 60.0 / iv.mean() if pr_ok else np.nan
    f['bvp_pr_sd'] = np.std(60.0 / iv, ddof=1) if pr_ok and len(iv) > 1 else np.nan
    f['bvp_amp_sd'] = np.std(P['amp_end'][m], ddof=1) if pr_ok and m.sum() > 1 else np.nan
    if hrv_ok and len(iv) > 2:
        idx = np.where(m)[0]; consec = np.diff(idx) == 1
        d = np.diff(P['ibi'][idx])[consec]
        f['bvp_sdnn'] = np.std(iv, ddof=1) * 1000; f['bvp_rmssd'] = np.sqrt(np.mean(d ** 2)) * 1000 if len(d) else np.nan
    else: f['bvp_sdnn'] = f['bvp_rmssd'] = np.nan
    # device HR (HR.csv starts 10 s after the other signals)
    if P['hr'] is not None and not P['ppg_invalid']:
        h = P['hr'][max(0, int((a - 10) * P['fh'])):max(0, int((b - 10) * P['fh']))]
        f['hr_mean'] = h.mean() if len(h) else np.nan
    else: f['hr_mean'] = np.nan
    # TEMP
    ft = P['ft']; t = P['temp'][int(round(a * ft)):int(round(b * ft))]
    f['temp_valid'] = int(len(t) > 0 and np.mean(np.isnan(t)) <= 0.5)
    if f['temp_valid']:
        f['temp_mean'] = np.nanmean(t); f['temp_sd'] = np.nanstd(t, ddof=1); f['temp_slope'] = slope(t, ft)
    else: f['temp_mean'] = f['temp_sd'] = f['temp_slope'] = np.nan
    # ACC
    fa = P['fa']; j0, j1 = int(round(a * fa)), int(round(b * fa))
    A = P['acc'][j0:j1]; M = P['mag'][j0:j1]
    f['acc_mag_mean'] = M.mean(); f['acc_mag_sd'] = M.std(ddof=1)
    for k, ax in enumerate('xyz'):
        f[f'acc_{ax}_mean'] = A[:, k].mean(); f[f'acc_{ax}_sd'] = A[:, k].std(ddof=1)
    fr, pw = welch(M - M.mean(), fs=fa, nperseg=min(256, len(M)))
    band = (fr >= 0.3) & (fr <= 3.0); tot = (fr >= 0.3) & (fr <= 15.0)
    f['acc_dom_freq'] = fr[band][np.argmax(pw[band])] if band.any() and pw[band].sum() > 0 else np.nan
    f['acc_pow_frac_0p5_2'] = pw[(fr >= 0.5) & (fr <= 2.0)].sum() / pw[tot].sum() if pw[tot].sum() > 0 else np.nan
    f['still'] = int(f['acc_mag_sd'] < 0.02)
    return f

segs = pd.read_csv(SEG); segs['flag'] = segs['flag'].fillna('')
segs = segs[~segs.flag.str.contains('UNUSABLE|non-positive')]
times = pd.read_csv(TIMES, parse_dates=['start'])
first_start = times.groupby(['protocol', 'participant']).start.min()
groups = pd.read_csv(GROUPS)
gmap = {(r.protocol, r.session): r.group for r in groups.itertuples()}
v2groups = {g: chr(ord('A') + i) for i, g in enumerate(sorted({g for (p, s), g in gmap.items() if s.startswith('f')}, key=lambda x: int(x[1:])))}

W, B = [], []
for (prot, sess), sg in segs.groupby(['protocol', 'session']):
    ver = 'V1' if sess.startswith('S') else 'V2'; pid = sess.split('_')[0]
    d = os.path.join(ROOT, 'Wearable_Dataset', prot, sess)
    P = process_session(prot, ver, sess, d)
    t0 = times[(times.protocol == prot) & (times.session == sess)].start.iloc[0]
    off = (t0 - first_start[(prot, pid)]).total_seconds()  # offset of a continuation file (S11_b, S16_b)
    rec_end = len(P['eda']) / P['fe']
    grp = v2groups.get(gmap.get((prot, sess)), '') if ver == 'V2' else ''
    for r in sg.itertuples():
        a0, b0 = max(r.start_s, 0.0), min(r.end_s, rec_end)
        base = dict(protocol=prot, version=ver, participant=pid, session=sess, block=r.block,
                    block_pos=ORDER[(prot, ver)].index(r.block) if r.block in ORDER[(prot, ver)] else np.nan,
                    intensity=intensity(prot, ver, r.block) if prot != 'STRESS' else np.nan, v2_group=grp)
        # window-level (E2-E6)
        a = a0
        while a + WIN <= b0 + 1e-9:
            row = dict(base, start_s=a, end_s=a + WIN, t_session=off + a, centred=0)
            row.update(feats(P, a, a + WIN)); W.append(row); a += STEP
        # plan v4: a block lasting 25-30 s contributes one 30 s window centred on the block
        if MIN_SHORT <= b0 - a0 < WIN:
            c = (a0 + b0) / 2; a, b = max(0.0, c - WIN / 2), min(rec_end, c + WIN / 2)
            row = dict(base, start_s=a, end_s=b, t_session=off + a, centred=1)
            row.update(feats(P, a, b)); W.append(row)
        # block-level E1 samples (definitions of the dataset article)
        e1 = None
        if prot == 'STRESS' and r.block in STRESS_BLOCKS: e1 = ('stress', a0, b0)
        elif prot == 'STRESS' and r.block == 'baseline': e1 = ('rest', a0, b0)
        elif prot == 'STRESS' and r.block in ('rest1', 'rest2'): e1 = ('rest', (a0 + b0) / 2, b0)
        elif prot == 'AEROBIC' and r.block in AER_STAGES: c = (a0 + b0) / 2; e1 = ('aerobic', c - 30, c + 30)
        elif prot == 'ANAEROBIC' and r.block.startswith('sprint'): e1 = ('anaerobic', a0, b0)
        if e1:
            row = dict(base, e1_class=e1[0], start_s=e1[1], end_s=e1[2], t_session=off + e1[1])
            row.update(feats(P, e1[1], e1[2])); B.append(row)
    print(f'{prot:9s} {sess:6s} windows so far {len(W):6d}', flush=True)

W = pd.DataFrame(W); B = pd.DataFrame(B)
# task labels for windows (Table 3)
W['y_stress_rest'] = np.where((W.protocol == 'STRESS') & W.block.isin(STRESS_BLOCKS), 'stress',
                      np.where((W.protocol == 'STRESS') & W.block.isin(REST_BLOCKS), 'rest', ''))
W['y_aer_anaer'] = np.where((W.protocol == 'AEROBIC') & W.block.isin(AER_STAGES), 'aerobic',
                    np.where((W.protocol == 'ANAEROBIC') & W.block.str.startswith('sprint'), 'anaerobic', ''))
W['y_four'] = np.where(W.y_stress_rest != '', W.y_stress_rest, W.y_aer_anaer)
W.to_csv(os.path.join(OUT, 'windows.csv'), index=False); B.to_csv(os.path.join(OUT, 'e1_blocks.csv'), index=False)
summary = dict(n_windows=len(W), n_centred=int(W.centred.sum()), n_e1_samples=len(B), e1_counts=B.e1_class.value_counts().to_dict(),
               e1_counts_excl_f07_f13=B[~((B.protocol == 'STRESS') & B.participant.isin(['f07', 'f13']))].e1_class.value_counts().to_dict(),
               windows_by_task={k: W[k].value_counts().to_dict() for k in ['y_stress_rest', 'y_aer_anaer', 'y_four']},
               missing_frac={c: round(float(W[c].isna().mean()), 3) for c in W.columns if W[c].dtype.kind == 'f'})
json.dump(summary, open(os.path.join(OUT, 'feature_summary.json'), 'w'), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
