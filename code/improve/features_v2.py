"""v2 features for the improved detector (development code; no labels used).
Per session: NeuroKit2 EDA decomposition (cvxEDA, fallback smooth-median), SCR peaks; PPG cleaning and Elgendi
peaks with template-matching quality; ACC magnitude. For each existing window (end time t), causal features are
computed over the trailing 30, 60 and 120 s [t-L, t]. Usage: python -I features_v2.py <dataset_root> <windows.csv> <out.csv>"""
import os, sys, warnings, numpy as np, pandas as pd
import neurokit2 as nk
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
ROOT, WIN, OUT = sys.argv[1:4]
S02_TRUNC_S = 1548.0

def rd(d, n):
    f = os.path.join(d, n + '.csv')
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None, None
    h = pd.read_csv(f, header=None, nrows=2); return pd.read_csv(f, header=None, skiprows=2).values.astype(float), float(h.iloc[1, 0])

def session(prot, sess, wins):
    d = os.path.join(ROOT, 'Wearable_Dataset', prot, sess); S = {}
    for n in ('EDA', 'BVP', 'TEMP', 'ACC'):
        x, fs = rd(d, n)
        if prot == 'STRESS' and sess == 'S02': x = x[: int(S02_TRUNC_S * fs)]
        S[n] = (x, fs)
    ppg_invalid = (prot, sess) == ('STRESS', 'f07')
    # EDA
    eda, fe = S['EDA'][0][:, 0], S['EDA'][1]
    try:
        clean = nk.eda_clean(eda, sampling_rate=fe, method='neurokit')
        comp = nk.eda_phasic(clean, sampling_rate=fe, method='cvxeda'); method = 'cvxeda'
    except Exception:
        clean = eda; comp = nk.eda_phasic(eda, sampling_rate=fe, method='smoothmedian'); method = 'smoothmedian'
    ton, pha = comp['EDA_Tonic'].values, comp['EDA_Phasic'].values
    try:
        _, info = nk.eda_peaks(pha, sampling_rate=fe, method='neurokit', amplitude_min=0.01)
        scr_t = np.asarray(info['SCR_Peaks'], float) / fe; scr_a = np.nan_to_num(np.asarray(info['SCR_Amplitude'], float))
    except Exception:
        scr_t, scr_a = np.array([]), np.array([])
    # PPG
    bvp, fb = S['BVP'][0][:, 0], S['BVP'][1]
    pc = nk.ppg_clean(bvp, sampling_rate=fb, method='elgendi')
    try:
        _, pinfo = nk.ppg_peaks(pc, sampling_rate=fb, method='elgendi'); pk = np.asarray(pinfo['PPG_Peaks'])
        q = nk.ppg_quality(pc, peaks=pk, sampling_rate=fb, method='templatematch')
    except Exception:
        pk = np.array([], int); q = np.full(len(pc), np.nan)
    pt = pk / fb; ibi = np.diff(pt); ibi_t = pt[1:]
    ok = (ibi > 0.33) & (ibi < 1.5)
    med = pd.Series(np.where(ok, ibi, np.nan)).rolling(11, center=True, min_periods=3).median().values
    ok &= np.abs(ibi - med) <= 0.2 * med
    temp, ft = S['TEMP'][0][:, 0].copy(), S['TEMP'][1]; temp[(temp < 20) | (temp > 40)] = np.nan
    if ppg_invalid: temp[:] = np.nan
    acc, fa = S['ACC'][0] / 64.0, S['ACC'][1]; mag = np.linalg.norm(acc, axis=1)
    rows = []
    for w in wins.itertuples():
        t1 = w.end_s; r = dict(row_id=w.Index)
        for L in (30, 60, 120):
            t0 = max(0.0, t1 - L); a, b = int(t0 * fe), int(t1 * fe); s = f'_{L}'
            tt, pp, rr = ton[a:b], pha[a:b], eda[a:b]
            valid = len(rr) > 0 and np.mean(rr < 0.05) <= 0.5
            r['e2_ton_mean' + s] = tt.mean() if valid else np.nan
            r['e2_ton_slope' + s] = np.polyfit(np.arange(len(tt)) / fe, tt, 1)[0] if valid and len(tt) > 2 else np.nan
            m = (scr_t >= t0) & (scr_t < t1)
            r['e2_scr_rate' + s] = m.sum() / (t1 - t0) * 60 if valid else np.nan
            r['e2_scr_amp' + s] = scr_a[m].mean() if valid and m.any() else (0.0 if valid else np.nan)
            r['e2_pha_auc' + s] = np.clip(pp, 0, None).sum() / fe if valid else np.nan
            # PPG
            m = (ibi_t >= t0) & (ibi_t < t1) & ok; iv = ibi[m]; cov = iv.sum() / (t1 - t0)
            qa, qb = int(t0 * fb), int(t1 * fb); qq = np.nanmean(q[qa:qb]) if qb > qa else np.nan
            good = (not ppg_invalid) and cov >= 0.5
            r['p2_quality' + s] = np.nan if ppg_invalid else qq
            r['p2_cov' + s] = np.nan if ppg_invalid else min(cov, 1)
            r['p2_hr' + s] = 60 / iv.mean() if good else np.nan
            if good and len(iv) > 3 and cov >= 0.7:
                dd = np.diff(iv); r['p2_rmssd' + s] = np.sqrt(np.mean(dd ** 2)) * 1000; r['p2_sdnn' + s] = iv.std(ddof=1) * 1000
                r['p2_pnn50' + s] = np.mean(np.abs(dd) > 0.05)
            else: r['p2_rmssd' + s] = r['p2_sdnn' + s] = r['p2_pnn50' + s] = np.nan
            # TEMP
            ta, tb = int(t0 * ft), int(t1 * ft); tp = temp[ta:tb]
            okt = len(tp) > 2 and np.mean(np.isnan(tp)) <= 0.5
            r['t2_mean' + s] = np.nanmean(tp) if okt else np.nan
            r['t2_slope' + s] = np.polyfit(np.arange(len(tp))[~np.isnan(tp)] / ft, tp[~np.isnan(tp)], 1)[0] if okt else np.nan
            # ACC
            aa, ab = int(t0 * fa), int(t1 * fa); mm = mag[aa:ab]
            r['a2_sd' + s] = mm.std() if len(mm) > 1 else np.nan
            r['a2_energy' + s] = np.mean((mm - mm.mean()) ** 2) if len(mm) > 1 else np.nan
        # HF power of the pulse-interval series over 120 s (0.15-0.4 Hz), via Lomb-Scargle on accepted intervals
        t0 = max(0.0, t1 - 120); m = (ibi_t >= t0) & (ibi_t < t1) & ok
        if (not ppg_invalid) and m.sum() >= 40:
            from scipy.signal import lombscargle
            f = np.linspace(0.04, 0.4, 60); x = ibi[m] - ibi[m].mean()
            pw = lombscargle(ibi_t[m], x, 2 * np.pi * f, normalize=False)
            lf, hf = pw[(f >= 0.04) & (f < 0.15)].sum(), pw[(f >= 0.15)].sum()
            r['p2_lnhf_120'] = np.log(hf + 1e-12); r['p2_lfhf_120'] = lf / (hf + 1e-12)
        else: r['p2_lnhf_120'] = r['p2_lfhf_120'] = np.nan
        r['eda_method'] = method; rows.append(r)
    return rows

W = pd.read_csv(WIN, index_col=0) if "scratchpad" in WIN else pd.read_csv(WIN)
jobs = [(p, s, g) for (p, s), g in W.groupby(['protocol', 'session'])]
res = Parallel(n_jobs=2, verbose=0)(delayed(session)(p, s, g) for p, s, g in jobs)
F = pd.DataFrame([r for rr in res for r in rr]).set_index('row_id').sort_index()
F.to_csv(OUT); print(F.shape, F.eda_method.value_counts().to_dict()); print(F.isna().mean().round(3).to_string())
