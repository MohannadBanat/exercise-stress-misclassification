"""POST HOC response magnitudes (manuscript Section 4.10: 'the TSST raised tonic EDA by ... and pulse rate by ...').
For each group, the median over participants of each participant's median calibrated feature (difference from the
participant's own baseline), rounded to 2 decimals, appended to improve/sum_WESAD_diagnostics.json as 'magnitude_check'.
Groups: training stress, rest (excluding the baseline used for calibration) and exercise windows of the detector frame;
WESAD TSST and all other WESAD windows (calibrated against the WESAD baseline).
This computation was first run inline on 2026-10-09 (23:52 UTC) right after wesad_diagnostics.py; this script was
written on 2026-10-14 and reproduces all 20 archived values exactly. Run after improve/wesad_diagnostics.py."""
import sys, json, warnings, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from framework2 import add_flags, detector_frame
from build_features import build
from features_v1 import PHYS
F = ['cal_eda_tonic_mean', 'cal_hr_mean', 'cal_bvp_pr_mean', 'cal_eda_scr_count']
W, FS, ACCF = build(); d = add_flags(detector_frame(W, final=True))
Wt = pd.read_csv('features_wesad/windows_wesad.csv')
base = Wt[Wt.block == 'baseline'].groupby('participant')[PHYS].mean()
for c in PHYS: Wt['cal_' + c] = Wt[c] - Wt.participant.map(base[c])
G = {'training_stress': d[d.kind == 'stress'], 'training_rest': d[(d.kind == 'rest') & (d.block != 'baseline')],
     'training_exercise': d[d.kind == 'exercise'], 'wesad_tsst': Wt[Wt.block == 'tsst'], 'wesad_rest': Wt[Wt.block != 'tsst']}
MC = {k: {f: round(float(v), 2) for f, v in g.groupby('participant')[F].median().median().items()} for k, g in G.items()}
P = 'improve/sum_WESAD_diagnostics.json'; out = json.load(open(P)); out['magnitude_check'] = MC
json.dump(out, open(P, 'w'), indent=1); print(json.dumps(MC, indent=1))
