"""M3: decision-level candidates, nested LOPO on the 24 development participants, with a fixed reference
(the M1/M2 winner: v1 calibrated + context features, extremely randomized trees, 45 s smoothing, global threshold, single model)."""
import sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from framework2 import *; from build_features import build
W, FS, ACC = build()
d = add_flags(detector_frame(W)); log('dev', d.participant.nunique(), len(d), 'lockbox present:', bool(set(d.participant) & set(LOCK)))
prev = {k: prev_index(d, k) for k in (45, 90)}
REF = dict(fs='v1_cal_ctx', model='et', arch='single', labels='all', k=45, rule='global')
C = [dict(fs=fs, model=m, arch=a, labels=l, k=k, rule=r) for fs, m in (('v2_z', 'xgb'), ('v1_cal_ctx', 'et'))
     for a in ('single', 'twostage') for l in ('all', 'clean') for k in (45, 90) for r in ('global', 'personal')]
sel, ref = nested(d, C, FS, ACC, prev, 'M3_decision', REF)
sel.to_csv('improve/res_M3_selected.csv', index=False); ref.to_csv('improve/res_M3_reference.csv', index=False)
out = dict(selected=summarize(sel, 'M3 selected (nested)'), reference=summarize(ref, 'M1/M2 reference pipeline'), gain_bds=paired_gain(sel, ref, 'bds'),
           gain_spec=paired_gain(sel, ref, 'spec'), gain_sens=paired_gain(sel, ref, 'sens'))
json.dump(out, open('improve/sum_M3.json', 'w'), indent=1); print(json.dumps(out, indent=1))
