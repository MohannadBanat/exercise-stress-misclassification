"""M4: ONE-TIME evaluation on the 12 sealed participants.
1. Select the candidate with the best grouped 5-fold CV score (BDS) on all 24 development participants (same grid as M3).
2. Fit it on all 24; threshold = 0.80 sensitivity on the development out-of-fold stress windows.
3. Apply once to the sealed participants, together with the fixed reference pipeline. Writes a marker so it cannot rerun.
Usage: python -I run_m4.py --i-understand-this-runs-once"""
import sys, os, json, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
if '--i-understand-this-runs-once' not in sys.argv: raise SystemExit('refusing: pass --i-understand-this-runs-once')
MARK = 'improve/LOCKBOX_OPENED'
if os.path.exists(MARK): raise SystemExit('lockbox already opened on ' + open(MARK).read())
from framework2 import *; from build_features import build
lk = json.load(open('improve/lockbox.json'))
assert hashlib.sha256(json.dumps(lk['lockbox']).encode()).hexdigest() == lk['sha256'], 'lockbox list altered'
W, FS, ACC = build()
full = add_flags(detector_frame(W, final=True))
dev = full[~full.participant.isin(LOCK)]; box = full[full.participant.isin(LOCK)]
prev = {k: prev_index(full, k) for k in (45, 90)}
REF = dict(fs='v1_cal_ctx', model='et', arch='single', labels='all', k=45, rule='global')
C = [dict(fs=fs, model=m, arch=a, labels=l, k=k, rule=r) for fs, m in (('v2_z', 'xgb'), ('v1_cal_ctx', 'et'))
     for a in ('single', 'twostage') for l in ('all', 'clean') for k in (45, 90) for r in ('global', 'personal')]
cache, sc = {}, {}
for c in C:
    key = (c['fs'], c['model'], c['arch'], c['labels'])
    if key not in cache: cache[key] = oof(dev, FS[c['fs']], ACC, c['model'], c['arch'], c['labels'])
    sc[json.dumps(c, sort_keys=True)] = float(eval_candidate(dev, cache[key], prev, c['k'], c['rule'])[0].bds.mean())
best = json.loads(max(sc, key=sc.get)); log('selected on dev:', best, round(max(sc.values()), 3))
open(MARK, 'w').write(pd.Timestamp.now().isoformat() + ' selected=' + json.dumps(best, sort_keys=True))
out = {'selected_config': best, 'dev_cv_bds_selected': max(sc.values()), 'dev_cv_bds_reference': sc[json.dumps(REF, sort_keys=True)]}
res = {}
for name, c in (('selected', best), ('reference', REF)):
    key = (c['fs'], c['model'], c['arch'], c['labels'])
    tau = threshold(smooth(cache[key], prev, dev.index.values, c['k'])[dev.y.values == 1])
    p = train_predict(dev, box, FS[c['fs']], ACC, c['model'], c['arch'], c['labels'])
    r, ps = eval_candidate(box, p, prev, c['k'], c['rule'], tau=tau)
    au = {pid: float(roc_auc_score(g.y.values, ps[np.searchsorted(box.index.values, g.index.values)])) for pid, g in box.groupby('participant') if g.y.sum() and (g.y == 0).any()}
    r['auroc'] = r.participant.map(au); r['chosen'] = json.dumps(c, sort_keys=True); res[name] = r
    r.to_csv(f'improve/res_M4_{name}.csv', index=False); out[name] = summarize(r, f'lockbox {name}')
# operating curve for the selected pipeline: thresholds from development out-of-fold stress windows at several sensitivities
c = best; key = (c['fs'], c['model'], c['arch'], c['labels'])
pd_dev = smooth(cache[key], prev, dev.index.values, c['k'])
p_box = train_predict(dev, box, FS[c['fs']], ACC, c['model'], c['arch'], c['labels'])
curve = []
for st in (0.5, 0.6, 0.7, 0.8, 0.9):
    t = threshold(pd_dev[dev.y.values == 1], sens=st)
    r, ps = eval_candidate(box, p_box, prev, c['k'], c['rule'], tau=t)
    curve.append(dict(target_sens=st, sens=float(r.sens.mean()), far_rest=float(r.far_rest.mean()), far_ex=float(r.far_ex.mean()), far_base=float(r.far_base.mean()), spec=float(r.spec.mean())))
out['operating_curve_lockbox'] = curve
# threshold-free: stress vs rest within the stress session (per participant AUROC)
ps_sel = smooth(p_box, prev, box.index.values, c['k']); pos = pd.Series(np.arange(len(box)), index=box.index.values)
ss = [float(roc_auc_score(g.y.values, ps_sel[pos[g.index].values])) for _, g in box[box.protocol == 'STRESS'].groupby('participant') if g.y.sum() and (g.y == 0).any()]
out['within_session_auroc_lockbox'] = [float(np.mean(ss)), len(ss)]
out['gain_bds'] = paired_gain(res['selected'], res['reference'], 'bds'); out['gain_spec'] = paired_gain(res['selected'], res['reference'], 'spec')
out['gain_sens'] = paired_gain(res['selected'], res['reference'], 'sens')
json.dump(out, open('improve/sum_M4.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
