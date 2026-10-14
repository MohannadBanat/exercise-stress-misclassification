"""Diagnose what limits the exercise-robust detector (development participants only; lockbox excluded).
Out-of-fold predictions of the final pipeline (calibrated + context features, extremely randomized trees, 45 s smoothing)."""
import sys, json, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore'); sys.path.insert(0, 'improve')
from framework2 import *; from build_features import build
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
W, FS, ACC = build(); d = add_flags(detector_frame(W)); prev = {45: prev_index(d, 45)}
assert not set(d.participant) & set(LOCK)
p = oof(d, FS['v1_cal_ctx'], ACC, 'et', 'single', 'all'); d['p'] = smooth(p, prev, d.index.values, 45)
tau = threshold(d.p[d.y == 1].values); d['alarm'] = d.p >= tau
out = {'tau': tau}
# 1. where do rest false alarms happen? position within rest block, baseline vs rest, cohort
r = d[d.kind == 'rest'].copy()
g = r.groupby(['session', 'block']).start_s; r['pos'] = (r.start_s - g.transform('min')) / (g.transform('max') - g.transform('min') + 1e-9)
r['third'] = pd.cut(r.pos, [-0.01, 1/3, 2/3, 1.01], labels=['first third', 'middle third', 'last third'])
r['blk'] = np.where(r.block == 'baseline', 'baseline', 'post-task rest')
out['far_by_restpart'] = r[r.blk != 'baseline'].groupby('third').alarm.mean().round(3).to_dict()
out['far_baseline_vs_rest'] = r.groupby('blk').alarm.mean().round(3).to_dict()
out['far_rest_by_cohort'] = r.groupby('version').alarm.mean().round(3).to_dict()
# 2. sensitivity by stress task
s = d[d.kind == 'stress']; out['sens_by_task'] = s.groupby('block').alarm.mean().round(3).to_dict()
out['n_windows_by_task'] = s.groupby('block').size().to_dict()
# 3. separability ceilings within the stress session (per participant AUROC), by rest subset
ss = d[d.protocol == 'STRESS']
def auc(sub):
    v = [roc_auc_score(gg.y, gg.p) for _, gg in sub.groupby('participant') if gg.y.nunique() == 2]; return round(float(np.mean(v)), 3), len(v)
late = ss[(ss.kind == 'stress') | ((ss.kind == 'rest') & ~ss.rest_early.astype(bool) & (ss.is_base == 0))]
early = ss[(ss.kind == 'stress') | ((ss.kind == 'rest') & ss.rest_early.astype(bool))]
base = ss[(ss.kind == 'stress') | (ss.is_base == 1)]
out['auroc_stress_vs_baseline'] = auc(base); out['auroc_stress_vs_early_rest'] = auc(early); out['auroc_stress_vs_late_rest'] = auc(late)
out['auroc_stress_vs_all_rest'] = auc(ss)
# 4. do block scores follow self-reported stress? (within person, Spearman over blocks)
R = 'raw/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/'
m1 = {'Baseline': 'baseline', 'Stroop': 'stroop', 'First Rest': 'rest1', 'TMCT': 'tmct', 'Second Rest': 'rest2', 'Real Opinion': 'real_opinion', 'Opposite Opinion': 'opposite_opinion', 'Subtract': 'subtract'}
sl = []
for f in ('Stress_Level_v1.csv', 'Stress_Level_v2.csv'):
    t = pd.read_csv(R + f, index_col=0).rename(columns=lambda c: m1.get(c.strip(), c.strip()))
    for pid, row in t.iterrows():
        for blk, v in row.items(): sl.append(dict(participant=str(pid).strip(), block=blk, sl=float(v)))
sl = pd.DataFrame(sl)
bs = ss.groupby(['participant', 'block']).agg(score=('p', 'mean'), alarm=('alarm', 'mean'), y=('y', 'max')).reset_index().merge(sl, on=['participant', 'block'])
rho = [spearmanr(g.sl, g.score)[0] for _, g in bs.groupby('participant') if g.sl.nunique() > 2]
out['within_person_spearman_score_vs_selfreport'] = [round(float(np.nanmean(rho)), 3), len(rho)]
# rest blocks that people reported as stressful: are they flagged more?
rb = bs[bs.y == 0].copy(); rb['sl_z'] = rb.groupby('participant').sl.transform(lambda v: v - v.mean())
out['rest_far_by_selfreport'] = {'rest blocks rated above own mean': round(float(rb[rb.sl_z > 0].alarm.mean()), 3), 'at or below own mean': round(float(rb[rb.sl_z <= 0].alarm.mean()), 3)}
sb = bs[bs.y == 1].copy(); sb['sl_z'] = sb.groupby('participant').sl.transform(lambda v: v - v.mean())
out['stress_sens_by_selfreport'] = {'stress blocks rated above own mean': round(float(sb[sb.sl_z > 0].alarm.mean()), 3), 'at or below own mean': round(float(sb[sb.sl_z <= 0].alarm.mean()), 3)}
# 5. how much of a stress block's label is "on" physiologically: score over time within stress blocks
st = s.copy(); g = st.groupby(['session', 'block']).start_s; st['pos'] = st.start_s - g.transform('min')
out['sens_by_seconds_into_task'] = st.groupby(pd.cut(st.pos, [-1, 14, 44, 89, 1e4], labels=['0-14 s', '15-44 s', '45-89 s', '90+ s'])).alarm.mean().round(3).to_dict()
# 6. per-participant spread
pp = d.groupby('participant').apply(lambda g: pd.Series(dict(sens=g[g.kind == 'stress'].alarm.mean(), far_rest=g[(g.kind == 'rest')].alarm.mean())))
out['participant_far_rest_quartiles'] = pp.far_rest.quantile([0.25, 0.5, 0.75]).round(2).tolist()
json.dump(out, open('improve/diagnosis.json', 'w'), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
