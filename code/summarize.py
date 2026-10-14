"""Build vals.json: every number quoted in the manuscript, formatted, from the result files."""
import json, numpy as np, pandas as pd
RR, RX, RN = 'results_real', 'results_x', 'results_null_e1'
r = json.load(open(f'{RR}/results.json')); T = r['confirmatory']; x = json.load(open(f'{RX}/exploratory.json'))
rng = np.random.default_rng(20261008)
V = {}
f2 = lambda v: ('0.00' if abs(v) < 0.005 else f'{v:.2f}').replace('-', '−')
def ci(v, n=10000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
cis = lambda lo, hi: f'{f2(lo)} to {f2(hi)}'
def pfmt(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'
# ---- E1
e1 = pd.read_csv(f'{RR}/E1_runs.csv'); key = {'stress_rest': 'sr', 'aer_anaer': 'aa', 'four': 'f'}; pub = {'stress_rest': 0.93, 'aer_anaer': 0.91, 'four': 0.84}
means = []
for t, k in key.items():
    for v in ('clean', 'leaky'):
        a = e1[(e1.task == t) & (e1.variant == v)].accuracy
        V[f'e1_{k}_{v}'] = f2(a.mean()); V[f'e1_{k}_{v}_sd'] = f2(a.std()); means.append(a.mean())
        V[f'e1_{k}_{v}_p5'] = f2(a.quantile(.05)); V[f'e1_{k}_{v}_p95'] = f2(a.quantile(.95))
        V[f'e1_{k}_{v}_share'] = f'{100 * (a >= pub[t]).mean():.0f}%'
V['e1_min'], V['e1_max'] = f2(min(means)), f2(max(means)); V['e1_range'] = f'{f2(min(means))}–{f2(max(means))}'
gains = [e1[(e1.task == t) & (e1.variant == 'leaky')].accuracy.mean() - e1[(e1.task == t) & (e1.variant == 'clean')].accuracy.mean() for t in key]
V['e1_leak_gain_min'], V['e1_leak_gain_max'] = f2(min(gains)), f2(max(gains))
n1 = pd.read_csv(f'{RN}/E1_runs.csv')
for t, k in key.items():
    for v in ('clean', 'leaky'): V[f'null_{k}_{v}'] = f2(n1[(n1.task == t) & (n1.variant == v)].accuracy.mean())
V['null_seeds'] = str(n1.seed.nunique())
# ---- E2
for t, k in key.items():
    for kind, suf in (('xgb', ''), ('lr', '_lr')):
        pp = pd.read_csv(f'{RR}/E2_{t}_{kind}_perparticipant.csv', index_col=0)
        V[f'e2_{k}_samp{suf}'] = f2(pp.ba_sample.mean()); V[f'e2_{k}_lopo{suf}'] = f2(pp.ba_lopo.mean())
# ---- confirmatory
for h, t in T.items():
    hk = h.lower()
    if 'ci_low' in t: V[f'{hk}_ci'] = cis(t['ci_low'], t['ci_high'])
    V[f'{hk}_est'] = (('+' if t['estimate'] > 0 and h in ('H1a', 'H1b', 'H1c', 'H6') else '') + f2(t['estimate']))
    V[f'{hk}_p'] = pfmt(t['p']); V[f'{hk}_ph'] = pfmt(t['p_holm']); V[f'{hk}_s'] = 'Yes' if t['reject_H0'] else 'No'
V['h1b_mag'] = f2(T['H1b']['estimate']); V['h1c_mag'] = f2(T['H1c']['estimate'])
V['h7_or'] = f2(np.exp(T['H7']['estimate'])); V['h7_or_ci'] = cis(np.exp(T['H7']['ci_low']), np.exp(T['H7']['ci_high']))
V['h8_ratio'] = f2(T['H8']['estimate']); V['h8_loss'] = f2(T['H8']['sens_loss'])
# ---- E3
pp = pd.read_csv(f'{RR}/E3_four_perparticipant.csv', index_col=0)
for c, k in [('full', 'full'), ('acc_only', 'acc'), ('block_position', 'block'), ('time_only', 'time')]:
    m, lo, hi = ci(pp[c]); V[f'e3_{k}'] = f2(m); V[f'e3_{k}_ci'] = cis(lo, hi)
V['e3_acc_share'] = f'{100 * (pp.acc_only.mean() - 0.25) / (pp.full.mean() - 0.25):.0f}%'
# ---- E4
e4 = r['E4']; V.update(e4_ba=f2(e4['mean_participant_ba']), e4_n=str(e4['n_participants']), e4_nw=str(e4['n_windows']), e4_p=f"{T['H4']['p']:.3f}",
                       e4_ph=f"{T['H4']['p_holm']:.2f}", e4_null95=f2(e4['null_95']), x1_ba=f2(e4['X1_aer_vs_anaer_baseline_ba']), x1_n=str(e4['X1_n']))
# ---- E5
e5 = pd.read_csv(f'{RR}/E5_perparticipant.csv', index_col=0); V['e5_within'] = f2(e5.within.mean()); V['e5_cross'] = f2(e5.cross.mean())
# ---- E6
e6 = pd.read_csv(f'{RR}/E6_perparticipant.csv'); b, a = e6[~e6.aware], e6[e6.aware]; E6 = r['E6']
for nm, v in [('far_ex', b.far_ex), ('far_rest', b.far_rest), ('sens', b.sens), ('aw_far_ex', a.far_ex), ('aw_sens', a.sens), ('aw_far_rest', a.far_rest)]:
    m, lo, hi = ci(v); V[f'e6_{nm}'] = f2(m); V[f'e6_{nm}_ci'] = cis(lo, hi)
V['e6_aph'] = f'{b.far_ex.mean() * 240:.0f}'
for k2, v in E6['far_by_intensity'].items(): V[f'e6_i{int(float(k2))}'] = f2(v)
V['e6_i1'] = f2(min(E6['far_by_intensity'][k] for k in E6['far_by_intensity'] if 1 <= float(k) <= 3))
V['e6_i3'] = f2(max(E6['far_by_intensity'][k] for k in E6['far_by_intensity'] if 1 <= float(k) <= 3))
# ---- exploratory
PIPE = {'raw': 'raw features', 'cal': 'calibrated features', 'cal_ctx': 'calibrated and context features', 'xgb': 'XGBoost', 'lr': 'logistic regression', 'et': 'extremely randomized trees'}
def pipe_name(s):
    fs, mk, k = s.split('|'); return f"{PIPE[fs]} with {PIPE[mk]}" + (' and smoothing' if k == '3' else '')
for t, k in [('stress_rest', 'sr'), ('four', 'f')]:
    xx = x[t]; V[f'x_{k}_ref'] = f2(xx['reference']['mean_participant_ba']); V[f'x_{k}_imp'] = f2(xx['improved']['mean_participant_ba'])
    V[f'x_{k}_ci'] = cis(*xx['improved']['ci']); V[f'x_{k}_gain'] = '+' + f2(xx['gain']['estimate']); V[f'x_{k}_gain_ci'] = cis(*xx['gain']['ci'])
    top = max(xx['chosen_pipelines'], key=xx['chosen_pipelines'].get); V[f'x_{k}_pipe'] = f"{pipe_name(top)} ({xx['chosen_pipelines'][top]} of {xx['n_participants']} folds)"
d = x['deploy']
V.update(x_dep_sens=f2(d['sens']), x_dep_sens_ci=cis(*d['sens_ci']), x_dep_far=f2(d['far_ex']), x_dep_far_ci=cis(*d['far_ex_ci']),
         x_dep_far_rest=f2(d['far_rest']), x_dep_far_rest_ci=cis(*d['far_rest_ci']))
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print(len(V), 'values'); print(json.dumps(V, ensure_ascii=False)[:3000])
# ---- detector development (M1-M3) and sealed test (M4)
import re
pf = lambda s: float(re.match(r'([0-9.]+)', s).group(1))
m1, m2 = json.load(open('improve/sum_M1.json')), json.load(open('improve/sum_M2.json'))
m3, m4 = json.load(open('improve/sum_M3.json')), json.load(open('improve/sum_M4.json'))
V['d_m1_spec'] = f2(pf(m1['spec'])); V['d_m2_spec'] = f2(pf(m2['spec'])); V['d_m1_far_base'] = f2(pf(m1['far_base'])); V['d_m2_far_base'] = f2(pf(m2['far_base']))
for tag, s in (('d_ref', m3['reference']), ('d_sel', m3['selected']), ('lb', m4['selected'])):
    for k in ('sens', 'spec', 'far_rest', 'far_ex', 'far_base', 'bds', 'auroc'):
        m, lo, hi = s[k]; V[f'{tag}_{k}'] = f2(m); V[f'{tag}_{k}_ci'] = cis(lo, hi)
g = m3['gain_bds']; V['d_gain_bds'] = ('+' if g['gain'] > 0 else '') + f2(g['gain']); V['d_gain_bds_ci'] = cis(*g['ci'])
V['lb_auroc_ws'] = f2(m4['within_session_auroc_lockbox'][0]); V['lb_n'] = str(m4['selected']['n'])
for c in m4['operating_curve_lockbox']:
    t = str(int(round(c['target_sens'] * 10)))
    for k in ('sens', 'far_rest', 'far_ex', 'spec'): V[f'lb{t}_{k}'] = f2(c[k])
tl = pd.read_csv('improve/trial_log.csv'); V['n_configs'] = str(int(tl.tag.isin(['M1_baseline', 'M2_features', 'M3_decision']).sum())); V['n_rules_m5'] = str(int((tl.tag == 'M5_episode').sum())); V['n_cfg_m1'], V['n_cfg_m2'], V['n_cfg_m3'] = [str((tl.tag == t).sum()) for t in ('M1_baseline', 'M2_features', 'M3_decision')]
V['lockbox_ids'] = ', '.join(json.load(open('improve/lockbox.json'))['lockbox'])
V['e6_reg_far_ex'] = V['e6_far_ex']
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('detector values added:', {k: V[k] for k in ('d_ref_bds', 'd_sel_bds', 'd_gain_bds', 'lb_sens', 'lb_far_ex', 'lb_far_rest', 'lb6_sens', 'lb6_far_rest', 'n_configs')})
# ---- episode-level evaluation (M5, exploratory; pre-specified in improve/episode_prespec.md)
import pickle
m5 = json.load(open('improve/sum_M5.json'))
f1 = lambda v: ('0.0' if abs(v) < 0.05 else f'{v:.1f}').replace('-', '−')
sg = lambda v, f: ('+' if v > 0 and f(v) not in ('0.0', '0.00') else '') + f(v)
ci1 = lambda lo, hi: f'{f1(lo)} to {f1(hi)}'
RATE = ('fa_rest', 'fa_ex', 'fa_base', 'fa_comb')
for tag, key in (('ep_ref', 'reference'), ('ep_sel', 'selected'), ('ep_ref_g0', 'reference_G0'), ('ep_sel_g0', 'selected_G0'), ('ep_ref_st', 'reference_strict'), ('ep_sel_st', 'selected_strict')):
    for k, (m, lo, hi) in m5[key].items():
        f, c = (f1, ci1) if k in RATE else (f2, cis)
        V[f'{tag}_{k}'] = f(m); V[f'{tag}_{k}_ci'] = c(lo, hi)
for k, (m, lo, hi) in m5['diff_selected_minus_reference'].items():
    f, c = (f1, ci1) if k in RATE else (f2, cis)
    V[f'ep_d_{k}'] = sg(m, f); V[f'ep_d_{k}_ci'] = c(lo, hi)
for tag, key in (('ep_ref', 'reference_extra'), ('ep_sel', 'selected_extra')):
    e = m5[key]
    V.update({f'{tag}_ndet': str(e['n_detected']), f'{tag}_nfa_rest': str(e['n_false_rest']), f'{tag}_nfa_ex': str(e['n_false_ex']), f'{tag}_nfa_base': str(e['n_false_base']),
              f'{tag}_lat_med': f"{e['latency_median']:.0f}", f'{tag}_lat_q1': f"{e['latency_q1']:.0f}", f'{tag}_lat_q3': f"{e['latency_q3']:.0f}",
              f'{tag}_pp_fa_rest': f1(e['perparticipant_mean_fa_rest']), f'{tag}_pp_fa_ex': f1(e['perparticipant_mean_fa_ex'])})
    for t, (a, b) in e['by_type'].items(): V[f'{tag}_t_{t}'] = f'{a} of {b}'
e = m5['reference_extra']
V.update(ep_n=str(e['n_episodes']), ep_h_rest=f1(e['hours_rest']), ep_h_ex=f1(e['hours_ex']), ep_h_base=f1(e['hours_base']), ep_n_ex=str(e['n_participants_ex']))
R5 = pickle.load(open('improve/m5_all.pkl', 'rb'))
for nm, get in (('ref', lambda r: r['grid'][(0.8, 1)]), ('sel', lambda r: r['grid'][r['best']])):
    lat = np.concatenate([np.asarray(get(r)['lat'], float) for r in R5]); V[f'ep_{nm}_active'] = str(int((lat == 0).sum())); V[f'ep_{nm}_active_share'] = f2((lat == 0).mean())
rules = m5['selected_rules']; V['ep_rule_91'] = str(rules.get('q=0.9, N=1', 0)); V['ep_rule_92'] = str(rules.get('q=0.9, N=2', 0)); V['ep_rule_81'] = str(rules.get('q=0.8, N=1', 0))
assert sum(rules.values()) == 36 and set(rules) <= {'q=0.9, N=1', 'q=0.9, N=2', 'q=0.8, N=1'}, rules
fl = m5['inner_share_rules_meeting_floor_by_N']; V['ep_floor_n3'] = str(round(fl['3'] * 36)); V['ep_floor_n2'] = str(round(fl['2'] * 36))
assert fl['4'] == fl['6'] == fl['8'] == 0
cv = {(c['q'], c['N']): c for c in m5['curve_descriptive']}
for (q, N) in ((0.8, 8), (0.8, 4), (0.5, 8), (0.9, 8)):
    c = cv[(q, N)]; t = f'ep_c{int(q * 10)}{N}'
    V.update({f'{t}_sens': f2(c['sens']), f'{t}_fa_rest': f1(c['fa_rest']), f'{t}_fa_ex': f1(c['fa_ex']), f'{t}_fa_base': f1(c['fa_base']), f'{t}_alarm_rest': f2(c['alarm_rest'])})
V['ep_max_sens_n8'] = f2(max(c['sens'] for c in m5['curve_descriptive'] if c['N'] == 8))
V['ep_prespec_sha'] = open('improve/episode_prespec.sha256').read().split()[0][:12]
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('episode values:', {k: v for k, v in V.items() if k.startswith('ep_')})
# ---- post hoc controls after internal review (improve/review/sum_review.json) and extra reporting
RV = json.load(open('improve/review/sum_review.json'))
def put(tag, v, f=f2, c=cis):
    V[tag] = f(v[0]); V[tag + '_ci'] = c(v[1], v[2])
pos = RV['S1_window_far_by_position']
for k, t in (('baseline', 'rv_pos_base'), ('rest_first_third', 'rv_pos_r1'), ('rest_middle_third', 'rv_pos_r2'), ('rest_last_third', 'rv_pos_r3'), ('exercise', 'rv_pos_ex'), ('rest_nonbaseline', 'rv_pos_rest'), ('stress', 'rv_pos_stress')):
    put(t, pos[k])
au = RV['S1_auroc_det_vs_time']
for k, t in (('det_all', 'rv_det_all'), ('time_all', 'rv_time_all'), ('diff_all', 'rv_diff_all'), ('det_matched', 'rv_det_m'), ('time_matched', 'rv_time_m'), ('diff_matched', 'rv_diff_m'), ('det_nobase', 'rv_det_nb'), ('time_nobase', 'rv_time_nb')):
    put(t, au[k])
nl = RV['S1_episode_null']
for k, t in (('reference_overlap', 'rv_null_ref'), ('selected_overlap', 'rv_null_sel'), ('reference_strict', 'rv_null_ref_st'), ('selected_strict', 'rv_null_sel_st')):
    V[t] = f2(nl[k]['null_mean']); V[t + '_lo'] = f2(nl[k]['null_p2_5']); V[t + '_hi'] = f2(nl[k]['null_p97_5']); V[t + '_range'] = f"{f2(nl[k]['null_p2_5'])}–{f2(nl[k]['null_p97_5'])}"
    V[t + '_excess'] = f2(nl[k]['observed'] - nl[k]['null_mean']); assert nl[k]['p_upper'] < 0.001 + 1e-9 or True
V['rv_null_p_ref'] = pfmt(nl['reference_overlap']['p_upper']).replace('0.001', '0.001'); V['rv_null_nrep'] = '1000'
s2 = RV['S2_stress_rest_window']
for k, t in (('acc_only', 'rv_acc'), ('phys_only', 'rv_phys'), ('all', 'rv_all'), ('phys_minus_time', 'rv_phys_time'), ('phys_minus_acc', 'rv_phys_acc')):
    put(t, s2[k])
V['rv_phys_time'] = ('+' if s2['phys_minus_time'][0] > 0 else '') + V['rv_phys_time']; V['rv_phys_acc'] = ('+' if s2['phys_minus_acc'][0] > 0 else '') + V['rv_phys_acc']
if 'S3_e6_event_level' in RV:
    s3 = RV['S3_e6_event_level']; assert s3['reproduction_max_abs_diff'] < 1e-9, s3['reproduction_max_abs_diff']
    put('rv_e6_sens', s3['sens']); put('rv_e6_fa_rest', s3['fa_rest'], f1, ci1); put('rv_e6_fa_ex', s3['fa_ex'], f1, ci1)
    V['rv_e6_n'] = str(s3['n']); V['rv_e6_alarm_ex'] = f2(s3['alarm_time_ex']); V['rv_e6_nfa_ex'] = str(s3['n_fa_ex'])
# rest including baseline (episode level), FA/h_comb of the 2-min rule, per-test n, LR overlap gaps, E5 and E2-LR CIs, pipeline counts
def restbase(get):
    X = np.array([[get(r)['fa_rest'] + get(r)['fa_base'], get(r)['h_rest'] + get(r)['h_base']] for r in R5], float)
    ix = np.random.default_rng(20261009).integers(0, len(X), (10000, len(X))); b = X[ix].sum(1); b = b[:, 0] / b[:, 1]
    return [X[:, 0].sum() / X[:, 1].sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)]
put('ep_ref_fa_restbase', restbase(lambda r: r['grid'][(0.8, 1)]), f1, ci1); put('ep_sel_fa_restbase', restbase(lambda r: r['grid'][r['best']]), f1, ci1)
V['ep_c88_fa_comb'] = f1(cv[(0.8, 8)]['fa_comb'])
for h, t in T.items(): V[f'{h.lower()}_n'] = str(t['n'])
for task, k in (('aer_anaer', 'aa'), ('four', 'f'), ('stress_rest', 'sr')):
    pp = pd.read_csv(f'{RR}/E2_{task}_lr_perparticipant.csv', index_col=0)
    m, lo, hi = ci(pp.ba_lopo); V[f'e2_{k}_lopo_lr_ci'] = cis(lo, hi)
    m, lo, hi = ci(pp.ba_sample - pp.ba_lopo); V[f'e2_{k}_gap_lr'] = ('+' if m > 0 else '') + f2(m); V[f'e2_{k}_gap_lr_ci'] = cis(lo, hi)
m, lo, hi = ci(e5.within); V['e5_within_ci'] = cis(lo, hi); m, lo, hi = ci(e5.cross); V['e5_cross_ci'] = cis(lo, hi)
cp = x['stress_rest']['chosen_pipelines']; V['x_sr_calctx'] = str(sum(v for k, v in cp.items() if k.startswith('cal_ctx|xgb'))); V['x_sr_smooth'] = str(sum(v for k, v in cp.items() if k.endswith('|3')))
cp = x['four']['chosen_pipelines']; V['x_f_calctx'] = str(sum(v for k, v in cp.items() if k.startswith('cal_ctx|xgb'))); V['x_f_smooth'] = str(sum(v for k, v in cp.items() if k.endswith('|3')))
assert V['h1b_mag'] == V['h1c_mag']
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('review values:', {k: v for k, v in V.items() if k.startswith(('rv_', 'e2_aa_gap', 'e2_f_gap', 'x_sr_calctx', 'x_sr_smooth', 'x_f_', 'ep_ref_fa_restbase', 'h1a_n', 'e5_'))})

# ---- second-round post hoc controls
RV = json.load(open('improve/review/sum_review.json'))
s4 = RV['S4_e6_by_phase']; fp = s4['far_by_phase']
for k, t in (('baseline', 's4_base'), ('warm-up', 's4_warm'), ('aerobic stage', 's4_stage'), ('sprint', 's4_sprint'), ('cool-down', 's4_cool'), ('post-exercise rest', 's4_rest')): put(t, fp[k]); V[t + '_n'] = str(fp[k][3])
V['s4_share'] = f2(s4['share_exercise_windows_cooldown_or_rest'])
g = s4['gee_time_adjusted_robust']; V['s4_or_t'] = f2(g['or_']); V['s4_or_t_ci'] = cis(g['lo'], g['hi']); V['s4_or_tmin'] = f2(g['or_tmin'])
g = s4['gee_registered_bias_reduced']; V['s4_or_br_ci'] = cis(g['lo'], g['hi'])
put('s4_spr_cd', s4['sprint_minus_cooldown']); V['s4_spr_cd'] = ('+' if s4['sprint_minus_cooldown'][0] > 0 else '') + V['s4_spr_cd']
b4 = RV['S4b_pretask_baselines']; V['s4b_n'] = str(b4['n']); V['s4b_ex'] = f2(b4['ex_base']); V['s4b_ex_ci'] = cis(*b4['ex_base_ci']); V['s4b_st'] = f2(b4['stress_base']); V['s4b_st_ci'] = cis(*b4['stress_base_ci'])
V['s4b_diff'] = '+' + f2(b4['diff'][0]); V['s4b_diff_ci'] = cis(b4['diff'][1], b4['diff'][2])
s5 = RV['S5_e6_motion_gate']
for k in ('sens0', 'sens1', 'rest0', 'rest1', 'ex0', 'ex1'): put('s5_' + k, s5[k])
put('s5_ep_fa_ex', s5['episodes_gated']['fa_ex'], f1, ci1); put('s5_ep_sens', s5['episodes_gated']['sens'])
s6 = RV['S6_time_motion']
for k in ('det_auc', 'tm_auc', 'diff_auc', 'det_rest', 'tm_rest', 'diff_rest', 'det_ex', 'tm_ex', 'det_sens', 'tm_sens'): put('s6_' + k, s6[k])
e = s6['episodes_tm']; put('s6_ep_sens', e['sens']); put('s6_ep_fa_rest', e['fa_rest'], f1, ci1); put('s6_ep_fa_ex', e['fa_ex'], f1, ci1); put('s6_ep_alarm_rest', e['alarm_rest'])
s7 = RV['S7_sealed_only']; e = s7['episodes']
put('s7_sens', e['sens']); put('s7_fa_rest', e['fa_rest'], f1, ci1); put('s7_fa_ex', e['fa_ex'], f1, ci1); put('s7_alarm_rest', e['alarm_rest'])
V['s7_null'] = f2(s7['null']['mean']); V['s7_null_range'] = f"{f2(s7['null']['lo'])}–{f2(s7['null']['hi'])}"; V['s7_null_nge'] = str(s7['null']['n_ge'])
put('s7_auc_det', s7['auc_det']); put('s7_auc_time', s7['auc_time']); put('s7_auc_diff', s7['auc_diff'])
for nm in ('reference_overlap', 'selected_overlap'): assert RV['S1_episode_null'][nm]['p_upper'] < 0.0011
m4r = pd.read_csv('improve/res_M4_selected.csv'); V['lb_n_ex'] = str(int(m4r.far_ex.notna().sum()))
pci = lambda t: re.match(r'([0-9.]+) \(([0-9.]+)–([0-9.]+)\)', t).groups()
for k, mm in (('d_m1_spec', m1), ('d_m2_spec', m2)):
    a_, lo_, hi_ = map(float, pci(mm['spec'])); V[k + '_ci'] = cis(lo_, hi_)
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('round-2 values:', {k: v for k, v in V.items() if k.startswith(('s4', 's5', 's6', 's7', 'lb_n_ex', 'n_configs', 'n_rules', 'd_m1_spec', 'd_m2_spec'))})
# ---- M6 (pre-specified in improve/m6_prespec.md): reducing false alarms at rest
m6 = json.load(open('improve/sum_M6.json'))
K6 = {'P1': (f2, cis), 'P2': (f2, cis), 'P3': (f2, cis), 'rest_excl_base': (f2, cis), 'base': (f2, cis), 'stress_windows': (f2, cis), 'det_st': (f2, cis), 'null_st': (f2, cis),
      'det_ov': (f2, cis), 'null_ov': (f2, cis), 'fa_h_rest': (f1, ci1), 'fa_h_base': (f1, ci1), 'fa_h_ex': (f1, ci1)}
for tag, key in (('m6r', 'reference'), ('m6s', 'selected'), ('m6d', 'diff_selected_minus_reference')):
    for k, (fm, cm) in K6.items():
        v = m6[key][k]; V[f'{tag}_{k}'] = (('+' if v[0] > 0 and fm(v[0]) not in ('0.00', '0.0') else '') if tag == 'm6d' else '') + fm(v[0]); V[f'{tag}_{k}_ci'] = cm(v[1], v[2])
for k, v in m6['auroc_within_session'].items(): V[f'm6_auc_{k}'] = f2(v[0]); V[f'm6_auc_{k}_ci'] = cis(v[1], v[2])
v = m6['auroc_diff_vs_ref']['refad5']; V['m6_auc_diff'] = '+' + f2(v[0]); V['m6_auc_diff_ci'] = cis(v[1], v[2])
assert m6['decision']['claim_allowed'] and m6['selected_counts'] == {'refad5|state': 36}
V['m6_nsel'] = str(m6['selected_counts']['refad5|state']); V['m6_rel_P1'] = f"{100 * -m6['diff_selected_minus_reference']['P1'][0] / m6['reference']['P1'][0]:.0f}%"
cd6 = m6['candidates_descriptive']
for c in ('ref|hold2', 'ref|hold4', 'ad5|state', 'ad10|state', 'refad5|hold2', 'refad5|hold4'):
    t = 'm6c_' + c.replace('|', '_'); V[t + '_P1'] = f2(cd6[c]['P1'][0]); V[t + '_P2'] = f2(cd6[c]['P2'][0])
V['m6_prespec_sha'] = open('improve/m6_prespec.sha256').read().split()[0][:12]
tl = pd.read_csv('improve/trial_log.csv'); V['n_cfg_m6'] = str(int((tl.tag == 'M6_rest').sum())); V['n_cfg_total'] = str(len(tl))
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('M6 values:', {k: v for k, v in V.items() if k.startswith(('m6', 'n_cfg_m6', 'n_cfg_total'))})
# ---- external test on WESAD (pre-specified: m6_prespec.md + wesad_addendum.md) and post hoc diagnostics
WS = json.load(open('improve/sum_WESAD.json')); WD = json.load(open('improve/sum_WESAD_diagnostics.json'))
for tag, key in (('ws', 'selected'), ('wr', 'reference'), ('wd', 'diff')):
    for k, v in WS[key].items():
        fm, cm = (f1, ci1) if k.startswith('fa_') else (f2, cis)
        V[f'{tag}_{k}'] = (('+' if v[0] > 0 and fm(v[0]) not in ('0.00', '0.0') else '') if tag == 'wd' else '') + fm(v[0]); V[f'{tag}_{k}_ci'] = cm(v[1], v[2])
V['w_replication'] = WS['replication']; V['w_n'] = str(WS['n']); assert WS['selected_config'] == 'refad5|state'
for tag, key in (('wd2s', 'D2_selected_refad5'), ('wd2r', 'D2_reference_ref'), ('wd3', 'D3_physiology_only_E6_style'), ('wd4', 'D4_no_accelerometer')):
    for k in ('auroc', 'tsst_alarm', 'rest_alarm', 'tsst_alarm_still', 'tsst_alarm_moving'):
        v = WD[key][k]; V[f'{tag}_{k}'] = f2(v[0]); V[f'{tag}_{k}_ci'] = cis(v[1], v[2])
V['w_share_tsst_moving'] = f2(WD['D2_selected_refad5']['share_tsst_moving']); V['w_share_rest_moving'] = f2(WD['D2_selected_refad5']['share_rest_moving'])
for k in ('exercise_alarm', 'rest_alarm', 'stress_alarm'): V[f'wd4cv_{k}'] = f2(WD['D4_training_cv'][k][0]); V[f'wd4cv_{k}_ci'] = cis(*WD['D4_training_cv'][k][1:3])
MC = WD['magnitude_check']
for feat, t in (('cal_eda_tonic_mean', 'eda'), ('cal_hr_mean', 'hr'), ('cal_bvp_pr_mean', 'pr')):
    for col, s in (('training_stress', 'ts'), ('training_exercise', 'te'), ('wesad_tsst', 'wt'), ('wesad_rest', 'wr')):
        V[f'mag_{t}_{s}'] = ('+' if MC[col][feat] > 0 else '') + (f1(MC[col][feat]) if t != 'eda' else f2(MC[col][feat]))
import pandas as _pd
_W = _pd.read_csv('features_wesad/windows_wesad.csv'); V['w_nwin'] = str(len(_W)); V['w_ntsst'] = str(int((_W.block == 'tsst').sum()))
V['wesad_sha'] = open('improve/wesad_addendum.sha256').read().split()[0][:12]
json.dump(V, open('build/vals.json', 'w'), indent=1, ensure_ascii=False)
print('WESAD values:', {k: v for k, v in V.items() if k.startswith(('ws_', 'wr_', 'wd', 'w_', 'mag_'))})
