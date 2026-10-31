"""Values for the post hoc numbers added in the revision (Section 4.11, Tables 11-12), read from the result files (no number is typed by hand)."""
import json
from decimal import Decimal, ROUND_HALF_UP
R = 'improve/review/'
A = json.load(open(R + 'sum_r1_ablations.json')); M = json.load(open(R + 'sum_r1_magnitude.json'))
f2 = lambda x: str(Decimal(repr(float(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
ci = lambda v: f'{f2(v[0])} ({f2(v[1])} to {f2(v[2])})'
V = {}
for src, pre in (('cv36', 'cv'), ('sealed', 'se')):
    for k, s in (('full', 'full'), ('physiology_only', 'phys'), ('motion_only', 'mot')):
        r = A[f'{src}_{k}']
        for m, short in (('stress_sens', 'sens'), ('rest_far', 'rest'), ('exercise_far', 'ex'), ('within_session_auroc', 'au')):
            V[f'{pre}_{s}_{short}'] = f2(r[m][0]); V[f'{pre}_{s}_{short}_ci'] = ci(r[m])
for k, s in (('reference_physiology_only', 'w_ref_phys'), ('motion_only', 'w_mot')):
    r = A[f'wesad_{k}']; V[s] = f2(r['auroc'][0]); V[s + '_ci'] = ci(r['auroc']); V[s + '_tsst'] = f2(r['tsst_alarm'][0]); V[s + '_rest'] = f2(r['rest_alarm'][0])
pr = 'cal_bvp_pr_mean'; B = ['<10 bpm', '10-20 bpm', '>=20 bpm']
tr_ref, tr_ad = M['train_cv36_stress_reference'][pr], M['train_cv36_stress_adaptive_reference'][pr]
assert f2(tr_ref[B[2]]['alarm']) == f2(tr_ad[B[2]]['alarm']), 'training >=20 bpm detection differs between detectors'
V['tr_hi'] = f2(tr_ref[B[2]]['alarm']); V['tr_hi_n'] = str(tr_ref[B[2]]['n_windows']); V['tr_hi_p'] = str(tr_ref[B[2]]['n_participants'])
V['tr_hi_frac'] = f"{round(100 * tr_ref[B[2]]['n_windows'] / sum(tr_ref[b]['n_windows'] for b in B))}%"
w_ref, w_ad = M['wesad_tsst_reference'][pr], M['wesad_tsst_adaptive_reference'][pr]
for b, k in zip(B, ('lo', 'mid', 'hi')): V[f'w_{k}_ref'] = f2(w_ref[b]['alarm']); V[f'w_{k}_ad'] = f2(w_ad[b]['alarm'])
V['w_hi_frac'] = f"{round(100 * w_ref[B[2]]['n_windows'] / sum(w_ref[b]['n_windows'] for b in B))}%"
e = M['wesad_tsst_reference']['cal_eda_tonic_mean']
for i, b in enumerate(['<1 uS', '1-3 uS', '>=3 uS']): V[f'w_eda{i + 1}'] = f2(e[b]['alarm'])
json.dump(V, open('build/vals_revision.json', 'w'), indent=1, sort_keys=True); print(json.dumps(V, indent=0, sort_keys=True))
