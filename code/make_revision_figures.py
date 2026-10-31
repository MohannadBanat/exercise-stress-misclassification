"""Figures added or redrawn for the revised manuscript (major revision, v1.1.0): Figure 1 (overview incl. exploratory analyses), Figure 3
(definitions of windows, episodes and alarm events; illustrative, synthetic signal), Figure 8 (post hoc ablations and
response magnitude; from improve/review/sum_r1_ablations.json and sum_r1_magnitude.json). 600 dpi, into fig/."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'TeX Gyre Pagella', 'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'savefig.dpi': 600, 'axes.spines.top': False, 'axes.spines.right': False})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#b9b8b2'

# ---------------------------------------------------------------- Figure 1: overview
fig, ax = plt.subplots(figsize=(6.5, 4.1)); ax.set_xlim(-1, 101); ax.set_ylim(5, 86); ax.axis('off')
def box(x, y, w, h, title, body, fc='#f3f2ee', tsz=7.3, bsz=6.2, dy=4.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + 1.0, y + h - 1.3, title, fontsize=tsz, fontweight='bold', va='top', color=INK, linespacing=1.12)
    ax.text(x + 1.0, y + h - dy, body, fontsize=bsz, va='top', color=INK, linespacing=1.22)
def arrow(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8))
def tier(y, s): ax.text(0, y, s, fontsize=7.4, fontstyle='italic', color=MUTED, va='bottom')
tier(83.3, 'Data and pipeline, fixed in the registered plan (Sections 3.1–3.4)')
box(0, 70, 22, 12.5, 'A  Data', '36 participants, Empatica E4\nstress (36), aerobic (30),\nanaerobic (31) sessions')
box(26, 70, 22, 12.5, 'B  Quality control', 'Checksums, exclusions,\nsignal-validity rules,\nprotocol-block index')
box(52, 70, 22, 12.5, 'C  Windows and features', '30 s windows, 15 s step\n29 features: EDA, pulse,\nHR, TEMP, ACC')
box(78, 70, 22, 12.5, 'D  Models', 'XGBoost; logistic regression\ntuned on training\nparticipants only')
for x in (22, 48, 74): arrow(x + 0.3, 76.2, x + 3.7, 76.2)
tier(64.3, 'Registered experiments and hypotheses H1–H8 (Section 3.5)')
exps = [('E1', 'Reproduction', 'original protocol:\n80/20 split, 10-fold\nCV, oversampling'), ('E2', 'Participant-\nlevel', 'held-out\nparticipants\n(H1a–H1c)'),
        ('E3', 'Shortcut\nbaselines', 'time, block\nposition, motion\n(H2, H3)'), ('E4', 'Pre-task\nbaselines', 'sessions told\napart before any\ntask? (H4)'),
        ('E5', 'Cross-\ncohort', 'train on one\ncohort, test on\nthe other (H5)'), ('E6', 'Deployment\ntest', 'stress detector\non exercise\n(H6–H8)')]
for i, (k, t, b) in enumerate(exps):
    box(i * 16.8, 45, 15.6, 18, f'{k}  {t}', b, fc='#fdeee7' if k in ('E1', 'E2', 'E6') else '#f3f2ee', tsz=7.0, bsz=6.1, dy=8.4)
tier(39.3, 'Exploratory analyses, each under rules fixed before it was run (Section 3.6)')
xs = [('Deployable\nimprovements', 'calibration, context\nfeatures, smoothing'), ('Exercise-robust\ndetector', '24 development →\n12 sealed participants'),
      ('Episode-level\nevaluation', 'episodes, false alarms\nper hour, chance'), ('Rest false\nalarms', '5- or 10-min\nadaptive reference'),
      ('External test', 'WESAD, 15 new\nparticipants')]
for i, (t, b) in enumerate(xs): box(i * 20.2, 22.5, 18.6, 15.7, t, b, fc='#eaf1fb', tsz=7.0, bsz=6.1, dy=8.2)
for i in range(4): arrow(i * 20.2 + 19.0, 30.3, i * 20.2 + 20.0, 30.3)
box(0, 6.0, 100, 12.5, 'F  Outputs and inference',
    'Balanced accuracy, AUROC, false-alarm rates per participant and exercise phase, episodes detected and false alarms per hour.\n'
    '95% CIs from participant-level bootstrap; Holm–Bonferroni correction across the ten registered tests only.\n'
    'Post hoc controls and ablations, added after the results were known, are labelled post hoc (Section 3.7).', dy=4.9)
fig.tight_layout(pad=0.2); fig.savefig('fig/fig1_overview_v8.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: definitions (illustrative)
dt = 0.25; t = np.arange(0, 14 + 1e-9, dt)                    # decision times every 15 s (minutes)
ep = [(3.0, 5.0, [(3.0, 3.6), (4.3, 5.0)]), (10.0, 11.0, [(10.0, 11.0)])]
s = 0.22 + 0.03 * np.sin(t * 2.1)
def bump(c0, c1, h, w=0.35): return h * (1 / (1 + np.exp(-(t - c0) / w * 4)) - 1 / (1 + np.exp(-(t - c1) / w * 4)))
s = s + bump(2.55, 5.55, 0.42) - bump(3.95, 4.3, 0.2, 0.1) + bump(7.55, 8.15, 0.37, 0.2) + bump(10.4, 11.55, 0.45, 0.25)
tau = 0.5; al = s >= tau
fig = plt.figure(figsize=(6.5, 3.9)); gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.6, 0.75], hspace=0.12)
a0, a1, a2 = [fig.add_subplot(gs[i]) for i in range(3)]
for a in (a0, a1, a2): a.set_xlim(0, 14)
# protocol strip
a0.set_ylim(0, 1); a0.axis('off')
a0.add_patch(Rectangle((0, 0.2), 14, 0.6, fc='#efeeea', ec='none'))
for e0, e1, tasks in ep:
    for k0, k1 in tasks: a0.add_patch(Rectangle((k0, 0.2), k1 - k0, 0.6, fc=ORANGE, ec='none'))
    a0.add_patch(Rectangle((e1, 0.2), 1.0, 0.6, fc='white', ec=ORANGE, hatch='////', lw=0.0, alpha=0.5))
    a0.plot([e0, e1], [0.93, 0.93], color=INK, lw=0.8); a0.text((e0 + e1) / 2, 0.97, 'stress episode', ha='center', va='bottom', fontsize=6.4)
a0.text(1.4, 0.5, 'rest', ha='center', va='center', fontsize=6.6); a0.text(8.9, 0.5, 'rest', ha='center', va='center', fontsize=6.6)
a0.text(13.0, 0.5, 'rest', ha='center', va='center', fontsize=6.6)
a0.text(3.3, 0.5, 'task', ha='center', va='center', fontsize=6.2, color='white'); a0.text(4.65, 0.5, 'task', ha='center', va='center', fontsize=6.2, color='white')
a0.text(10.5, 0.5, 'task', ha='center', va='center', fontsize=6.2, color='white')
a0.text(5.5, 0.5, 'grace\n60 s', ha='center', va='center', fontsize=5.6, color=INK, linespacing=1.0); a0.text(11.5, 0.5, 'grace\n60 s', ha='center', va='center', fontsize=5.6, color=INK, linespacing=1.0)
a0.text(-0.15, 0.5, 'A', fontsize=9, fontweight='bold', ha='right', va='center')
# smoothed probability
a1.plot(t, s, color=BLUE, lw=1.2, zorder=3)
a1.scatter(t[~al], s[~al], s=6, color=BLUE, zorder=4, lw=0); a1.scatter(t[al], s[al], s=11, color=PURPLE, zorder=5, lw=0)
a1.axhline(tau, ls='--', color=INK, lw=0.8); a1.text(13.95, tau + 0.02, 'threshold τ', ha='right', va='bottom', fontsize=6.4)
for e0, e1, _ in ep: a1.axvspan(e0, e1, color=ORANGE, alpha=0.12, lw=0)
a1.set_ylim(0, 1); a1.set_ylabel('Smoothed\nstress probability', fontsize=7); a1.tick_params(labelbottom=False, labelsize=6.5)
a1.grid(axis='y', color=GRID, lw=0.5); a1.set_axisbelow(True)
a1.text(-1.05, 1.0, 'B', fontsize=9, fontweight='bold', ha='right', va='top')
a1.annotate('dip shorter than 120 s:\nruns merged into one event', xy=(4.12, 0.47), xytext=(5.6, 0.86), fontsize=6.2, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
a1.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, marker='o', ms=2.5, label='one decision every 15 s'),
                   Line2D([], [], color=PURPLE, lw=0, marker='o', ms=3.4, label='alarm state (N = 1)')],
          loc='upper right', bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=6.3, handlelength=1.4)
# events
a2.set_ylim(0, 1); a2.spines['left'].set_visible(False); a2.set_yticks([])
runs = []; i = 0
while i < len(t):
    if al[i]:
        j = i
        while j + 1 < len(t) and al[j + 1]: j += 1
        runs.append([t[i], t[j]]); i = j + 1
    else: i += 1
ev = []
for r in runs:
    if ev and r[0] - ev[-1][1] < 2.0: ev[-1][1] = r[1]
    else: ev.append(r[:])
lab = ['episode 1 detected by any overlap,\nnot strictly (alarm already on at onset)', 'false alarm\n(begins in rest)', 'episode 2 detected\n(any overlap and strict)']
for (e0, e1), l in zip(ev, lab):
    a2.add_patch(Rectangle((e0, 0.62), max(e1 - e0, 0.12), 0.26, fc=PURPLE, ec='none'))
    a2.text((e0 + e1) / 2, 0.5, l, ha='center', va='top', fontsize=6.0, linespacing=1.05)
for e0, e1, _ in ep: a2.axvspan(e0, e1, color=ORANGE, alpha=0.12, lw=0)
a2.text(-0.3, 0.75, 'alarm\nevents', ha='right', va='center', fontsize=6.6)
a2.set_xlabel('Time (min)', fontsize=7); a2.tick_params(labelsize=6.5)
a2.text(-1.05, 1.0, 'C', fontsize=9, fontweight='bold', ha='right', va='top')
fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.11)
fig.savefig('fig/fig3_definitions_v8.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 8: ablations and response magnitude
A = json.load(open('improve/review/sum_r1_ablations.json')); M = json.load(open('improve/review/sum_r1_magnitude.json'))
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
mets = [('exercise_far', 'Exercise windows\nin alarm'), ('rest_far', 'Rest windows\nin alarm'), ('within_session_auroc', 'Within-session\nAUROC')]
sets = [('full', 'Full detector', BLUE), ('physiology_only', 'Physiology only', ORANGE), ('motion_only', 'Motion only', GREEN)]
for k, (key, lab, col) in enumerate(sets):
    r = A[f'sealed_{key}']
    for j, (m, _) in enumerate(mets):
        v = r[m]; x = j + (k - 1) * 0.22
        b1.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt='o', color=col, ms=4, capsize=1.8, lw=0.9, label=lab if j == 0 else None)
b1.set_xticks(range(3)); b1.set_xticklabels([l for _, l in mets], fontsize=6.6); b1.set_ylim(0, 1); b1.set_ylabel('Fraction or AUROC', fontsize=7)
b1.grid(axis='y', color=GRID, lw=0.5); b1.set_axisbelow(True); b1.tick_params(labelsize=6.5)
b1.legend(frameon=False, fontsize=6.3, loc='upper left', handletextpad=0.2)
b1.set_title('A  Sealed test (12 participants)', loc='left', fontsize=7.6, fontweight='bold')
bins = ['<10 bpm', '10-20 bpm', '>=20 bpm']; xl = ['< 10', '10–20', '≥ 20']
for key, col, lab in (('reference', BLUE, 'reference'), ('adaptive_reference', ORANGE, '+ 5-min adaptive')):
    for src, ls, mk, mf, nm in (('train_cv36_stress', '-', 'o', col, 'training CV'), ('wesad_tsst', '--', 's', 'white', 'WESAD TSST')):
        y = [M[f'{src}_{key}']['cal_bvp_pr_mean'][b]['alarm'] for b in bins]
        b2.plot(range(3), y, ls=ls, color=col, marker=mk, mfc=mf, mec=col, ms=4, lw=1.0, label=f'{nm}, {lab}')
b2.set_xticks(range(3)); b2.set_xticklabels(xl, fontsize=6.6); b2.set_xlabel('Pulse-rate rise from baseline (beats/min)', fontsize=7)
b2.set_ylim(0, 1); b2.set_ylabel('Stress windows in alarm', fontsize=7); b2.grid(axis='y', color=GRID, lw=0.5); b2.set_axisbelow(True); b2.tick_params(labelsize=6.5)
b2.legend(frameon=False, fontsize=5.9, loc='center right', bbox_to_anchor=(1.0, 0.55), handlelength=2.2)
b2.set_title('B  Detection by response size', loc='left', fontsize=7.6, fontweight='bold')
fig.tight_layout(w_pad=1.5); fig.savefig('fig/fig8_ablation_v8.png'); plt.close(fig)
print('ok')
