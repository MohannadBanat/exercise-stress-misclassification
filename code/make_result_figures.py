"""Result figures for the manuscript (300 dpi). Reads results_real/ and results_x/ (exploratory)."""
import os, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'TeX Gyre Pagella', 'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
RR = 'results_real'; RX = 'results_x'; os.makedirs('fig', exist_ok=True)
rng = np.random.default_rng(20261008)
def ci(v, n=10000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def panel(ax, s): ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
TASK = {'stress_rest': 'Stress vs. rest', 'aer_anaer': 'Aerobic vs.\nanaerobic', 'four': 'Four-class'}
PUB = {'stress_rest': 0.93, 'aer_anaer': 0.91, 'four': 0.84}

def fig3():
    d = pd.read_csv(f'{RR}/E1_runs.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axs[0]; xs = np.arange(3)
    for j, (v, col, lab) in enumerate([('clean', C1, 'Oversampling inside training'), ('leaky', C2, 'Oversampling before split')]):
        for i, t in enumerate(TASK):
            a = d[(d.task == t) & (d.variant == v)].accuracy.values
            bp = ax.boxplot(a, positions=[i + (j - 0.5) * 0.32], widths=0.26, patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', lw=1.2), whiskerprops=dict(color=col), capprops=dict(color=col), boxprops=dict(facecolor=col, edgecolor=col))
        ax.plot([], [], 's', color=col, label=lab)
    for i, t in enumerate(TASK): ax.plot([i - 0.36, i + 0.36], [PUB[t]] * 2, color=INK, lw=1.2, ls='--')
    ax.plot([], [], '--', color=INK, label='Published value')
    ax.set_xticks(xs); ax.set_xticklabels(TASK.values()); ax.set_ylabel('Test accuracy (100 random splits)'); ax.set_ylim(0.7, 1.01)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); panel(ax, 'A')
    ax = axs[1]
    for i, t in enumerate(TASK):
        pp = pd.read_csv(f'{RR}/E2_{t}_xgb_perparticipant.csv', index_col=0)
        for j, (c, col, lab) in enumerate([('ba_sample', C2, 'Sample-level 10-fold CV'), ('ba_lopo', C1, 'Leave-one-participant-out')]):
            m, lo, hi = ci(pp[c]); x = i + (j - 0.5) * 0.3
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', color=col, ms=5, capsize=2, lw=1, label=lab if i == 0 else None)
            ax.text(x + 0.05, m, f'{m:.2f}', fontsize=6.5, va='center', ha='left', color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels(TASK.values()); ax.set_ylabel('Balanced accuracy per participant'); ax.set_ylim(0.7, 1.0)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); panel(ax, 'B')
    fig.set_size_inches(6.5, 3.0); fig.tight_layout(); fig.savefig('fig/fig3_e1_e2.png'); plt.close(fig)

def fig4():
    pp = pd.read_csv(f'{RR}/E3_four_perparticipant.csv', index_col=0)
    r = json.load(open(f'{RR}/results.json'))
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; names = [('full', 'All 29 features'), ('acc_only', 'Accelerometer only'), ('block_position', 'Block position + cohort'), ('time_only', 'Time in session only')]
    for i, (k, lab) in enumerate(names):
        m, lo, hi = ci(pp[k]); col = C1 if k == 'full' else (C2 if k == 'acc_only' else MUTED)
        ax.barh(i, m, color=col, height=0.55); ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color=INK, capsize=2, lw=0.8)
        ax.text(hi + 0.015, i, f'{m:.2f} ({lo:.2f}–{hi:.2f})', va='center', fontsize=6.5, color=MUTED)
    ax.axvline(0.25, color=INK, ls='--', lw=0.8); ax.text(0.26, -0.55, 'chance', fontsize=6.5, color=MUTED)
    ax.set_yticks(range(4)); ax.set_yticklabels([n for _, n in names]); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel('Four-class balanced accuracy (LOPO, 30 s windows)'); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); panel(ax, 'A')
    ax = axs[1]; e4 = r['E4']
    # reconstruct the permutation null summary (mean and 95th percentile are stored); show as reference band
    ax.bar([0, 1], [e4['mean_participant_ba'], e4['X1_aer_vs_anaer_baseline_ba']], color=[C2, MUTED], width=0.55)
    ax.axhline(0.5, color=INK, ls='--', lw=0.8); ax.fill_between([-0.45, 0.45], 0, e4['null_95'], color=GRID, zorder=0)
    ax.plot([-0.45, 0.45], [e4['null_95']] * 2, color=MUTED, lw=0.8)
    for x, v, n in [(0, e4['mean_participant_ba'], e4['n_participants']), (1, e4['X1_aer_vs_anaer_baseline_ba'], e4['X1_n'])]:
        ax.text(x, v + 0.06, f'{v:.2f}\n(n = {n})', ha='center', fontsize=6.5, color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Stress vs.\nexercise (H4)', 'Aerobic vs.\nanaerobic (X1)']); ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1); ax.set_ylabel('Balanced accuracy, pre-task\nbaseline windows (V2)'); panel(ax, 'B')
    fig.tight_layout(); fig.savefig('fig/fig4_e3_e4.png'); plt.close(fig)

def fig5():
    e6 = pd.read_csv(f'{RR}/E6_perparticipant.csv'); r = json.load(open(f'{RR}/results.json'))['E6']
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axs[0]; b = e6[~e6.aware]; a = e6[e6.aware]
    items = [('Rest (held-out)', b.far_rest, MUTED), ('Exercise', b.far_ex, C2), ('Exercise,\nexercise-aware', a.far_ex, C1)]
    for i, (lab, v, col) in enumerate(items):
        m, lo, hi = ci(v); ax.bar(i, m, color=col, width=0.6); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=2, lw=0.8)
        ax.text(i, hi + 0.03, f'{m:.2f}', ha='center', fontsize=6.5, color=MUTED)
    ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in items], fontsize=7); ax.set_ylim(0, 1)
    ax.set_ylabel('False-alarm rate (fraction of windows)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); panel(ax, 'A')
    ax = axs[1]; fi = {int(float(k)): v for k, v in r['far_by_intensity'].items()}
    labs = ['Warm-up', 'Aerobic\n60–75 rpm', 'Aerobic\n80–95 rpm', 'Aerobic\nhighest', 'Sprint']
    ax.plot(list(fi.keys()), list(fi.values()), '-o', color=C2, ms=5, lw=1.5)
    for k, v in fi.items(): ax.text(k, v + 0.04, f'{v:.2f}', ha='center', fontsize=6.5, color=MUTED)
    ax.axhline(b.far_rest.mean(), color=MUTED, ls='--', lw=0.8); ax.text(-0.35, b.far_rest.mean() + 0.03, 'rest windows', fontsize=6.5, color=MUTED, va='bottom')
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=7); ax.set_ylim(0, 1); ax.set_xlim(-0.4, 4.5)
    ax.set_ylabel('False-alarm rate (pooled windows)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); panel(ax, 'B')
    fig.tight_layout(); fig.savefig('fig/fig5_e6.png'); plt.close(fig)

def fig6():
    x = json.load(open(f'{RX}/exploratory.json'))
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axs[0]
    for i, t in enumerate(['stress_rest', 'four']):
        pp = pd.read_csv(f'{RX}/X_{t}_perparticipant.csv', index_col=0)
        for j, (c, col, lab) in enumerate([('ba_reference', MUTED, 'Reference (raw features, XGBoost)'), ('ba_improved', C1, 'Selected pipeline (nested)')]):
            m, lo, hi = ci(pp[c]); xx = i + (j - 0.5) * 0.3
            ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], fmt='o', color=col, ms=5, capsize=2, lw=1, label=lab if i == 0 else None)
            ax.text(xx + 0.05, m, f'{m:.2f}', fontsize=6.5, va='center', color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Stress vs. rest', 'Four-class']); ax.set_ylim(0.4, 1.0); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel('Balanced accuracy (LOPO)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left'); panel(ax, 'A')
    ax = axs[1]; dp = pd.read_csv(f'{RX}/X_deploy_perparticipant.csv'); e6 = pd.read_csv(f'{RR}/E6_perparticipant.csv'); b = e6[~e6.aware]
    rows = [('Registered detector\n(physiology only)', b.sens, b.far_ex, MUTED), ('Selected exercise-robust\ndetector (exploratory)', dp.sens, dp.far_ex, C1)]
    for i, (lab, s, f, col) in enumerate(rows):
        for j, (v, mk, nm) in enumerate([(s, 'o', 'Stress sensitivity'), (f, 's', 'Exercise false-alarm rate')]):
            m, lo, hi = ci(v); yy = i + (j - 0.5) * 0.3
            ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt=mk, color=col if j == 0 else C2, ms=5, capsize=2, lw=1, label=nm if i == 0 else None)
            ax.text(hi + 0.02, yy, f'{m:.2f}', va='center', fontsize=6.5, color=MUTED)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.set_xlim(0, 1.08); ax.invert_yaxis()
    ax.set_xlabel('Fraction of windows (held-out participants)'); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2); panel(ax, 'B')
    fig.tight_layout(); fig.savefig('fig/fig6_exploratory.png'); plt.close(fig)


def fig7():
    m3 = json.load(open('improve/sum_M3.json')); m4 = json.load(open('improve/sum_M4.json'))
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    rows = [('Development, reference\n(nested LOPO, n = 24)', m3['reference']), ('Development, selected\n(nested LOPO, n = 24)', m3['selected']), ('Sealed test, final\ndetector (n = 12)', m4['selected'])]
    mets = [('sens', 'Sensitivity', C1, 'o'), ('far_rest', 'FAR, rest', MUTED, 's'), ('far_ex', 'FAR, exercise', C2, 'D')]
    for i, (lab, s) in enumerate(rows):
        for j, (k, nm, col, mk) in enumerate(mets):
            m, lo, hi = s[k]; y = i + (j - 1) * 0.22
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=4.5, capsize=2, lw=1, label=nm if i == 0 else None)
            ax.text(hi + 0.02, y, f'{m:.2f}', va='center', fontsize=6, color=MUTED)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_xlabel('Fraction of windows'); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=3); panel(ax, 'A')
    ax = axs[1]; cv = pd.DataFrame(m4['operating_curve_lockbox'])
    ax.plot(cv.sens, cv.far_rest, '-s', color=MUTED, ms=4, lw=1.4, label='Rest (non-baseline)')
    ax.plot(cv.sens, cv.far_ex, '-D', color=C2, ms=4, lw=1.4, label='Exercise')
    pre = cv[cv.target_sens == 0.8].iloc[0]
    ax.annotate('pre-specified\nthreshold', xy=(pre.sens, pre.far_rest), xytext=(pre.sens - 0.2, pre.far_rest + 0.22), fontsize=6.3, color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    for _, r in cv.iterrows(): ax.text(r.sens + 0.012, r.far_rest - 0.06, f'{r.target_sens:.1f}', fontsize=5.8, color=MUTED, ha='center')
    ax.set_xlim(0.65, 1.0); ax.set_ylim(0, 1); ax.set_xlabel('Sensitivity on sealed participants'); ax.set_ylabel('False-alarm rate')
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.5, loc='upper left'); panel(ax, 'B')
    fig.tight_layout(); fig.savefig('fig/fig7_detector.png'); plt.close(fig)
def fig8():
    """Episode-level evaluation (M5). A: example held-out participant (closest to the median fraction of rest time in alarm; ties -> first id).
    B: descriptive operating curve over the 30 pre-listed rules on outer LOPO predictions. C: scoring variants for reference and selected rules."""
    import sys, pickle; sys.path.insert(0, 'improve')
    from framework2 import add_flags, detector_frame, prev_index, smooth, train_predict
    from build_features import build
    from run_m5_episode import build_streams, events_for_stream
    m5 = json.load(open('improve/sum_M5.json')); R = pickle.load(open('improve/m5_all.pkl', 'rb'))
    ar = pd.Series({r['pid']: r['grid'][(0.8, 1)]['al_rest'] / r['grid'][(0.8, 1)]['n_rest'] for r in R}); dist = (ar - ar.median()).abs()
    pid = sorted(dist[dist == dist.min()].index)[0]; r = [x for x in R if x['pid'] == pid][0]; tau = r['taus'][0.8]
    W, FS, ACC = build(); d = add_flags(detector_frame(W, final=True)); seg = pd.read_csv('data/segments.csv')
    prev = {45: prev_index(d, 45)}; st = build_streams(d, seg); tr, te = d[d.participant != pid], d[d.participant == pid]
    p = smooth(train_predict(tr, te, FS['v1_cal_ctx'], ACC, 'et', 'single', 'all'), prev, te.index.values, 45)
    S = st[('STRESS', pid)]; Sf = np.full(len(d), np.nan); Sf[te.index.values] = p; sc = Sf[S['rows']]
    al, ev = events_for_stream(S['start'], S['dec'], S['epi'], sc >= tau, 1)
    # v37 layout: three full-width panels stacked; B as small multiples (one facet per persistence N)
    FS_T, FS_L = 7.6, 7.6
    fig = plt.figure(figsize=(6.5, 8.35))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.62, 0.92], hspace=0.6, left=0.095, right=0.985, top=0.975, bottom=0.075)
    ax = fig.add_subplot(gs[0]); t0 = S['dec'].min() - 30; axA = ax
    sg = seg[(seg.protocol == 'STRESS') & (seg.participant == pid)]
    NAME = {'baseline': 'Baseline', 'stroop': 'Stroop', 'rest1': 'Rest', 'tmct': 'TMCT', 'rest2': 'Rest', 'real_opinion': 'O', 'opposite_opinion': 'O', 'subtract': 'S'}
    for _, b in sg.iterrows():
        x0, x1 = (b.start_s - t0) / 60, (b.end_s - t0) / 60
        if b.block in ('stroop', 'tmct', 'real_opinion', 'opposite_opinion', 'subtract'): ax.axvspan(x0, x1, color=C2, alpha=0.18, lw=0)
        elif b.block == 'baseline': ax.axvspan(x0, x1, color=MUTED, alpha=0.10, lw=0)
        ax.text((x0 + x1) / 2, 1.02, NAME[b.block], ha='center', va='bottom', fontsize=7.2, color=MUTED, transform=ax.get_xaxis_transform())
    for _, _, a, b in S['episodes']: ax.axvspan((b - t0) / 60, (b + 60 - t0) / 60, color=C2, alpha=0.06, lw=0, hatch='////', ec=C2)
    x = (S['dec'] - t0) / 60; brk = np.r_[False, np.diff(S['start']) > 45]
    xs, ys = [], []
    for i in range(len(x)):
        if brk[i]: xs.append(np.nan); ys.append(np.nan)
        xs.append(x[i]); ys.append(sc[i])
    ax.plot(xs, ys, color=C1, lw=1.3, marker='o', ms=2.2, label='Smoothed stress probability')
    ax.axhline(tau, color=INK, lw=0.9, ls='--', label='Threshold (0.80 window sensitivity)')
    for k, e in enumerate(ev):
        ax.plot([(e[2] - t0) / 60, (e[3] - t0) / 60], [0.93, 0.93], color=C4, lw=3.6, solid_capstyle='butt', label='Alarm event' if k == 0 else None)
    ax.set_ylim(0, 1); ax.set_xlim(0, (S['dec'].max() - t0) / 60 + 1.5); ax.set_xlabel('Time since start of baseline (min)'); ax.set_ylabel('Stress probability')
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=FS_T)
    ax.legend(frameon=False, fontsize=FS_L, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3, handlelength=2.2, columnspacing=2.0)
    # B: operating curves as small multiples, one facet per persistence N; the other N in grey for context
    cv = pd.DataFrame(m5['curve_descriptive']); Ns = sorted(cv.N.unique()); ref = cv[(cv.q == 0.8) & (cv.N == 1)].iloc[0]; s = m5['selected']
    TIT = {1: 'N = 1 (single window)', 8: 'N = 8 (about 2 min)'}
    gB = gs[1].subgridspec(2, 3, hspace=0.42, wspace=0.13); axesB = []
    for k, N in enumerate(Ns):
        ax = fig.add_subplot(gB[k // 3, k % 3], sharex=axesB[0] if axesB else None, sharey=axesB[0] if axesB else None); axesB.append(ax)
        for N2, g in cv.groupby('N'):
            if N2 != N: g = g.sort_values('q'); ax.plot(g.fa_comb, g.sens, '-', color='#c9c8c3', lw=0.8, zorder=1)
        g = cv[cv.N == N].sort_values('q')
        ax.plot(g.fa_comb, g.sens, '-o', color=C1, lw=1.6, ms=3.6, mec='white', mew=0.6, zorder=3, label='Rules with this persistence N' if k == 0 else None)
        ax.axhline(0.90, color=MUTED, lw=0.7, ls=':', zorder=0, label='Detection floor (0.90)' if k == 0 else None)
        if N == 1:
            ax.plot(ref.fa_comb, ref.sens, marker='s', ms=6, color=INK, mec='white', mew=0.8, ls='none', zorder=5, label='Reference rule')
            ax.plot(s['fa_comb'][0], s['sens'][0], marker='D', ms=6, color=C2, mec='white', mew=0.8, ls='none', zorder=6, label='Rules chosen in training folds')
            for _, r_ in g.iterrows():
                OFF = {0.5: ((-3, 4), 'right', 'bottom'), 0.6: ((-3, 4), 'right', 'bottom'), 0.7: ((5, -3), 'left', 'top'), 0.9: ((-6, 0), 'right', 'center')}
                if r_.q in OFF: ax.annotate(f'{r_.q:.1f}', (r_.fa_comb, r_.sens), xytext=OFF[r_.q][0], textcoords='offset points', ha=OFF[r_.q][1], va=OFF[r_.q][2], fontsize=6.8, color=MUTED)
        ax.set_title(TIT.get(N, f'N = {N}'), fontsize=8, color=INK, pad=3)
        ax.set_xlim(0, 2.15); ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=FS_T)
        if k % 3: plt.setp(ax.get_yticklabels(), visible=False)
        if k < 3: plt.setp(ax.get_xticklabels(), visible=False)
    axesB[0].set_ylabel('Episodes detected'); axesB[3].set_ylabel('Episodes detected')
    axesB[4].set_xlabel('False alarms per hour (mean of rest and exercise)')
    h, l = [], []
    for a_ in axesB[:1]:
        hh, ll = a_.get_legend_handles_labels(); h += hh; l += ll
    from matplotlib.lines import Line2D
    h.insert(1, Line2D([], [], color='#c9c8c3', lw=0.8)); l.insert(1, 'Other persistence values')
    axesB[4].legend(h, l, frameon=False, fontsize=FS_L, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=3, handlelength=2.2, columnspacing=1.6)
    # C: scoring variants, full width; groups on the x axis
    ax = fig.add_subplot(gs[2]); axC = ax
    items = [('Episodes detected\n(any overlap)', '', 'sens'), ('Episodes detected\n(alarm begins in task)', '_strict', 'sens'), ('Rest time\nin alarm', '', 'alarm_rest')]
    nul = json.load(open('improve/review/sum_review.json'))['S1_episode_null']
    for j, (nm, col, mk, lab) in enumerate((('reference', INK, 's', 'Reference rule'), ('selected', C2, 'D', 'Rules chosen in training folds'))):
        for i, (_, suf, k) in enumerate(items):
            m, lo, hi = m5[nm + suf][k]; xx = i + (j - 0.5) * 0.36
            if k == 'sens':
                nl = nul[f"{nm}_{'strict' if suf else 'overlap'}"]
                ax.bar(xx, nl['null_p97_5'] - nl['null_p2_5'], bottom=nl['null_p2_5'], width=0.2, color='#c9c8c3', lw=0, zorder=0,
                       label='Expected by chance (2.5th–97.5th percentile)' if (i == 0 and j == 0) else None)
            ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=6, mec='white', mew=0.8, capsize=3, lw=1.3, zorder=4, label=lab if i == 0 else None)
            ax.text(xx + 0.05, m, f'{m:.2f}', va='center', ha='left', fontsize=7.4, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([i[0] for i in items], fontsize=8); ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis='y', labelsize=FS_T); ax.tick_params(axis='x', length=0)
    hh, ll = ax.get_legend_handles_labels(); order = [ll.index('Reference rule'), ll.index('Rules chosen in training folds'), ll.index('Expected by chance (2.5th–97.5th percentile)')]
    ax.legend([hh[o] for o in order], [ll[o] for o in order], frameon=False, fontsize=FS_L, loc='upper right', bbox_to_anchor=(1.0, 1.04), ncol=1, handlelength=1.8)
    # panel letters at a common left position
    xL = fig.transFigure.inverted().transform(axA.transAxes.transform((0, 0)))[0] - 0.085
    for a_, lt in ((axA, 'A'), (axesB[0], 'B'), (axC, 'C')):
        yT = fig.transFigure.inverted().transform(a_.transAxes.transform((0, 1)))[1]
        fig.text(xL, yT + 0.012, lt, fontsize=11, fontweight='bold', va='bottom', ha='left')
    fig.savefig('fig/fig8_episodes.png', bbox_inches='tight', pad_inches=0.05); plt.close(fig)
    print('fig8 example participant', pid)

import sys
for f in sys.argv[1:]: globals()[f]()
print('ok')
