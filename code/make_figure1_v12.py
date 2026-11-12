"""Figure 1 of the second revision: overview of the study, with identifiers for the exploratory analyses (EA1-EA5) and a tier
for the post hoc controls (PC1-PC7). Same drawing code and style as make_revision_figures.py (v1.1.0). 600 dpi.
Usage: python make_figure1_v12.py [output.png]"""
import sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'TeX Gyre Pagella', 'font.size': 8, 'savefig.dpi': 600})
INK, MUTED = '#0b0b0b', '#52514e'
out = sys.argv[1] if len(sys.argv) > 1 else 'fig1_overview_v12.png'

fig, ax = plt.subplots(figsize=(6.5, 5.15)); ax.set_xlim(-1, 101); ax.set_ylim(4.5, 107); ax.axis('off')
def box(x, y, w, h, title, body, fc='#f3f2ee', tsz=7.3, bsz=6.2, dy=4.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + 1.0, y + h - 1.3, title, fontsize=tsz, fontweight='bold', va='top', color=INK, linespacing=1.12)
    ax.text(x + 1.0, y + h - dy, body, fontsize=bsz, va='top', color=INK, linespacing=1.22)
def arrow(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8))
def tier(y, s): ax.text(0, y, s, fontsize=7.4, fontstyle='italic', color=MUTED, va='bottom')

# tier 1: data and pipeline
tier(104.3, 'Data and pipeline, fixed in the registered plan (Sections 3.1–3.4)')
box(0, 91, 22, 12.5, 'A  Data', '36 participants, Empatica E4\nstress (36), aerobic (30),\nanaerobic (31) sessions')
box(26, 91, 22, 12.5, 'B  Quality control', 'Session exclusions,\nsignal-validity rules,\nprotocol-block index')
box(52, 91, 22, 12.5, 'C  Windows and features', '30 s windows, 15 s step\n29 features: EDA, pulse,\nHR, TEMP, ACC')
box(78, 91, 22, 12.5, 'D  Models', 'XGBoost; logistic regression\ntuned on training\nparticipants only')
for x in (22, 48, 74): arrow(x + 0.3, 97.2, x + 3.7, 97.2)

# tier 2: registered experiments
tier(85.3, 'Registered experiments and hypotheses H1–H8 (Section 3.5)')
exps = [('E1', 'Published\nevaluation', 're-implemented\nwith our features:\n80/20 split, 10-fold\nCV, oversampling'),
        ('E2', 'Participant-\nlevel', 'held-out\nparticipants\n(H1a–H1c)'),
        ('E3', 'Shortcut\nbaselines', 'time, block\nposition, motion\n(H2, H3)'), ('E4', 'Pre-task\nbaselines', 'sessions told\napart before any\ntask? (H4)'),
        ('E5', 'Cross-\ncohort', 'train on one\ncohort, test on\nthe other (H5)'), ('E6', 'Deployment\ntest', 'stress detector\non exercise\n(H6–H8)')]
for i, (k, t, b) in enumerate(exps):
    box(i * 16.8, 66, 15.6, 18, f'{k}  {t}', b, fc='#fdeee7' if k in ('E1', 'E2', 'E6') else '#f3f2ee', tsz=7.0, bsz=6.1, dy=7.9)

# tier 3: exploratory analyses EA1-EA5
tier(60.3, 'Exploratory analyses EA1–EA5, selection rules written down before each analysis was run (Section 3.6)')
xs = [('EA1  Deployable\nimprovements', 'calibration, context\nfeatures, smoothing;\nexercise-robust detector'),
      ('EA2  Development,\nsealed test', '24 development →\n12 sealed participants'),
      ('EA3  Episode-level\nevaluation', 'episodes, false alarms\nper hour, chance'),
      ('EA4  Rest false\nalarms', '5- or 10-min\nadaptive reference'),
      ('EA5  External test', 'WESAD, 15 new\nparticipants')]
for i, (t, b) in enumerate(xs): box(i * 20.2, 43.5, 18.6, 15.7, t, b, fc='#eaf1fb', tsz=7.0, bsz=6.1, dy=7.9)
for i in range(4): arrow(i * 20.2 + 19.0, 51.3, i * 20.2 + 20.0, 51.3)

# tier 4: post hoc controls PC1-PC7
tier(37.6, 'Post hoc controls PC1–PC7, added after the results were known (Section 3.7, Appendix B)')
pcs = [('PC1', 'Chance\nbaseline', 'episodes\ndetected by\nchance'),
       ('PC2', 'Time-only\ncomparator', 'elapsed time\nand cohort'),
       ('PC3', 'Single\nmodality', 'motion or\nphysiology\nalone'),
       ('PC4', 'E6 detector\nanalyses', 'phase, time,\nmotion gate'),
       ('PC5', 'Motion and\ntime', 'motion, time,\ncohort only'),
       ('PC6', 'Sealed\nepisodes', 'episode rules,\n12 sealed\nparticipants'),
       ('PC7', 'Ablations,\nresponse size', 'features\nremoved;\nresponse size')]
w = 13.2
for i, (k, t, b) in enumerate(pcs):
    box(i * (w + 1.27), 20.5, w, 16.4, f'{k}\n{t}', b, fc='#f6f0fb', tsz=6.7, bsz=5.9, dy=9.3)

# F: outputs
box(0, 5.5, 100, 12.2, 'F  Outputs and inference',
    'Balanced accuracy, AUROC, false-alarm rates per participant and exercise phase, episodes detected and false alarms per hour.\n'
    '95% CIs from participant-level bootstrap; Holm–Bonferroni correction across the ten registered tests only.\n'
    'Exploratory and post hoc results are labelled as such wherever they are reported.', dy=4.9)
fig.tight_layout(pad=0.2); fig.savefig(out); plt.close(fig)
print('saved', out)
