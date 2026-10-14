"""Figures for the analysis plan. Fig 1: analysis overview (diagram). Fig 2: protocol timelines from tag-derived
median block durations (segments.csv; no physiological signal used)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
plt.rcParams.update({'font.family':'TeX Gyre Pagella','font.size':8,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b',
                     'xtick.color':'#52514e','ytick.color':'#52514e','savefig.dpi':300})
INK='#0b0b0b'; MUTED='#52514e'
C={'rest':'#b9b8b2','stress':'#eb6834','stage':'#2a78d6','sprint':'#4a3aa7','warm':'#1baf7a'}
# ---------- Figure 2: timelines
S=pd.read_csv('data/segments.csv'); S=S[~S.flag.fillna('').str.contains('UNUSABLE|non-positive')]
ORDER={('STRESS','V1'):['baseline','stroop','rest1','tmct','rest2','real_opinion','opposite_opinion','subtract'],
 ('STRESS','V2'):['baseline','tmct','rest1','real_opinion','opposite_opinion','rest2','subtract'],
 ('AEROBIC','V1'):['warmup']+[f'stage_{r}rpm' for r in [60,70,75,80,85,90,95,100,105,110]]+['cooldown','rest'],
 ('AEROBIC','V2'):['baseline','warmup']+[f'stage_{r}rpm' for r in [70,75,80,85]]+['stage_90_95rpm','cooldown','rest'],
 ('ANAEROBIC','V1'):['warmup','sprint1','cooldown1','sprint2','cooldown2','sprint3','cooldown3','rest'],
 ('ANAEROBIC','V2'):['baseline','warmup']+sum([[f'sprint{i}',f'cooldown{i}'] for i in range(1,5)],[])+['rest']}
LAB={'baseline':'Base','stroop':'Stroop','rest1':'Rest 1','tmct':'TMCT','rest2':'Rest 2','real_opinion':'RO','opposite_opinion':'OO',
     'subtract':'Sub','warmup':'Warm-up','cooldown':'Cool-down','rest':'Rest'}
def cat(b):
    if b in('baseline','rest','rest1','rest2'): return 'rest'
    if b in('stroop','tmct','real_opinion','opposite_opinion','subtract'): return 'stress'
    if b.startswith('stage'): return 'stage'
    if b.startswith('sprint'): return 'sprint'
    return 'warm'
fig,ax=plt.subplots(figsize=(6.5,2.75)); y=0; yt=[]; yl=[]
for (p,v),blocks in ORDER.items():
    sub=S[(S.protocol==p)&(S.version==v)]; npart=sub.participant.nunique(); x=0
    for b in blocks:
        d=sub[sub.block==b].duration_s
        if d.empty: continue
        w=d.median()/60
        ax.barh(y,w,left=x,height=0.62,color=C[cat(b)],edgecolor='white',linewidth=1.2)
        lab=LAB.get(b, b.replace('stage_','').replace('rpm','').replace('90_95','90/95').replace('sprint','S').replace('cooldown','C'))
        SHORT={'Warm-up':'W','Cool-down':'CD','Base':'B','Rest':'R','Stroop':'St','TMCT':'T','Rest 1':'R1','Rest 2':'R2'}
        fits=lambda t:len(t)*0.42+0.25<=w
        if not fits(lab): lab=SHORT.get(lab,lab)
        if fits(lab): ax.text(x+w/2,y,lab,ha='center',va='center',fontsize=6,color='white' if cat(b)!='rest' else INK)
        x+=w
    ax.text(x+0.4,y,f'{x:.0f} min',va='center',fontsize=6.5,color=MUTED)
    yt.append(y); yl.append(f'{p.title()} {v} (n = {npart})'); y-=1
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlabel('Elapsed time from first block (min; medians of tag-derived block durations, inter-block gaps omitted)')
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.tick_params(axis='y',length=0); ax.set_xlim(0,41)
ax.legend(handles=[Patch(color=C[k],label=l) for k,l in [('rest','Baseline / rest'),('stress','Stress task'),('warm','Warm-up / cool-down'),('stage','Aerobic stage'),('sprint','Sprint')]],
          ncol=5,loc='upper center',bbox_to_anchor=(0.45,1.13),frameon=False,fontsize=7,handlelength=1.2)
fig.tight_layout(); fig.savefig('fig/fig2_protocols.png'); plt.close(fig)

# ---------- Figure 1: analysis overview
fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.set_xlim(-1,101); ax.set_ylim(-2.5,60); ax.axis('off')
def box(x,y,w,h,title,body,fc='#f3f2ee',ec=MUTED,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.2',fc=fc,ec=ec,lw=0.8))
    ax.text(x+1.2,y+h-1.6,title,fontsize=7.5,fontweight='bold',va='top',color=tc)
    ax.text(x+1.2,y+h-5.3,body,fontsize=6.4,va='top',color=tc,linespacing=1.25)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=MUTED,lw=0.8))
box(0,42,22,17,'A  Data','36 participants, Empatica E4\nSTRESS (36), AEROBIC (30),\nANAEROBIC (31) sessions\nV1: S01–S18; V2: f01–f18')
box(26,42,22,17,'B  Quality control','SHA-256 check (708 files)\nDocumented exclusions\nSignal-range and coverage rules\nTag-derived segment index')
box(52,42,22,17,'C  Windows and features','30 s windows within blocks\nEDA, BVP pulse, TEMP, ACC\nFixed feature list\n(Appendix A)')
box(78,42,22,17,'D  Models','L2 logistic regression\nXGBoost (small fixed grid)\nTuning on inner grouped folds\nonly')
for x in (22,48,74): arrow(x+0.3,50.5,x+3.7,50.5)
exps=[('E1','Original\nprotocol','sample-level 80/20\n+ 10-fold CV,\noversampling'),('E2','Participant-\nlevel','nested leave-one-\nparticipant-out'),
      ('E3','Shortcut\nbaselines','time-only,\nblock-position,\nACC-only'),('E4','Pre-task\nbaselines','V2 sessions before\nany task (session\ncontext)'),
      ('E5','Cross-\ncohort','train V1 → test V2\nand V2 → V1'),('E6','Deployment\ntest','stress detector\napplied to exercise\nsessions')]
for i,(k,t,b) in enumerate(exps):
    x=i*16.8; fc='#fdeee7' if k in('E1','E2','E6') else '#f3f2ee'
    ax.add_patch(FancyBboxPatch((x,15),15.6,19,boxstyle='round,pad=0.25,rounding_size=1.2',fc=fc,ec=MUTED,lw=0.8))
    ax.text(x+1.0,32.4,f'{k}  {t}',fontsize=7.2,fontweight='bold',va='top',color=INK,linespacing=1.15)
    ax.text(x+1.0,24.2,b,fontsize=6.2,va='top',color=INK,linespacing=1.25)
ax.plot([89,89],[41.6,37.6],color=MUTED,lw=0.8); ax.plot([7.8,91.8],[37.6,37.6],color=MUTED,lw=0.8)
for i in range(6): arrow(i*16.8+7.8,37.6,i*16.8+7.8,34.6)
box(0,-1,100,12.5,'F  Outputs and inference','Balanced accuracy, macro-F1, AUROC; false-alarm rate per participant and exercise phase; 95% CIs by participant-level bootstrap;\npaired participant-level comparisons; Holm–Bonferroni correction across the confirmatory family (H1–H8).\nExploratory (not registered): personal calibration, causal context features and smoothing, chosen by nested LOPO.')
ax.text(50,36.9,'',fontsize=1)
fig.tight_layout(pad=0.2); fig.savefig('fig/fig1_overview.png'); plt.close(fig)
print('ok')
