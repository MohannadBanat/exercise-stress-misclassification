"""Map tags.csv button presses to labelled protocol blocks for every session.
Mapping derived from the protocol diagrams embedded in Wearable_Dataset.ipynb and the notebook's own
STRESS shading code (V1: Stroop tags[3]-[4], TMCT [5]-[6], ...; V2: TMCT [2]-[3], ...), where tags[k]
is the k-th row of tags.csv (1-based). Times are seconds from the session start (first header row).
Output: segments.csv (one row per block) and tag_qc.csv (one row per session)."""
import os, sys, pandas as pd, numpy as np
ROOT, OUTDIR = sys.argv[1], sys.argv[2]
NOMINAL = {  # nominal block durations (s) from the paper / diagrams
 'baseline':180,'stroop':120,'rest1':300,'tmct':180,'rest2':300,'real_opinion':30,'opposite_opinion':30,'subtract':30,
 'warmup':180,'cooldown':240,'rest':120,'sprint':30}
NOMINAL_V2={('STRESS','rest1'):600,('STRESS','rest2'):600,('ANAEROBIC','sprint'):45,('ANAEROBIC','warmup'):270,
 ('ANAEROBIC','cooldown'):225,('AEROBIC','warmup'):135,('AEROBIC','cooldown'):180}
# sessions whose tag timing makes specific blocks unusable (decided from tag QC, before any signal analysis)
UNRESOLVED={('AEROBIC','f05'):'stage_90_95rpm;cooldown',('AEROBIC','f06'):'stage_90_95rpm;cooldown',
 ('AEROBIC','f07'):'stage_90_95rpm;cooldown',('AEROBIC','f02'):'baseline;warmup',('ANAEROBIC','f04'):'baseline;warmup'}
def rows(tag_path):
    if os.path.getsize(tag_path)==0: return []
    return list(pd.to_datetime(pd.read_csv(tag_path,header=None)[0]))
def seg(out,blk,a,b,**kw): out.append(dict(block=blk,start_s=a,end_s=b,**kw))

def map_session(prot,ver,sess,t,dur):
    r=lambda i:t[i-1]; n=len(t); out=[]; note=''
    if prot=='STRESS' and ver=='V1' and n==13:
        seg(out,'baseline',r(1),r(2)); seg(out,'stroop',r(3),r(4)); seg(out,'rest1',r(4),r(5)); seg(out,'tmct',r(5),r(6))
        seg(out,'rest2',r(6),r(7)); seg(out,'real_opinion',r(7),r(8)); seg(out,'opposite_opinion',r(9),r(10)); seg(out,'subtract',r(11),r(12))
    elif prot=='STRESS' and ver=='V2' and n==9:
        seg(out,'baseline',max(0,r(1)-180),r(1)); seg(out,'tmct',r(2),r(3)); seg(out,'rest1',r(3),r(4)); seg(out,'real_opinion',r(4),r(5))
        seg(out,'opposite_opinion',r(6),r(7)); seg(out,'rest2',r(7),r(8)); seg(out,'subtract',r(8),r(9))
        note='baseline = last 180 s before tag 1 (baseline start not tagged)'
    elif prot=='AEROBIC' and ver=='V1' and n>=7:
        stages=[60,70,75,80,85,90,95,100,105,110]
        seg(out,'warmup',0,r(1))
        nst={12:10,9:7,7:5,10:9}[n] if n in (12,9,7,10) else None
        for k in range(nst): seg(out,f'stage_{stages[k]}rpm',r(k+1),r(k+2))
        if n==10: note='S11_a: stages to 105 rpm; continues in S11_b'
        else:
            seg(out,'cooldown',r(nst+1),r(nst+2)); seg(out,'rest',r(nst+2),dur)
            if n!=12: note=f'incomplete protocol: last tagged stage {stages[nst-1]} rpm'
    elif prot=='AEROBIC' and sess=='S11_b':
        seg(out,'stage_110rpm',0,r(1)); seg(out,'cooldown',r(1),dur); note='second file of S11 after connection loss'
    elif prot=='AEROBIC' and ver=='V2' and n==8:
        seg(out,'baseline',max(0,r(1)-135-180),r(1)-135); seg(out,'warmup',r(1)-135,r(1))
        for k,s in enumerate([70,75,80,85]): seg(out,f'stage_{s}rpm',r(k+1),r(k+2))
        seg(out,'stage_90_95rpm',r(5),r(6)); seg(out,'cooldown',r(6),r(7)); seg(out,'rest',r(7),r(8))
        note='baseline/warm-up boundary not tagged: warm-up = 135 s before tag 1 (verify with ACC)'
    elif prot=='ANAEROBIC' and ver=='V1' and sess not in ('S16_a','S16_b'):
        seg(out,'warmup',0,r(1)); k=1; sp=1
        while k+1<=n:
            seg(out,f'sprint{sp}',r(k),r(k+1))
            if k+2<=n: seg(out,f'cooldown{sp}',r(k+1),r(k+2))
            k+=2; sp+=1
        if n==7: seg(out,'rest',r(7),dur)
        else: note='incomplete protocol: last sprint missing'
    elif sess=='S16_a':
        seg(out,'warmup',0,r(1)); seg(out,'sprint1',r(1),r(2)); seg(out,'cooldown1',r(2),r(3)); seg(out,'sprint2',r(3),r(4)); seg(out,'cooldown2',r(4),dur)
        note='connection loss; continues in S16_b'
    elif sess=='S16_b':
        seg(out,'sprint3',r(1),r(2)); seg(out,'cooldown3',r(2),r(3)); seg(out,'sprint4',r(3),r(4)); seg(out,'cooldown4',r(4),r(5)); seg(out,'rest',r(5),dur)
        note='4 sprints in a V1 session (protocol had 3)'
    elif prot=='ANAEROBIC' and ver=='V2' and n==11:
        seg(out,'baseline',max(0,r(1)-180),r(1)); seg(out,'warmup',r(1),r(2))
        note=f'baseline = last 180 s of {r(1):.0f} s pre-warm-up period'
        for sp in range(1,5):
            seg(out,f'sprint{sp}',r(2*sp),r(2*sp+1)); seg(out,f'cooldown{sp}',r(2*sp+1),r(2*sp+2))
        seg(out,'rest',r(10),r(11))
    else:
        note=f'UNMAPPED ({n} tags)'
    return out,note

segs=[];qc=[]
for prot in ['STRESS','AEROBIC','ANAEROBIC']:
    pdir=os.path.join(ROOT,'Wearable_Dataset',prot)
    for sess in sorted(os.listdir(pdir)):
        d=os.path.join(pdir,sess)
        if not os.path.isdir(d): continue
        ver='V1' if sess.startswith('S') else 'V2'
        hdr=pd.read_csv(os.path.join(d,'EDA.csv'),header=None,nrows=2)
        t0=pd.to_datetime(hdr.iloc[0,0]); fs=float(hdr.iloc[1,0])
        dur=(sum(1 for _ in open(os.path.join(d,'EDA.csv')))-2)/fs
        tags=[(x-t0).total_seconds() for x in rows(os.path.join(d,'tags.csv'))]
        nonmono=int(np.sum(np.diff(tags)<=0)) if len(tags)>1 else 0
        out,note=map_session(prot,ver,sess,tags,dur) if tags else ([],'no tags')
        for s in out:
            s.update(protocol=prot,session=sess,participant=sess.split('_')[0],version=ver,duration_s=round(s['end_s']-s['start_s'],1))
            base=s['block'].rstrip('0123456789') if s['block'].startswith(('sprint','cooldown')) else s['block']
            nom=NOMINAL_V2.get((prot,base),NOMINAL.get(base)) if ver=='V2' else NOMINAL.get(base); s['nominal_s']=nom
            fl=[]
            if s['duration_s']<=0: fl.append('non-positive duration')
            elif nom and not s['block'].startswith('stage') and abs(s['duration_s']-nom)/nom>0.5: fl.append('duration deviates >50% from nominal')
            if s['start_s']<0 or s['end_s']>dur+1: fl.append('outside recording')
            if s['block'] in UNRESOLVED.get((prot,sess),'').split(';'): fl.append('UNUSABLE: tag timing unresolved')
            s['flag']='; '.join(fl)
        segs+=out
        qc.append(dict(protocol=prot,session=sess,version=ver,n_tags=len(tags),recording_s=round(dur,1),
                       pre_first_tag_s=round(tags[0],1) if tags else None,non_monotonic_tags=nonmono,note=note))
S=pd.DataFrame(segs)[['protocol','version','participant','session','block','start_s','end_s','duration_s','nominal_s','flag']]
S.to_csv(os.path.join(OUTDIR,'segments.csv'),index=False); Q=pd.DataFrame(qc); Q.to_csv(os.path.join(OUTDIR,'tag_qc.csv'),index=False)
pd.set_option('display.width',220); pd.set_option('display.max_rows',500)
print(Q[(Q.note!='')|(Q.non_monotonic_tags>0)].to_string(index=False))
print('\nFLAGGED SEGMENTS'); print(S[S.flag!=''].to_string(index=False))
print('\nMEDIAN DURATION (s) per block'); print(S.groupby(['protocol','version','block']).duration_s.agg(['count','median','min','max']).round(0).to_string())
