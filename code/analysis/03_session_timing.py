"""Session timing structure: same-day protocol pairs within a participant (order/carry-over) and
concurrent group sessions across participants (shared room). Uses de-identified UTC starts; the
dataset applies one consistent shift, so relative timing is preserved (verified: group members' tags coincide)."""
import os,sys,pandas as pd,numpy as np
ROOT,OUT=sys.argv[1],sys.argv[2]
r=[]
for p in ['STRESS','AEROBIC','ANAEROBIC']:
    for s in sorted(os.listdir(os.path.join(ROOT,'Wearable_Dataset',p))):
        f=os.path.join(ROOT,'Wearable_Dataset',p,s,'EDA.csv')
        if not os.path.exists(f): continue
        h=pd.read_csv(f,header=None,nrows=2); n=sum(1 for _ in open(f))-2
        t0=pd.to_datetime(h.iloc[0,0]); r.append(dict(protocol=p,session=s,participant=s.split('_')[0],start=t0,end=t0+pd.Timedelta(seconds=n/float(h.iloc[1,0]))))
d=pd.DataFrame(r)
# same-day pairs within participant
pairs=[]
for pid,g in d.groupby('participant'):
    g=g.sort_values('start')
    for i in range(len(g)):
        for j in range(i+1,len(g)):
            a,b=g.iloc[i],g.iloc[j]
            if a.protocol==b.protocol: continue
            gap=(b.start-a.end).total_seconds()/60
            if b.start.date()==a.start.date(): pairs.append(dict(participant=pid,first=a.protocol,second=b.protocol,gap_min=round(gap,1)))
P=pd.DataFrame(pairs); print('SAME-DAY PROTOCOL PAIRS (gap = start of 2nd minus end of 1st)'); print(P.to_string(index=False))
# concurrent sessions (overlap > 50% of shorter) across different participants
grp=[]
for i in range(len(d)):
    for j in range(i+1,len(d)):
        a,b=d.iloc[i],d.iloc[j]
        if a.participant==b.participant: continue
        ov=(min(a.end,b.end)-max(a.start,b.start)).total_seconds()
        short=min((a.end-a.start).total_seconds(),(b.end-b.start).total_seconds())
        if ov>0.5*short: grp.append((a.protocol+'/'+a.session,b.protocol+'/'+b.session))
# connected components
import collections
adj=collections.defaultdict(set)
for a,b in grp: adj[a].add(b); adj[b].add(a)
seen=set();comps=[]
for k in adj:
    if k in seen: continue
    st=[k];c=set()
    while st:
        x=st.pop()
        if x in c: continue
        c.add(x); st+=list(adj[x])
    seen|=c; comps.append(sorted(c))
print('\nCONCURRENT RECORDING GROUPS (>50% overlap)')
rows=[]
for i,c in enumerate(comps,1):
    print(f'G{i}:',', '.join(c))
    for x in c: rows.append(dict(group=f'G{i}',protocol=x.split('/')[0],session=x.split('/')[1]))
d.to_csv(os.path.join(OUT,'session_times.csv'),index=False); P.to_csv(os.path.join(OUT,'same_day_pairs.csv'),index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT,'concurrent_groups.csv'),index=False)
