"""Inventory of the Hongn et al. (2025) Empatica E4 dataset: per-session signal durations, tags, IBI coverage, anomalies."""
import os, sys, pandas as pd, numpy as np
ROOT = sys.argv[1]; OUT = sys.argv[2]
rows=[]
for prot in ['STRESS','AEROBIC','ANAEROBIC']:
    pdir=os.path.join(ROOT,'Wearable_Dataset',prot)
    for sess in sorted(os.listdir(pdir)):
        d=os.path.join(pdir,sess)
        if not os.path.isdir(d): continue
        r={'protocol':prot,'session':sess,'participant':sess.split('_')[0]}
        starts={}
        for sig in ['BVP','EDA','TEMP','ACC','HR']:
            f=os.path.join(d,sig+'.csv')
            if not os.path.exists(f) or os.path.getsize(f)==0: r[sig+'_min']=np.nan; continue
            hdr=pd.read_csv(f,header=None,nrows=2)
            fs=float(hdr.iloc[1,0]); starts[sig]=str(hdr.iloc[0,0])
            x=pd.read_csv(f,header=None,skiprows=2).values
            r[sig+'_min']=round(len(x)/fs/60,2)
            if sig=='EDA':
                r['EDA_frac_lt0.05uS']=round(float((x[:,0]<0.05).mean()),3)
            if sig=='TEMP':
                r['TEMP_median']=round(float(np.median(x[:,0])),2)
        r['starts_consistent']=len(set(starts.values()))<=1
        f=os.path.join(d,'IBI.csv')
        if os.path.getsize(f)>0:
            ibi=pd.read_csv(f,header=None,skiprows=1).apply(pd.to_numeric,errors="coerce").dropna().values
            r['IBI_n']=len(ibi); r['IBI_coverage_frac']=round(float(ibi[:,1].sum()/(r['BVP_min']*60)),3) if r.get('BVP_min') else np.nan
        else: r['IBI_n']=0; r['IBI_coverage_frac']=0.0
        f=os.path.join(d,'tags.csv')
        r['n_tags']=0 if os.path.getsize(f)==0 else len(pd.read_csv(f,header=None))
        rows.append(r)
df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df.to_string(index=False))
print(df.groupby('protocol')[['BVP_min','EDA_min','ACC_min']].agg(['count','sum','median']).round(1))
print(df.groupby('protocol').n_tags.value_counts().sort_index())
