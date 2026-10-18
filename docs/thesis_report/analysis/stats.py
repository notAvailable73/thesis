import json,os,statistics as st,itertools,math
D=os.path.dirname(os.path.abspath(__file__))
rows=json.load(open(D+'/rows.json'))
REPO=str(__import__('pathlib').Path(__file__).resolve().parents[3])
mv=json.load(open(REPO+'/results/mvt_results.json'))['results']
# AUROC/FPR cross-check against grid
bad=0
for r in rows:
    c=mv[r['dataset']][f"{r['shot']}shot"][r['backbone']][r['adapter']][r['head']]; sc='vacuity' if r['head']=='evidential' else 'msp'
    for p in ['svhn_far','tin_near','gaussian_far']:
        k=f'ood_auroc__{p}__{sc}'
        if k in c and str(r['seed']) in c[k]['per_seed'] and abs(c[k]['per_seed'][str(r['seed'])]-r['auroc_'+p])>1e-9: bad+=1
print('auroc mismatches (expect only smoke seed):',bad)
from collections import defaultdict
cells=defaultdict(list)
for r in rows: cells[(r['dataset'],r['shot'],r['backbone'],r['adapter'],r['head'])].append(r)
def mean(cell,k): return st.mean(x[k] for x in cell)
agg={}
for k,v in cells.items():
    a={f:mean(v,f) for f in v[0] if isinstance(v[0][f],(int,float)) and f not in('seed','shot') and all(x[f] is not None for x in v)}
    a['acc_sd']=st.pstdev([x['acc'] for x in v]); a['ece_sd']=st.pstdev([x['ece'] for x in v])
    a['near_pool']=v[0]['near_pool']; agg[k]=a
json.dump({ '|'.join(map(str,k)):v for k,v in agg.items()},open(D+'/agg.json','w'),indent=1)
# baseline variance
for k,v in agg.items():
    if k[3] in('full_ft','linear_probe'): print('baseline',k,'acc_sd',round(v['acc_sd']*100,4),'ece_sd',round(v['ece_sd'],5))
# 16 pairs
core=[k for k in agg if k[3] in('bottleneck_parallel','lora')]
pairs=[(k, (k[0],k[1],k[2],'lora',k[4])) for k in core if k[3]=='bottleneck_parallel']
w=dict(acc=0,ece=0,near=0,tin=0,svhn=0)
for b,l in pairs:
    B,L=agg[b],agg[l]
    w['acc']+=B['acc']>L['acc']; w['ece']+=B['ece']<L['ece']; w['near']+=B['auroc_near']>L['auroc_near']; w['tin']+=B['auroc_tin_near']>L['auroc_tin_near']; w['svhn']+=B['auroc_svhn_far']>L['auroc_svhn_far']
    print('pair',b[:3],b[4],'dAcc %+.2f'%(100*(B['acc']-L['acc'])),'ECE b %.4f l %.4f'%(B['ece'],L['ece']))
print('bottleneck wins',w)
# 20 matched evid vs soft
ratios=[];ts=[];accev=0
for k in agg:
    if k[4]=='evidential':
        s=agg[(k[0],k[1],k[2],k[3],'softmax')]; e=agg[k]
        ratios.append(e['ece']/s['ece']); ts.append(e['ece']/s['ece_ts']); accev+= e['acc']>s['acc']
print('evid worse ECE',sum(r>1 for r in ratios),'/',len(ratios),'range %.2f-%.2f'%(min(ratios),max(ratios)),'vsTS %.1f-%.1f'%(min(ts),max(ts)),'evid more accurate',accev)
# vacuity vs softmax scores: native vacuity of evid cell vs softmax sibling's msp/ts/energy, per OOD set
res=defaultdict(lambda: defaultdict(list))
for k in agg:
    if k[4]!='evidential': continue
    e=agg[k]; s=agg[(k[0],k[1],k[2],k[3],'softmax')]
    for p,grp in [('svhn_far','far'),('gaussian_far','far'),('near','near'),('tin_near','near')]:
        for sc in ['msp','ts_msp','energy']:
            res[sc][grp].append(e['auroc_'+p]-s[f'auroc_{p}_{sc}'])
for sc in res:
    for g in res[sc]:
        d=res[sc][g]; print('vacuity vs',sc,g,'wins %d/%d'%(sum(x>0 for x in d),len(d)),'meanD %+.4f'%st.mean(d),'range %+.3f..%+.3f'%(min(d),max(d)))
# spearman ECE vs AUROC across 40 cells
def rank(a):
    s=sorted(range(len(a)),key=lambda i:a[i]); r=[0]*len(a)
    i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[s[j+1]]==a[s[i]]: j+=1
        for t in range(i,j+1): r[s[t]]=(i+j)/2
        i=j+1
    return r
def spear(x,y):
    rx,ry=rank(x),rank(y); mx,my=st.mean(rx),st.mean(ry)
    return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
for sub,name in [(list(agg),'all40'),([k for k in agg if k[4]=='evidential'],'evid'),([k for k in agg if k[4]=='softmax'],'soft')]:
    print(name,'rho svhn %.3f tin %.3f'%(spear([agg[k]['ece'] for k in sub],[agg[k]['auroc_svhn_far'] for k in sub]),spear([agg[k]['ece'] for k in sub],[agg[k]['auroc_tin_near'] for k in sub])))
