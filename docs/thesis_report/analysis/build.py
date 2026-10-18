"""Build a corrected per-seed table from results/rq_factorial (120/120, retrained
smoke seed), cross-checked against results/mvt_results.json."""
import json, glob, os, statistics as st
REPO=str(__import__('pathlib').Path(__file__).resolve().parents[3])
rows=[]
for f in sorted(glob.glob(REPO+'/results/rq_factorial/*_seed*.json')):
    r=json.load(open(f)); m=r['meta']; s=r['summary']
    head=m['head']; native='evidential_native' if head=='evidential' else 'softmax'
    score='vacuity_native' if head=='evidential' else 'msp'
    near_key=[k for k in s if k.startswith('ood_auroc__') and k.endswith('__msp') and 'near' in k and 'tin' not in k][0].split('__')[1]
    row=dict(dataset=m['dataset'],shot=int(m['k_shot']),backbone=m['backbone'],adapter=m['adapter'],head=head,seed=int(m['seed']),
             n_params=int(r['n_params']),
             acc=s['accuracy_mean__'+native], f1=s['f1_macro_mean__'+native],
             ece=s['ece_pooled__'+native], ece_ep=s['ece_per_episode_mean__'+native], brier=s['brier_mean__'+native],
             ece_ts=s.get('ece_pooled__ts'), near_pool=near_key, n_ep=s.get('num_episodes'))
    for p in ['svhn_far','gaussian_far',near_key,'tin_near']:
        tag='near' if p==near_key else p
        for sc in ['msp','ts_msp','energy','vacuity_native','vacuity_valfit']:
            row[f'auroc_{tag}_{sc}']=s.get(f'ood_auroc__{p}__{sc}')
            row[f'fpr_{tag}_{sc}']=s.get(f'fpr_at_95_tpr__{p}__{sc}')
        row[f'auroc_{tag}']=s[f'ood_auroc__{p}__{score}']; row[f'fpr_{tag}']=s[f'fpr_at_95_tpr__{p}__{score}']
    rows.append(row)
print(len(rows))
json.dump(rows,open(os.path.dirname(__file__)+'/rows.json','w'),indent=0)
# cross-check vs grid
mv=json.load(open(REPO+'/results/mvt_results.json'))['results']
amap={'bottleneck_parallel':'bottleneck_parallel','lora':'lora','full_ft':'full_ft','linear_probe':'linear_probe'}
bad=[]
for r in rows:
    try: c=mv[r['dataset']][f"{r['shot']}shot"][r['backbone']][r['adapter']][r['head']]
    except KeyError as e: bad.append(('missing',r['dataset'],r['adapter'],r['head'])); continue
    sd=str(r['seed'])
    for k_m,k_r in [('accuracy_mean','acc'),('ece_pooled','ece'),('f1_macro_mean','f1')]:
        if abs(c[k_m]['per_seed'][sd]-r[k_r])>1e-9: bad.append((r['dataset'],r['shot'],r['backbone'],r['adapter'],r['head'],sd,k_m,c[k_m]['per_seed'][sd],r[k_r]))
print('mismatches:',len(bad)); [print(b) for b in bad[:20]]
print(set(r['n_ep'] for r in rows))
