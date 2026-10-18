import json,os
D=os.path.dirname(os.path.abspath(__file__))
agg={tuple(k.split('|')):v for k,v in json.load(open(D+'/agg.json')).items()}
DS={'cifar_fs':'CIFAR-FS','mini_imagenet':'MiniImageNet'}
BB={'resnet18':'ResNet-18','mobilenetv3_small':'MobileNetV3-S'}
AD={'bottleneck_parallel':'Bottleneck','lora':'LoRA','full_ft':'Full FT$^\\ast$','linear_probe':'Linear probe$^\\ast$'}
HD={'evidential':'Evid.','softmax':'Softmax'}
order=[]
for ds in ['cifar_fs','mini_imagenet']:
  for sh in ['1','5']:
    for bb in ['resnet18','mobilenetv3_small']:
      for ad in ['bottleneck_parallel','lora','full_ft','linear_probe']:
        for hd in ['evidential','softmax']:
          if (ds,sh,bb,ad,hd) in agg: order.append((ds,sh,bb,ad,hd))
print(len(order))
def lead(k,prev):
    ds,sh,bb,ad,hd=k
    a=DS[ds] if (prev is None or prev[0]!=ds) else ''
    b=f'{sh}' if (prev is None or prev[:2]!=k[:2]) else ''
    c=BB[bb] if (prev is None or prev[:3]!=k[:3]) else ''
    return a,b,c
out={}
L=[];prev=None
for k in order:
    v=agg[k];a,b,c=lead(k,prev);prev=k
    L.append(f"{a} & {b} & {c} & {AD[k[3]]} & {HD[k[4]]} & {int(round(v['n_params'])):,} & {100*v['acc']:.2f} & {100*v['f1']:.2f} \\\\")
out['A1']='\n'.join(L)
L=[];prev=None
for k in order:
    v=agg[k];a,b,c=lead(k,prev);prev=k
    ts=f"{v['ece_ts']:.4f}" if k[4]=='softmax' else '--'
    L.append(f"{a} & {b} & {c} & {AD[k[3]]} & {HD[k[4]]} & {v['ece']:.4f} & {v['ece_ep']:.4f} & {v['brier']:.4f} & {ts} \\\\")
out['A2']='\n'.join(L)
L=[];prev=None
for k in order:
    v=agg[k];a,b,c=lead(k,prev);prev=k
    xs=[v['auroc_svhn_far'],v['auroc_gaussian_far'],v['auroc_near'],v['auroc_tin_near']]
    L.append(f"{a} & {b} & {c} & {AD[k[3]]} & {HD[k[4]]} & "+' & '.join(f'{x:.3f}' for x in xs)+" \\\\")
out['A3']='\n'.join(L)
L=[];prev=None
for k in order:
    v=agg[k];a,b,c=lead(k,prev);prev=k
    xs=[v['fpr_svhn_far'],v['fpr_gaussian_far'],v['fpr_near'],v['fpr_tin_near']]
    L.append(f"{a} & {b} & {c} & {AD[k[3]]} & {HD[k[4]]} & "+' & '.join(f'{x:.3f}' for x in xs)+" \\\\")
out['A4']='\n'.join(L)
json.dump(out,open(D+'/tables.json','w'))
short={k:v.replace('MobileNetV3-S','MNv3-S').replace('MiniImageNet','MiniIN').replace('Linear probe$^\\ast$','Lin.\\ probe$^\\ast$') for k,v in out.items()}
json.dump(short,open(D+'/tables_short.json','w'))
k=('cifar_fs','5','mobilenetv3_small','lora','evidential'); v=agg[k]
print('corrected cell',{x:round(v[x],4) for x in ['acc','f1','ece','ece_ep','brier','auroc_svhn_far','auroc_gaussian_far','auroc_near','auroc_tin_near','fpr_svhn_far']})
print(out['A1'][:400])
