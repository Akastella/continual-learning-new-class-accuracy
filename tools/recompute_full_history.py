"""CPU-only reproduction of all longer-history statistics from saved logits."""
from pathlib import Path
import argparse,json,csv
from collections import defaultdict
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--raw',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);rows=[];units=defaultdict(list)
for file in sorted(a.raw.glob('*.npz')):
    parts=file.stem.split('_');method=parts[0];seed=int(parts[1][1:]);task=int(parts[2][1:]);branch='_'.join(parts[3:])
    with np.load(file,allow_pickle=False) as d:
        y=d['labels'];ids=d['official_indices'];tid=d['task_ids']
        assert y.shape==tid.shape==ids.shape==(10000,) and np.array_equal(np.sort(ids),np.arange(10000))
        assert np.array_equal(tid,y//2) and np.array_equal(np.bincount(y),np.full(10,1000))
        if 'logits' in d:
            z=d['logits'];assert np.isfinite(z).all() and z.shape==(10000,10)
            pred=z.argmax(1);local=z[:,2*task:2*task+2].argmax(1)+2*task
        else:
            pred=d['predictions'];local=d['current_class_predictions']
            assert pred.shape==local.shape==y.shape and np.all((0<=pred)&(pred<10))
            assert np.all((2*task<=local)&(local<2*task+2))
        for metric,mask,guess in [('current',tid==task,pred),('old',tid<task,pred),('seen',tid<=task,pred),('current_classes',tid==task,local)]:
            n=int(mask.sum());correct=int((guess[mask]==y[mask]).sum());v=100*correct/n
            rows.append(dict(seed=seed,method=method,task=task,branch=branch,metric=metric,correct=correct,n=n,accuracy=v))
            units[method,seed,branch,metric].append(v)
assert len(rows)==480 and all(len(v)==4 for v in units.values())
paired=[];summary=[]
for method in ['sgd','er']:
    for branch in ['usual','current_only','initialized']:
        for metric in ['current','old','seen','current_classes']:
            vals=[np.mean(units[method,seed,branch,metric]) for seed in range(201,206)]
            summary.append(dict(method=method,branch=branch,metric=metric,n=5,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1))))
for branch in ['usual','current_only','initialized']:
    for metric in ['current','old','seen','current_classes']:
        vals=[np.mean(units['er',s,branch,metric])-np.mean(units['sgd',s,branch,metric]) for s in range(201,206)]
        paired.append(dict(branch=branch,metric=metric,n=5,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),positive=int(sum(v>1e-10 for v in vals)),negative=int(sum(v < -1e-10 for v in vals)),values=[float(v) for v in vals]))
for name,data in [('endpoint_counts',rows),('accuracy_summary',summary)]:
    with (a.output/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(a.output/'paired_summary.json').write_text(json.dumps(paired,indent=2))
print(json.dumps(paired,indent=2))
