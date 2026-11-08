"""Fit one old/new score offset on validation images, then evaluate test images.

This is an ordinary classifier-score control. It updates no network parameter
and never uses the true group of an individual image at prediction time.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','score'],required=True)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def desc(v):return dict(n=len(v),mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)))
def fit(z,y):
    def lse(x):
        m=x.max(1);return m+np.log(np.exp(x-m[:,None]).sum(1))
    delta=lse(z[:,8:])-lse(z[:,:8]);target=float(np.mean(y>=8))
    def grad(b):return np.mean(1/(1+np.exp(np.clip(-delta-b,-700,700))))-target
    lo,hi=-1.,1.
    while grad(lo)>0:lo*=2
    while grad(hi)<0:hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if grad(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2
if a.phase=='fit':
    a.output.mkdir(parents=True,exist_ok=False);fits=[]
    for seed in range(201,211):
        for method in ['er','sgd']:
            f=ROOT/f'evidence/weight_retention/validation_predictions/s{seed}_{method}_standard.npz'
            with np.load(f,allow_pickle=False) as d:
                z=d['logits'].astype(np.float64);y=d['labels']
                assert z.shape==(1000,10) and np.array_equal(np.bincount(y),np.full(10,100))
                assert np.isfinite(z).all()
                b=fit(z,y)
            fits.append(dict(seed=seed,method=method,b=b,validation_file=str(f.relative_to(ROOT)),validation_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (a.output/'fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8')
    print('Saved 20 validation-only fits. Test files were not read.')
else:
    assert not (a.output/'test_results.json').exists()
    rows=[]
    for r in read(a.output/'fits.json'):
        validation_file=ROOT/r['validation_file']
        assert hashlib.sha256(validation_file.read_bytes()).hexdigest()==r['validation_sha256']
        assert np.isfinite(r['b'])
        run=ROOT/f'evidence/retention_selection_original/split_cifar10_classil_{r["method"]}_s{r["seed"]}'
        with np.load(run/'test.npz',allow_pickle=False) as d:
            y=d['labels'];z=d['logits_actual'].astype(np.float64)
            assert z.shape==(1000,10) and np.array_equal(np.bincount(y),np.full(10,100))
            assert np.isfinite(z).all()
            for condition,b in [('usual',0.),('adjusted',r['b'])]:
                scores=z.copy();scores[:,8:]+=b;pred=scores.argmax(1)
                measures={}
                for metric,mask in [('old',y<8),('current',y>=8),('all',np.ones(len(y),dtype=bool))]:
                    correct=int((pred[mask]==y[mask]).sum());n=int(mask.sum())
                    measures[metric]=dict(correct=correct,n=n,accuracy=100*correct/n)
                rows.append(dict(seed=r['seed'],method=r['method'],condition=condition,b=b,**measures))
    summary={};paired={}
    for method in ['er','sgd']:
        for condition in ['usual','adjusted']:
            take=[r for r in rows if r['method']==method and r['condition']==condition]
            summary[method+'_'+condition]={m:desc([r[m]['accuracy'] for r in take]) for m in ['old','current','all']}
    for condition in ['usual','adjusted']:
        for metric in ['old','current','all']:
            vals=[next(r[metric]['accuracy'] for r in rows if r['seed']==s and r['method']=='er' and r['condition']==condition)-next(r[metric]['accuracy'] for r in rows if r['seed']==s and r['method']=='sgd' and r['condition']==condition) for s in range(201,211)]
            paired[condition+'_'+metric]=desc(vals)|dict(values=vals,positive=sum(v>0 for v in vals))
    result=dict(rows=rows,summary=summary,paired=paired)
    (a.output/'test_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))
