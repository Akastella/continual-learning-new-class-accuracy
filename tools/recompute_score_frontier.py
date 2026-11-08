"""Recompute the score comparison from a portable evidence directory (NumPy only).

Expected layout: ARTIFACT/evidence/{retention_selection_original,weight_retention}
and DATA/{choices.json,test_logits/,diagnostics/test_logits/}. No training imports.
"""
from pathlib import Path
import argparse, csv, json, math
import numpy as np

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def all_names():
    return [f'{m}_current_only_step_{s:04d}' for m in ['er','sgd'] for s in [0,1,2,4,8,16,32,64,128,157,314,471,628]]+['er_standard','sgd_standard','er_current_weighted','er_current_weighted_unit_sum']
def loadval(artifact,s,n):
    if '_current_only_step_' in n:
        p=artifact/f'evidence/retention_selection_original/split_cifar10_classil_{n.split("_")[0]}_s{s}/validation.npz'
        key=f'logits_{int(n.rsplit("_",1)[1])}'
    else:p=artifact/f'evidence/weight_retention/validation_predictions/s{s}_{n}.npz';key='logits'
    with np.load(p,allow_pickle=False) as d:return d[key].astype(float),d['labels'].copy()
def calc(z,y,b):
    z=z.copy();z[:,8:]+=b;p=z.argmax(1)
    return {k:100*float(np.mean(p[m]==y[m])) for k,m in [('old',y<8),('current',y>=8),('all',np.ones(len(y),bool))]}
def upper(z,y):return 100*float(np.mean(z[y<8,:8].argmax(1)==y[y<8]))
def optimal(z,y,q):
    m=(y<8)&(z[:,:8].argmax(1)==y);r=z[m,:8].max(1)-z[m,8:].max(1);k=math.ceil(q*sum(y<8)/100)
    return float(sorted(r,reverse=True)[k-1]) if len(r)>=k else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    choices=[];checked=0;val={}
    for s in range(201,211):
        for n in all_names():
            z,y=loadval(a.artifact,s,n);val[s,n]=z,y
            for q in [10,20,30]:
                b=optimal(z,y,q)
                if b is not None:
                    # Independent check: full ten-class argmax satisfies q;
                    # any larger offset loses at least one required old success.
                    assert calc(z,y,b)['old']+1e-10>=q
                    # A full-score addition can round a one-ULP change away.
                    assert calc(z,y,b+1e-10*max(1.,abs(b),float(np.abs(z).max())))['old']<q
                    checked+=1
        for q in [10,20,30]:
            for policy in ['raw26','raw28','raw30','offset_usual','offset30']:
                ns=all_names()
                if policy=='raw26':ns=ns[:26]
                elif policy=='raw28':ns=ns[:28]
                elif policy=='offset_usual':ns=['er_standard','sgd_standard']
                eligible=[]
                for n in ns:
                    z,y=val[s,n];b=optimal(z,y,q) if policy.startswith('offset') else 0.
                    if b is not None:
                        sc=calc(z,y,b)
                        if sc['old']+1e-10>=q:eligible.append((n,b,sc))
                chosen=min(eligible,key=lambda x:(-x[2]['current'],x[0]))
                choices.append(dict(seed=s,q=q,policy=policy,candidate=chosen[0],b=chosen[1]))
    original=read(a.data/'choices.json')['rows']
    for r in choices:
        ref=next(x for x in original if all(r[k]==x[k] for k in ['seed','q','policy']))
        assert r['candidate']==ref['candidate'] and r['b']==ref['b']
    results=[];upperrows=[]
    for dataset in ['cifar101','cifar102']:
        for s in range(201,211):
            for n in all_names():
                path=a.data/f'diagnostics/test_logits/{dataset}_s{s}_{n}.npz'
                if not path.exists():path=a.data/f'test_logits/{dataset}_s{s}_{n}.npz'
                with np.load(path,allow_pickle=False) as d:z,y=d['logits'].astype(float),d['labels'].copy()
                zv,yv=val[s,n]
                upperrows.append(dict(dataset=dataset,seed=s,candidate=n,validation_upper=upper(zv,yv),test_upper=upper(z,y)))
                for r in choices:
                    if r['seed']==s and r['candidate']==n:
                        sc=calc(z,y,r['b']);results.append(dict(dataset=dataset,**r,**sc,violation=sc['old']<r['q']))
    original_rows=list(csv.DictReader((a.data/'test_scores.csv').open(encoding='utf8')))
    for row in results:
        ref=next(r for r in original_rows if r['dataset']==row['dataset'] and int(r['seed'])==row['seed'] and int(r['q'])==row['q'] and r['policy']==row['policy'])
        assert all(abs(row[k]-float(ref[k]))<1e-10 for k in ['old','current','all'])
    for file,rows in [('test_scores.csv',results),('upper_bound_transfer.csv',upperrows)]:
        with (a.output/file).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(a.output/'verification.json',dict(status='PASS',validation_models=300,maximum_offset_checks=checked,choices_verified=len(choices),test_choices_verified=len(results),test_candidate_models=len(upperrows)))
    print(json.dumps(read(a.output/'verification.json')))
if __name__=='__main__':main()
