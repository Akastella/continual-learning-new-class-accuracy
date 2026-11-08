"""CPU-only recomputation of prediction classes, retention, budgets, and CBP.

The small decision arrays preserve argmax outputs from genuine archived logits.
This command checks saved evidence and statistics; it does not rerun training.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def describe(xs):
    values = [float(x) for x in xs]
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else 0.0,
                positive=sum(x>0 for x in xs), negative=sum(x<0 for x in xs),
                zero=sum(x==0 for x in xs))


def mean(xs):
    return sum(xs,Q())/len(xs)


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision_statistics(root):
    directory=root/'evidence/ijcnn_decisions'
    manifest=read(directory/'manifest.json')
    groups=defaultdict(dict)
    endpoint_rows=[]
    for row in manifest['records']:
        path=directory/row['file']
        assert sha(path)==row['sha256'],path
        with np.load(path,allow_pickle=False) as z:
            y,pred,local,ids=[z[k] for k in ('labels','prediction','current_prediction','indices')]
        assert y.shape==pred.shape==local.shape==ids.shape
        assert y.ndim==1 and np.unique(ids).size==len(y)
        assert np.all(y<10) and np.all(pred<10)
        t=row['task']; current=y//2==t; old=y//2<t; seen=y//2<=t
        assert int((pred[current]==y[current]).sum())==row['current_correct']
        assert int(current.sum())==row['current_n']==2000
        assert int((local[current]==y[current]).sum())==row['current_masked_correct']
        metrics={}
        for name,mask,prediction in [('current',current,pred),('current_classes',current,local),('old',old,pred),('seen',seen,pred),('all',np.ones(len(y),dtype=bool),pred)]:
            if not mask.any():
                continue
            correct,n=int((prediction[mask]==y[mask]).sum()),int(mask.sum())
            metrics[name]=Q(100*correct,n)
        full=pred[current]; local_cur=local[current]; target=y[current]
        for name,mask in [('rescued_old',(full<2*t)&(local_cur==target)),('rescued_future',(full>=2*t+2)&(local_cur==target))]:
            metrics[name]=Q(100*int(mask.sum()),len(target))
        for task_group in ('mean_noninitial','last_task'):
            if task_group=='last_task' and t!=4:
                continue
            for metric,value in metrics.items():
                key=(row['stage'],row['method'],row['branch'],row['epoch'],task_group,metric)
                groups[key].setdefault(row['seed'],[]).append(value)
        endpoint_rows.append({k:row[k] for k in ('stage','seed','method','branch','task','epoch')}|{k:float(v) for k,v in metrics.items()})
    seed_rows=[]; summaries=[]; means={}
    for key, by_seed in sorted(groups.items()):
        stage,method,branch,epoch,tasks,metric=key
        identity=dict(stage=stage,method=method,branch=branch,epoch=epoch,tasks=tasks,metric=metric)
        for seed,values in sorted(by_seed.items()):
            required=4 if tasks=='mean_noninitial' and stage!='final_budget' else 1
            assert len(values)==required,(key,seed,len(values))
            v=mean(values); means[key+(seed,)]=v
            seed_rows.append(identity|dict(seed=seed,value=float(v),exact=str(v)))
        summaries.append(identity|describe([means[key+(s,)] for s in sorted(by_seed)]))
    contrasts=[]; paired=[]
    for stage in ('head','budget','retention'):
        identities=sorted({(branch,epoch,tasks,metric) for st,method,branch,epoch,tasks,metric in groups if st==stage and method=='sgd'})
        for branch,epoch,tasks,metric in identities:
            left=(stage,'er',branch,epoch,tasks,metric)
            right=(stage,'sgd',branch,epoch,tasks,metric)
            if left not in groups:
                continue
            seeds=sorted(groups[left])
            assert seeds==sorted(groups[right])==list(range(201,211))
            identity=dict(stage=stage,contrast='er_minus_sgd',branch=branch,epoch=epoch,tasks=tasks,metric=metric)
            xs=[means[left+(s,)]-means[right+(s,)] for s in seeds]
            contrasts.append(identity|describe(xs))
            paired.extend(identity|dict(seed=s,value=float(v),exact=str(v)) for s,v in zip(seeds,xs))
    # Final-task DER++ uses its own continuation source; comparison uses paired ER/SGD endpoints.
    for baseline in ('sgd','er'):
        for branch in ('actual','probe'):
            for epoch in (4,8,16):
                for metric in ('current','current_classes','old','seen'):
                    left=('final_budget','derpp',branch,epoch,'last_task',metric)
                    right=('budget',baseline,branch,epoch,'last_task',metric)
                    if left not in groups:
                        continue
                    xs=[means[left+(s,)]-means[right+(s,)] for s in range(201,211)]
                    identity=dict(stage='final_budget',contrast='derpp_minus_'+baseline,branch=branch,epoch=epoch,tasks='last_task',metric=metric)
                    contrasts.append(identity|describe(xs))
                    paired.extend(identity|dict(seed=s,value=float(v),exact=str(v)) for s,v in zip(range(201,211),xs))
    return dict(endpoints=len(endpoint_rows),endpoint_rows=endpoint_rows,seed_rows=seed_rows,
                summaries=summaries,contrasts=contrasts,paired_values=paired)


def original_selection(root):
    directory=root/'evidence/retention_selection_original'
    expected=read(directory/'selection.json')
    seal=read(directory/'selection_seal.json')
    assert sha(directory/'selection.json')==seal['selection_sha256']
    for relative,expected_hash in seal['validation_files_sha256'].items():
        assert sha(directory/relative)==expected_hash,relative
    with np.load(directory/'split.npz',allow_pickle=False) as raw:
        split={name:raw[name].copy() for name in raw.files}
    selection_rows=[]; anchors=[]
    for seed in range(201,211):
        for method in ('sgd','er'):
            rid=f'split_cifar10_classil_{method}_s{seed}'
            run=directory/rid
            val={}; test={}
            for phase,values in [('validation',val),('test',test)]:
                with np.load(run/(phase+'.npz'),allow_pickle=False) as z:
                    y=z['labels']
                    assert y.shape==(1000,) and np.array_equal(np.bincount(y),np.full(10,100))
                    ids=split[phase+'_indices']
                    assert np.array_equal(z['indices'],ids)
                    order=np.random.Generator(np.random.PCG64(seed+40000)).permutation(10)
                    assert np.array_equal(y,np.argsort(order)[split['original_labels'][ids]])
                    if phase=='test':
                        metadata=read(run/'test.json')
                        assert metadata['npz_sha256']==sha(run/'test.npz')
                        assert metadata['selection_seal_sha256']==sha(directory/'selection_seal.json')
                    for name in z.files:
                        if not name.startswith('logits_'):
                            continue
                        logits=z[name]
                        assert logits.shape==(1000,10) and np.isfinite(logits).all()
                        pred=logits.argmax(1)
                        key=int(name[len('logits_'):]) if phase=='validation' else name[len('logits_'):]
                        values[key]={'old_correct':int(((pred==y)&(y<8)).sum()),'current_correct':int(((pred==y)&(y>=8)).sum())}
            for threshold in (20,10,0):
                eligible=[s for s in val if 100*val[s]['old_correct']>=threshold*800]
                step=min(eligible,key=lambda s:(-val[s]['current_correct'],s)) if eligible else None
                assert expected[rid][str(threshold)]==step,(rid,threshold,step,expected[rid])
                row=dict(seed=seed,method=method,threshold=threshold,selected_step=step)
                if step is not None:
                    scored=test['step_'+str(step)]
                    row.update(old=scored['old_correct']/8,current=scored['current_correct']/2,
                               test_violation=scored['old_correct']*100<threshold*800)
                selection_rows.append(row)
            scored=test['actual']
            anchors.append(dict(seed=seed,method=method,old=scored['old_correct']/8,current=scored['current_correct']/2))
    groups=[]
    for threshold in (20,10,0):
        for method in ('sgd','er'):
            rows=[r for r in selection_rows if r['method']==method and r['threshold']==threshold and r['selected_step'] is not None]
            groups.append(dict(method=method,threshold=threshold,n=len(rows),violations=sum(r['test_violation'] for r in rows),
                               old=describe([r['old'] for r in rows]),current=describe([r['current'] for r in rows])))
    return dict(selection_rows=selection_rows,groups=groups,actual_anchors=anchors,
                actual_groups={method:{metric:describe([r[metric] for r in anchors if r['method']==method]) for metric in ('old','current')} for method in ('sgd','er')})


def write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve(); out=a.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory: '+str(out))
    out.mkdir(parents=True)
    decisions=decision_statistics(root)
    selection=original_selection(root)
    primary=load_module(root/'tools/recompute_reviewer_subset.py','ijcnn_primary').primary(root)
    result=dict(status='PASS',analysis_unit='Paired seed; task scores equally averaged within seed before seed statistics.',
                source='Archived dataset-label counts and lossless prediction decisions; no new training or forward inference.',
                primary=primary,selection=selection,**decisions)
    (out/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(out/'accuracy_summary.csv',decisions['summaries'])
    write_csv(out/'paired_comparisons.csv',decisions['contrasts'])
    write_csv(out/'paired_seed_values.csv',decisions['paired_values'])
    write_csv(out/'endpoint_accuracies.csv',decisions['endpoint_rows'])
    write_csv(out/'selection.csv',selection['selection_rows'])
    print(json.dumps(dict(status='PASS',endpoints=decisions['endpoints'],summary_rows=len(decisions['summaries']),paired_comparisons=len(decisions['contrasts']),selection_rows=len(selection['selection_rows']))))


if __name__=='__main__':
    main()
