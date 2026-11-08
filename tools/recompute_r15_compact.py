"""Recompute R15 accuracy statistics from lossless argmax decision exports.

These are derived predictions, not logits or independently executed inference.
Export provenance binds each array to the original full-logit archive.
"""
import argparse
import hashlib
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(3101, 3106))
TASKS = list(range(6, 10))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    x = np.array(values, dtype=np.float64)
    draws = np.array(list(itertools.product(range(5), repeat=5)), dtype=np.int64)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)),
            'approximate_bootstrap_percentile_95': np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
            'bootstrap_resamples': 3125}


def recompute(root, output):
    if output.exists():
        raise FileExistsError(output)
    folder = root / 'evidence/r15_decisions'
    provenance = json.loads((folder / 'EXPORT.json').read_text())
    archived = json.loads((folder / 'reference_analysis.json').read_text())
    assert sha(folder / 'reference_analysis.json') == provenance['reference_analysis_sha256']
    expected = {(s,m,t,b) for s in SEEDS for m in ('sgd','er') for t in TASKS for b in ('actual','probe','fresh')}
    records, identities, predictions = {}, {}, {}
    refs = {(r['seed'],r['method'],r['task'],r['branch']):r for r in archived['raw_endpoint_counts']}
    for item in provenance['endpoints']:
        key = item['seed'],item['method'],item['task'],item['branch']
        assert key in expected and key not in records
        seed,method,task,branch = key
        p = folder / item['file']
        assert sha(p) == item['sha256'] and item['source_npz_sha256'] == refs[key]['sha256']
        with np.load(p, allow_pickle=False) as z:
            indices, labels, original, pred, local = [z[n] for n in ('indices','mapped_labels','original_labels','seen_predictions','current_predictions')]
        assert all(a.shape == (10000,) and np.issubdtype(a.dtype,np.integer) for a in (indices,labels,original,pred,local))
        assert np.array_equal(np.sort(indices),np.arange(10000))
        assert np.all((original >= 0) & (original < 100))
        assert np.array_equal(np.bincount(original,minlength=100),np.full(100,100))
        order = np.random.Generator(np.random.PCG64(seed+40000)).permutation(100)
        assert np.array_equal(labels,np.argsort(order)[original])
        assert np.all((pred>=0)&(pred<10*(task+1))) and np.all((local>=10*task)&(local<10*(task+1)))
        identity = hashlib.sha256(np.stack([indices,original,labels]).astype('<i8').tobytes()).hexdigest()
        assert identity == refs[key]['identity_sha256']
        assert identities.setdefault((seed,task),identity) == identity
        count = lambda mask: {'correct':int(((pred==labels)&mask).sum()),'n':int(mask.sum())}
        current = labels//10 == task
        rec = {'current':count(current), 'old':count(labels//10<task), 'seen':count(labels//10<=task),
               'current_masked':{'correct':int(((local==labels)&current).sum()),'n':int(current.sum())},
               'old_tasks':[count(labels//10==t) for t in range(task)],
               'current_predictions_to_old':int(((pred<10*task)&current).sum())}
        assert all(rec[n] == refs[key][n] for n in rec), key
        records[key] = rec
        predictions[key] = pred.copy(),local.copy()
    assert set(records) == expected
    for s in SEEDS:
        for t in TASKS:
            for a,b in [(('sgd','actual'),('sgd','probe')), (('sgd','fresh'),('er','fresh'))]:
                assert all(np.array_equal(x,y) for x,y in zip(predictions[s,a[0],t,a[1]], predictions[s,b[0],t,b[1]]))
    pct = lambda r: Q(100*r['correct'],r['n'])
    paired = []
    for seed in SEEDS:
        row = {'seed':seed}
        for name,branch,metric in [('d_actual','actual','current'),('d_probe','probe','current'),('d_masked','actual','current_masked')]:
            values = [pct(records[seed,'er',t,branch][metric])-pct(records[seed,'sgd',t,branch][metric]) for t in TASKS]
            mean = sum(values,Q())/4
            row.update({name:float(mean), name+'_exact':str(mean), name+'_per_task':[float(x) for x in values]})
        row['joint_direction'] = row['d_actual']<0 and row['d_probe']>0 and row['d_masked']>0
        paired.append(row)
    assert paired == archived['paired_seed_contrasts']
    summary = {}
    for name,sign in [('d_actual',-1),('d_probe',1),('d_masked',1)]:
        summary[name] = stats([r[name] for r in paired])
        summary[name]['direction_count'] = sum(sign*Q(r[name+'_exact'])>0 for r in paired)
        summary[name]['zero_count'] = sum(Q(r[name+'_exact'])==0 for r in paired)
    assert summary == archived['summary']
    # Keep the historical hypothesis count for provenance, and make its sign
    # explicit so it cannot be mistaken for the number of ER advantages.
    for name, sign in [('d_actual', -1), ('d_probe', 1), ('d_masked', 1)]:
        values = [Q(r[name+'_exact']) for r in paired]
        summary[name]['direction_count_means'] = 'ER_minus_SGD_negative' if sign < 0 else 'ER_minus_SGD_positive'
        summary[name]['positive_count'] = sum(v > 0 for v in values)
        summary[name]['negative_count'] = sum(v < 0 for v in values)
    endpoints = []
    for method in ('sgd','er'):
        for branch in ('actual','probe','fresh'):
            row = {'method':method,'branch':branch}
            for metric in ('current','current_masked','old','seen'):
                values = [float(sum((pct(records[s,method,t,branch][metric]) for t in TASKS),Q())/4) for s in SEEDS]
                row[metric] = {'per_seed':values,**stats(values)}
            endpoints.append(row)
    assert endpoints == archived['endpoint_summary']
    result = {'schema':'r15_analysis_v1','seeds':SEEDS,'n_endpoints':120,
              'manifest_sha256':archived['manifest_sha256'], 'paired_seed_contrasts':paired,
              'summary':summary,'endpoint_summary':endpoints,
              'joint_direction_seeds':sum(r['joint_direction'] for r in paired),
              'scope':'Recomputed from derived per-example argmax decisions; no source logits, weights or fresh inference in this compact path.'}
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','endpoints':120,'paired_seeds':5,'summary':summary},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();recompute(a.root.resolve(),a.output.resolve())
