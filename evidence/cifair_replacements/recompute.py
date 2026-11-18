"""Recompute all paired replacement-image results with Python and NumPy only."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p,rs):
    cols=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rs)
def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,
       'sample_sd':float(a.std(ddof=1)) if len(a)>1 else None,'values':a.tolist()}
def metric(correct,take):
    n=int(take.sum());c=int(correct[take].sum())
    return {'n':n,'correct':c,'accuracy_pct':100*c/n if n else None}

def calculate(root):
    inputs=json.loads((root/'inputs.json').read_text(encoding='utf-8'))['models']
    completed=json.loads((root/'execution_complete.json').read_text(encoding='utf-8'))
    assert completed['status']=='complete' and len(completed['results'])==20
    scores=[];old_tasks=[];counts=[];data_lookup={}
    for result in completed['results']:
        path=root/result['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==result['sha256']
        with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        seed,task,condition=int(d['seed']),int(d['task']),str(d['condition'])
        item=next(r for r in inputs if (r['seed'],r['task'],r['condition'])==(seed,task,condition))
        assert float(d['offset'])==item['offset'] and item['q']==20 and item['calibration_margin_pp']==3
        y=d['mapped_labels'];old=10*task;seen=old+10;n=len(y)
        assert np.all(y<seen) and np.array_equal(d['class_order'][y],d['original_labels'])
        assert len(np.unique(d['test_rows']))==n
        data_lookup[(task,seed,condition)]=d
        for mode,bias in [('raw',0.),('adjusted',float(d['offset']))]:
            pred=np.where(d['current_score'].astype(np.float64)+bias>d['old_score'].astype(np.float64),
                d['current_prediction'],d['old_prediction'])
            ok=pred==y;old_take=y<old;new_take=y>=old
            row={'task_zero_based':task,'task_one_based':task+1,'seed':seed,'condition':condition,'mode':mode,'offset':bias,
                 'old_target_below20':int(ok[old_take].sum())*5<int(old_take.sum())}
            for label,take in [('new',new_take),('old',old_take),('seen',np.ones(n,dtype=bool))]:
                row.update({label+'_'+k:v for k,v in metric(ok,take).items()})
            row['current10_accuracy_pct']=100*int((d['current_prediction'][new_take]==y[new_take]).sum())/int(new_take.sum())
            scores.append(row)
            for earlier in range(task):
                old_tasks.append({'task_zero_based':task,'task_one_based':task+1,'seed':seed,
                    'condition':condition,'mode':mode,'old_task_zero_based':earlier,
                    'old_task_one_based':earlier+1,**metric(ok,y//10==earlier)})
        if condition=='full':
            for original in range(100):
                mapped=int(np.flatnonzero(d['class_order']==original)[0])
                counts.append({'task_zero_based':task,'seed':seed,'original_fine_label':original,
                    'mapped_label':mapped,'role':'old' if mapped<old else 'new' if mapped<seen else 'unseen',
                    'evaluated_n':int((d['original_labels']==original).sum())})
    lookup={(r['task_zero_based'],r['seed'],r['condition'],r['mode']):r for r in scores}
    pairs=[]
    for task in (7,9):
        for seed in range(3101,3106):
            a=data_lookup[(task,seed,'full')];b=data_lookup[(task,seed,'restricted')]
            for key in ('test_rows','original_labels','mapped_labels','class_order'):assert np.array_equal(a[key],b[key])
            f,r,fa,ra=[lookup[(task,seed,c,m)] for c,m in [('full','raw'),('restricted','raw'),('full','adjusted'),('restricted','adjusted')]]
            assert f['new_n']==r['new_n']==fa['new_n']==ra['new_n']
            raw=f['new_accuracy_pct']-r['new_accuracy_pct']
            adjusted=fa['new_accuracy_pct']-ra['new_accuracy_pct']
            reduction=raw-abs(adjusted)
            # An absolute adjusted gap prevents apparent success through reversal/cancellation.
            passed=raw>0 and abs(adjusted)<=raw/2+1e-12
            pairs.append({'task_zero_based':task,'task_one_based':task+1,'seed':seed,'new_n':f['new_n'],'old_n':f['old_n'],
                'raw_gap_pp':raw,'adjusted_gap_pp':adjusted,'absolute_adjusted_gap_pp':abs(adjusted),
                'attenuation_pp':reduction,'half_reduction_surplus_pp':raw/2-abs(adjusted),
                'raw_gap_positive':raw>0,'at_least_half_reduction':passed,
                'adjusted_gap_reversed':adjusted<0})
    boundaries={}
    for task in (7,9):
        ps=[r for r in pairs if r['task_zero_based']==task]
        raw=stats([r['raw_gap_pp'] for r in ps]);absolute=stats([r['absolute_adjusted_gap_pp'] for r in ps])
        success=raw['mean']>0 and absolute['mean']<=raw['mean']/2+1e-12 and sum(r['at_least_half_reduction'] for r in ps)>=3
        models=[]
        for condition in ('full','restricted'):
            for mode in ('raw','adjusted'):
                ss=[r for r in scores if r['task_zero_based']==task and r['condition']==condition and r['mode']==mode]
                models.append({'condition':condition,'mode':mode,
                    'new_accuracy_pct':stats([r['new_accuracy_pct'] for r in ss]),
                    'old_accuracy_pct':stats([r['old_accuracy_pct'] for r in ss]),
                    'seen_accuracy_pct':stats([r['seen_accuracy_pct'] for r in ss]),
                    'old_target_below20_n':sum(r['old_target_below20'] for r in ss),
                    'old_target_below20_seeds':[r['seed'] for r in ss if r['old_target_below20']]})
        old_task_summary=[]
        for condition in ('full','restricted'):
            for mode in ('raw','adjusted'):
                for earlier in range(task):
                    ss=[r for r in old_tasks if r['task_zero_based']==task and r['condition']==condition
                        and r['mode']==mode and r['old_task_zero_based']==earlier]
                    old_task_summary.append({'condition':condition,'mode':mode,'old_task_one_based':earlier+1,
                        'accuracy_pct':stats([r['accuracy_pct'] for r in ss if r['accuracy_pct'] is not None]),
                        'image_counts':[r['n'] for r in ss]})
        boundaries[str(task)]={'task_one_based':task+1,'raw_gap_pp':raw,
            'adjusted_gap_pp':stats([r['adjusted_gap_pp'] for r in ps]),'absolute_adjusted_gap_pp':absolute,
            'attenuation_pp':stats([r['attenuation_pp'] for r in ps]),
            'half_reduction_surplus_pp':stats([r['half_reduction_surplus_pp'] for r in ps]),
            'passing_seed_n':sum(r['at_least_half_reduction'] for r in ps),'prespecified_prediction_supported':success,
            'new_image_counts':[r['new_n'] for r in ps],'old_image_counts':[r['old_n'] for r in ps],
            'models':models,'old_task_summary':old_task_summary}
    summary={'status':'complete','seed_unit':'paired seed, with image-level accuracy within each seed',
        'absolute_reduction_aggregation':'mean of per-seed absolute adjusted gaps; no signed cancellation across seeds',
        'offsets_fitted_on_cifair':0,'models_trained':0,'old_target_is_descriptive_only':True,
        'source_scope':'Only genuinely replaced ciFAIR-100 test rows; targeted duplicate replacements, not a complete independently sampled IID test source.',
        'boundaries':boundaries}
    return scores,pairs,old_tasks,counts,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    scores,pairs,old_tasks,counts,summary=calculate(ROOT)
    rows(out/'seed_scores.csv',scores);rows(out/'paired_scores.csv',pairs)
    rows(out/'old_task_scores.csv',old_tasks);rows(out/'seed_class_counts.csv',counts)
    dump(out/'summary.json',summary)
    print(json.dumps({'status':'passed','npz_files':20,'seed_score_rows':len(scores),'paired_seed_rows':len(pairs),
         'old_task_rows':len(old_tasks),'boundaries':{k:{'raw_gap_pp':v['raw_gap_pp']['mean'],
           'adjusted_gap_pp':v['adjusted_gap_pp']['mean'],'absolute_adjusted_gap_pp':v['absolute_adjusted_gap_pp']['mean'],
           'passing_seed_n':v['passing_seed_n'],'supported':v['prespecified_prediction_supported']} for k,v in summary['boundaries'].items()}},indent=2))

if __name__=='__main__':main()
