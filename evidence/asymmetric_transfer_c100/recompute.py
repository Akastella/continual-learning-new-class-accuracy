"""NumPy-only counts and paired five-seed analysis for the prospective transfer test."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
MODELS=('usual_er','current_mask','common_sgd')
METRICS=('new_all100_pct','new_current10_pct','competition_loss_pp','old_all100_pct','all100_pct','common_minus_usual_pp')


def describe(values):
    a=np.asarray(values,dtype=np.float64)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),
            'positive_seeds':int((a>0).sum()),'zero_seeds':int((a==0).sum()),'negative_seeds':int((a<0).sum())}


def calculate():
    rows=[]
    for seed in range(3101,3106):
        reference_labels=None;reference_indices=None
        for name in MODELS:
            with np.load(ROOT/f's{seed}_{name}.npz',allow_pickle=False) as d:
                pred=d['predictions_all100'];pair=d['predictions_current10'];y=d['mapped_labels']
                idx=d['official_test_indices'];original=d['original_labels'];order=d['class_order']
                assert int(d['seed'])==seed and str(d['model_name'])==name and int(d['task'])==9
                assert pred.shape==pair.shape==y.shape==idx.shape==original.shape==(10000,)
                assert np.array_equal(idx,np.arange(10000))
                assert np.array_equal(order[y],original)
                assert np.all((pair>=90)&(pair<100)) and np.all((pred>=0)&(pred<100))
                assert np.array_equal(np.bincount(y,minlength=100),np.full(100,100))
                if reference_labels is None:reference_labels=y.copy();reference_indices=idx.copy()
                else:assert np.array_equal(y,reference_labels) and np.array_equal(idx,reference_indices)
                new=y>=90;old=~new
                new_correct=int((pred[new]==y[new]).sum());within_correct=int((pair[new]==y[new]).sum())
                competition=int(((pair[new]==y[new])&(pred[new]!=y[new])).sum())
                assert within_correct-new_correct==competition
                assert np.all(pred[new][(pair[new]==y[new])&(pred[new]!=y[new])]<90)
                old_correct=int((pred[old]==y[old]).sum());total_correct=int((pred==y).sum())
                assert new_correct+old_correct==total_correct
                rows.append({'seed':seed,'model':name,'new_n':1000,'new_all100_correct':new_correct,
                             'new_current10_correct':within_correct,'competition_errors':competition,
                             'old_n':9000,'old_all100_correct':old_correct,'all_n':10000,'all100_correct':total_correct,
                             'new_all100_pct':new_correct/10,'new_current10_pct':within_correct/10,
                             'competition_loss_pp':competition/10,'old_all100_pct':old_correct/90,
                             'all100_pct':total_correct/100})
    for row in rows:
        common=next(r for r in rows if r['seed']==row['seed'] and r['model']=='common_sgd')
        row['common_minus_usual_pp']=common['new_all100_pct']-row['new_all100_pct']
    methods={name:{metric:describe([r[metric] for r in rows if r['model']==name]) for metric in METRICS} for name in MODELS}
    differences={metric:describe([next(r for r in rows if r['seed']==seed and r['model']=='current_mask')[metric]-
                                 next(r for r in rows if r['seed']==seed and r['model']=='usual_er')[metric]
                                 for seed in range(3101,3106)]) for metric in METRICS}
    comp=differences['competition_loss_pp'];new=differences['new_all100_pct']
    tests={'competition_mean_positive':comp['mean']>0,'competition_strict_majority_positive':comp['positive_seeds']>=3,
           'usual_new_mean_negative':new['mean']<0,'usual_new_strict_majority_negative':new['negative_seeds']>=3}
    tests['both_necessary_directions_supported']=all(tests.values())
    summary={'independent_unit':'paired seed','n_seeds':5,'task':9,'unit':'percentage points except accuracy percentages',
             'methods':methods,'current_mask_minus_usual_er':differences,'prespecified_directional_test':tests}
    return rows,summary


def main():
    rows,summary=calculate()
    retained=ROOT/'original_statistics.json'
    if retained.exists():assert json.loads(retained.read_text())==summary
    original_counts=ROOT/'counts.csv'
    if original_counts.exists():
        expected=list(csv.DictReader(original_counts.open()))
        assert len(expected)==15
        for a,b in zip(expected,rows):
            for k in ('seed','new_n','new_all100_correct','new_current10_correct','competition_errors',
                      'old_n','old_all100_correct','all_n','all100_correct'):assert int(a[k])==b[k]
    with (ROOT/'recomputed_counts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'recomputed_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','models':15,'prespecified_test':summary['prespecified_directional_test'],
                      'paired_effects':summary['current_mask_minus_usual_er']}))


if __name__=='__main__':main()
