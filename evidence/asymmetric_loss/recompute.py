"""NumPy-only verification and seed aggregation; run from any working directory."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('sgd','er_concat','er_equal_sum','er_ace_style')


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1))}


def main():
    count_rows=[];tasks=[]
    for path in sorted(ROOT.glob('s*_t*_*.npz')):
        with np.load(path,allow_pickle=False) as d:
            seed=int(d['seed']);method=str(d['method']);tid=int(d['training_task']);endpoint=str(d['endpoint'])
            y=d['mapped_labels'];pred=d['predictions_all10'];pair=d['predictions_task2'];et=d['evaluation_task']
            assert y.shape==pred.shape==pair.shape==et.shape==d['official_test_indices'].shape
            assert len(np.unique(d['official_test_indices']))==len(y)
            assert np.array_equal(np.sort(np.unique(et)),np.arange(tid+1))
            assert np.all((pair//2)==(y//2))
            for j in range(tid+1):
                sel=et==j;n=int(sel.sum());assert n==2000
                count_rows.append({'seed':seed,'method':method,'task':tid,'endpoint':endpoint,'evaluation_task':j,
                                   'n':n,'all10_correct':int((pred[sel]==y[sel]).sum()),
                                   'current2_correct':int((pair[sel]==y[sel]).sum())})
            new=et==tid;old=et<tid
            tasks.append({'seed':seed,'method':method,'task':tid,'endpoint':endpoint,
                          'new_all10_pct':100*float((pred[new]==y[new]).mean()),
                          'new_current2_pct':100*float((pair[new]==y[new]).mean()),
                          'old_all10_pct':100*float((pred[old]==y[old]).mean()) if tid else None})
    expected=list(csv.DictReader((ROOT/'counts.csv').open()))
    assert len(count_rows)==len(expected)==600
    key=lambda r:(int(r['seed']),r['method'],int(r['task']),r['endpoint'],int(r['evaluation_task']))
    observed={key(r):r for r in count_rows}
    for r in expected:assert all(int(r[k])==observed[key(r)][k] for k in ('n','all10_correct','current2_correct'))
    seeds=[]
    for seed in range(201,206):
        for method in METHODS:
            row={'seed':seed,'method':method}
            for endpoint,prefix in [('usual','usual'),('current_only','common_sgd')]:
                selected=[t for t in tasks if t['seed']==seed and t['method']==method and t['endpoint']==endpoint and t['task']>0]
                assert len(selected)==4
                row[prefix+'_all10_pct']=float(np.mean([t['new_all10_pct'] for t in selected]))
                row[prefix+'_current2_pct']=float(np.mean([t['new_current2_pct'] for t in selected]))
                row[prefix+'_old_all10_pct']=float(np.mean([t['old_all10_pct'] for t in selected]))
            row['competition_loss_pp']=row['usual_current2_pct']-row['usual_all10_pct']
            row['common_minus_usual_pp']=row['common_sgd_all10_pct']-row['usual_all10_pct']
            seeds.append(row)
    with (ROOT/'recomputed_seed_values.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(seeds[0]));w.writeheader();w.writerows(seeds)
    summary={m:{k:stats([r[k] for r in seeds if r['method']==m]) for k in seeds[0] if k not in ('seed','method')} for m in METHODS}
    original=json.loads((ROOT/'original_statistics.json').read_text())
    for method in METHODS:
        for k in ('usual_all10_pct','common_sgd_all10_pct','usual_current2_pct','usual_old_all10_pct','competition_loss_pp','common_minus_usual_pp'):
            for field in ('mean','sample_sd'):assert abs(summary[method][k][field]-original['methods'][method][k][field])<1e-10
    (ROOT/'recomputed_summary.json').write_text(json.dumps({'status':'passed','npz_files':200,'integer_count_rows':600,
        'independent_unit':'seed','tasks_averaged':[1,2,3,4],'methods':summary},indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','old_accuracy':{m:{endpoint:summary[m][endpoint+'_old_all10_pct'] for endpoint in ('usual','common_sgd')} for m in METHODS}}))


if __name__=='__main__':main()
