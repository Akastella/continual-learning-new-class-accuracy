"""Summarize all-candidate upper bounds and export exact score tradeoff curves."""
from pathlib import Path
import argparse,csv,json,importlib.util
import numpy as np

def csvwrite(p,rs):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    path=Path(__file__).with_name('recompute_score_frontier.py')
    s=importlib.util.spec_from_file_location('recompute',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    rows=[];curves=[]
    with (a.data/'complete_frontier.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','seed','candidate','b','old','current','all','old_correct','current_correct','old_n','current_n'])
        writer.writeheader()
        for dataset in ['validation','cifar101','cifar102']:
            for seed in range(201,211):
                for n in m.all_names():
                    if dataset=='validation':z,y=m.loadval(a.artifact,seed,n)
                    else:
                        with np.load(a.data/f'diagnostics/test_logits/{dataset}_s{seed}_{n}.npz',allow_pickle=False) as d:z,y=d['logits'].astype(float),d['labels'].copy()
                    oldpred=z[:,:8].argmax(1);newpred=z[:,8:].argmax(1)+8
                    margin=z[:,8:].max(1)-z[:,:8].max(1)
                    od=np.sort(margin[(y<8)&(oldpred==y)]);nd=np.sort(margin[(y>=8)&(newpred==y)])
                    oldn=int(sum(y<8));newn=int(sum(y>=8))
                    auc=float(np.mean((nd[:,None]>od[None,:])+0.5*(nd[:,None]==od[None,:]))) if len(od) and len(nd) else None
                    rows.append(dict(dataset=dataset,seed=seed,candidate=n,old_within=100*len(od)/oldn,current_within=100*len(nd)/newn,conditional_margin_auc=auc))
                    bs=sorted(set((-od).tolist())|{0.,float(-margin.max()-1),float(-margin.min()+1)})
                    for b in bs:
                        oc=int(np.searchsorted(od,-b,side='right'));nc=len(nd)-int(np.searchsorted(nd,-b,side='right'))
                        writer.writerow(dict(dataset=dataset,seed=seed,candidate=n,b=b,old=100*oc/oldn,current=100*nc/newn,all=100*(oc+nc)/len(y),old_correct=oc,current_correct=nc,old_n=oldn,current_n=newn))
    csvwrite(a.data/'all_candidate_group_metrics.csv',rows)
    summaries=[]
    for n in m.all_names():
        for dataset in ['cifar101','cifar102']:
            vs=sorted([r for r in rows if r['candidate']==n and r['dataset']=='validation'],key=lambda x:x['seed'])
            ts=sorted([r for r in rows if r['candidate']==n and r['dataset']==dataset],key=lambda x:x['seed'])
            vals=[r['old_within'] for r in vs];tests=[r['old_within'] for r in ts]
            r=dict(candidate=n,dataset=dataset,validation_old_within_mean=float(np.mean(vals)),test_old_within_mean=float(np.mean(tests)),test_old_within_min=min(tests),test_old_within_max=max(tests),test_current_within_mean=float(np.mean([r['current_within'] for r in ts])),validation_margin_auc=float(np.mean([r['conditional_margin_auc'] for r in vs])),test_margin_auc=float(np.mean([r['conditional_margin_auc'] for r in ts])))
            for q in [20,30]:
                ve=np.array(vals)<q;te=np.array(tests)<q
                r.update({f'q{q}_validation_excluded':int(ve.sum()),f'q{q}_test_excluded':int(te.sum()),f'q{q}_excluded_still_infeasible':int((ve&te).sum()),f'q{q}_excluded_but_test_feasible':int((ve&~te).sum())})
            summaries.append(r)
    csvwrite(a.data/'candidate_upper_bound_summary.csv',summaries)
    print(json.dumps(dict(group_metric_rows=len(rows),candidate_summary_rows=len(summaries),curve_bytes=(a.data/'complete_frontier.csv').stat().st_size)))
if __name__=='__main__':main()
