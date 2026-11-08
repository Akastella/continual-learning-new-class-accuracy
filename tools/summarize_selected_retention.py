"""Report each earlier task and retention shortfalls on selected models."""
from pathlib import Path
import argparse, csv, json
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,default=Path('evidence/score_frontier'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for r in csv.DictReader((a.data/'test_scores.csv').open()):
    name=f"{r['dataset']}_s{r['seed']}_{r['candidate']}.npz"
    path=a.data/'diagnostics/test_logits'/name
    if not path.exists():path=a.data/'test_logits'/name
    with np.load(path,allow_pickle=False) as d:
        scores=d['logits'].astype(float);labels=d['labels']
    scores[:,8:]+=float(r['b']);pred=scores.argmax(1)
    tasks=[100*float(np.mean(pred[labels//2==t]==labels[labels//2==t])) for t in range(4)]
    q=int(r['q'])
    rows.append({'dataset':r['dataset'],'seed':int(r['seed']),'q':q,'policy':r['policy'],
                 'candidate':r['candidate'],'pooled_old':float(np.mean(tasks)),
                 **{f'old_task_{t}':v for t,v in enumerate(tasks)},
                 'minimum_old_task':min(tasks),'pooled_shortfall':max(0,q-float(np.mean(tasks)))})
with (a.output/'selected_old_tasks.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for dataset in ['cifar101','cifar102']:
    for q in [10,20,30]:
        for policy in ['raw30','offset_usual','offset30']:
            rr=[r for r in rows if r['dataset']==dataset and r['q']==q and r['policy']==policy]
            v=np.array([r['minimum_old_task'] for r in rr]);short=np.array([r['pooled_shortfall'] for r in rr])
            summary.append(dict(dataset=dataset,q=q,policy=policy,n=len(rr),minimum_old_task_mean=float(v.mean()),
                minimum_old_task_sd=float(v.std(ddof=1)),all_old_tasks_meet_q=int((v>=q).sum()),
                mean_pooled_shortfall=float(short.mean()),maximum_pooled_shortfall=float(short.max())))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f'Reported all four old tasks for {len(rows)} selected models')
