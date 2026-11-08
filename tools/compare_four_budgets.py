"""Read intermediate evaluations to complete the paired budget comparison."""
from pathlib import Path
import argparse,json,csv
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--history16',type=Path,required=True)
p.add_argument('--matched-three',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
old=json.loads(a.matched_three.read_text())['comparisons']; rows=[]; sources={}
for seed in range(201,206):
    histories={m:json.loads((a.history16/f'{m}_s{seed}/result.json').read_text()) for m in ['sgd','er']}
    for m,d in histories.items():
        cfg=d['config']; assert cfg['epochs']==16 and cfg['lr']==.01 and cfg.get('momentum',0)==0
    sources[str(seed)]={m:[t['task'] for t in d['tasks']] for m,d in histories.items()}
    for current in [4,16]:
        for branch,label in [('actual','usual'),('probe','current_only')]:
            v={}
            for m,d in histories.items():
                v[m]=np.mean([next(r for r in t[branch]['trace'] if r['epoch']==current)['evaluation']['accuracy_pct']
                              for t in d['tasks'] if t['task'] in [1,2,3,4]])
            rows.append(dict(seed=seed,prior=16,current=current,measurement=label,sgd=float(v['sgd']),er=float(v['er']),difference=float(v['er']-v['sgd'])))
    for current in [4,16]:
        for label in ['usual','current_only']:
            value=old[label][f'4/{current}']['values'][seed-201]
            rows.append(dict(seed=seed,prior=4,current=current,measurement=label,sgd=None,er=None,difference=value))
def stats(v):
    v=np.asarray(v,float);return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),positive=int((v>0).sum()),negative=int((v<0).sum()),values=v.tolist())
summary={}
for label in ['usual','current_only']:
    lookup={(r['seed'],r['prior'],r['current']):r['difference'] for r in rows if r['measurement']==label}
    summary[label]={f'{h}/{k}':stats([lookup[s,h,k] for s in range(201,206)]) for h in [4,16] for k in [4,16]}
    summary[label]['history_effect_at4']=stats([lookup[s,16,4]-lookup[s,4,4] for s in range(201,206)])
    summary[label]['history_effect_at16']=stats([lookup[s,16,16]-lookup[s,4,16] for s in range(201,206)])
    summary[label]['interaction']=stats([(lookup[s,16,16]-lookup[s,16,4])-(lookup[s,4,16]-lookup[s,4,4]) for s in range(201,206)])
with (a.output/'seed_budgets.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for label,d in summary.items():print(label,{k:{x:v[x] for x in ['mean','sd','positive','negative']} for k,v in d.items()})
