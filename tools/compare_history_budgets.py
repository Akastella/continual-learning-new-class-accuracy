"""Compare current-task and whole-history training on exactly the same seeds."""
from pathlib import Path
import argparse,csv,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--recomputed-root',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
rows=list(csv.DictReader((a.recomputed_root/'ijcnn/paired_seed_values.csv').open()))
full=json.loads((a.recomputed_root/'full_history16/paired_summary.json').read_text())
result={};changes={}
for name,branch,metric in [('usual','actual','current'),('current_only','probe','current'),('current_classes','actual','current_classes')]:
    values={}
    for epoch in [4,16]:
        values[f'4/{epoch}']=[float(next(r['value'] for r in rows if r['stage']=='budget' and r['tasks']=='mean_noninitial' and r['branch']==branch and r['metric']==metric and r['seed']==str(s) and r['epoch']==str(epoch))) for s in range(201,206)]
    values['16/16']=next(r['values'] for r in full if r['branch']==('usual' if branch=='actual' else 'current_only') and r['metric']==metric)
    result[name]={k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),values=v) for k,v in values.items()}
    for left,right in [('4/16','4/4'),('16/16','4/16')]:
        v=np.array(values[left])-np.array(values[right])
        changes[name+' '+left+' minus '+right]=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),values=v.tolist())
a.output.write_text(json.dumps(dict(seeds=list(range(201,206)),settings='prior-task/current-task epochs',comparisons=result,paired_changes=changes),indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
