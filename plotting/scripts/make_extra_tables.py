"""Produce supplementary tables directly from recomputed study statistics."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'tables'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def table(name,headers,rows,align=None):
    lines=[r'\begin{tabular}{'+(align or 'r'*len(headers))+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    lines+=[' & '.join(map(str,r))+r'\\' for r in rows]
    lines+=[r'\bottomrule',r'\end{tabular}']
    (out/name).write_text('\n'.join(lines)+'\n',encoding='utf8')

d=read(ROOT/'figure_inputs/cifar100_statistics.json')['paired_seed_contrasts']
rows=[]
for i,t in enumerate(range(6,10)):
    vals=[np.array([r[k][i] for r in d]) for k in ['d_actual_per_task','d_probe_per_task','d_masked_per_task']]
    rows.append([t]+[f'${v.mean():+.2f}\\pm{v.std(ddof=1):.2f}$' for v in vals]+['/'.join(str(int((v>0).sum())) for v in vals)])
table('cifar100_tasks.tex',['Task','Usual','Current-only','Current-class','Positive seeds'],rows)

controls=read(ROOT/'figure_inputs/control_statistics.json')['summary']
rows=[[4,'Both','64.615','37.551','42.964']]
# The pre-continuation all-class value is computed from the same old/current pair.
rows[0][-1]=f'{(.2*64.615+.8*37.55125):.3f}'
for e in [8,16]:
    for mode,label in [('fixed','Unchanged'),('evolving','Updated')]:
        r=controls[f'epoch{e}_{mode}']
        rows.append([e,label]+[f"${r[k]['mean']:.3f}\\pm{r[k]['sd']:.3f}$" for k in ['current','old','all']])
table('memory_points.tex',['Epochs','Memory','Current','Old','All'],rows,align='rlrrr')

h=read(ROOT/'figure_inputs/results/r9_head_diagnostic/analysis.json')
u={(r['seed'],r['method'],r['view']):r for r in h['seed_units']}
rows=[]
for seed in range(201,206):
    rows.append(['4 epochs, all outputs',seed]+[f'{u[seed,"er",w][k]-u[seed,"sgd",w][k]:+.4f}' for k,w in [('Y','all'),('P','all'),('Y','local')]])
for name,label in [('full_history16','16 epochs, all outputs'),('seen_classes4','4 epochs, seen outputs')]:
    path=ROOT/f'figure_inputs/history_controls/{name}.json'
    if not path.exists():continue
    d=read(path)
    take=lambda b,m:next(r for r in d if r['branch']==b and r['metric']==m)
    vals=[take('usual','current')['values'],take('current_only','current')['values'],take('usual','current_classes')['values']]
    for i,seed in enumerate(range(201,206)):
        rows.append([label,seed]+[f'{v[i]:+.4f}' for v in vals])
table('history_control_seeds.tex',['Training','Seed','Usual','Current-only','Current-class'],rows,align='lrrrr')
d=read(ROOT/'figure_inputs/four_budgets.json')
rows=[[k]+[f"${d[m][k]['mean']:+.2f}\\pm{d[m][k]['sd']:.2f}$" for m in ['usual','current_only']]+[f"{d['current_only'][k]['positive']}/5"] for k in ['4/4','4/16','16/4','16/16']]
table('matched_budgets.tex',['Prior/current','Usual','Current-only','Positive seeds'],rows,align='lrrr')
rows=[[s]+[f"{d[m][k]['values'][i]:+.4f}" for m,k in [('usual','16/4'),('current_only','16/4')]] for i,s in enumerate(range(201,206))]
table('history16_current4_seeds.tex',['Seed','Usual difference','Current-only difference'],rows)
if (ROOT/'figure_inputs/validation_calibration.json').exists():
    d=read(ROOT/'figure_inputs/validation_calibration.json')['summary'];rows=[]
    for m in ['er','sgd']:
        for c in ['usual','adjusted']:
            rows.append([m.upper()+', '+c]+[f"${d[m+'_'+c][metric]['mean']:.2f}\\pm{d[m+'_'+c][metric]['sd']:.2f}$" for metric in ['old','current','all']])
    table('validation_calibration.tex',['Model','Old','Current','All'],rows,align='lrrr')
print('Generated supplementary tables from saved statistics.')
