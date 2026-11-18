"""Tables for the recorded score-adjustment comparison."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'new_experiments/score_frontier/summary.csv').open()))
names={'raw26':'26 training candidates','raw28':'28 training candidates','raw30':'30 training candidates',
       'offset_usual':'2 usual endpoints + offset','offset30':'30 candidates + offset'}
names['offset30_margin5']='30 + offset, 5-point margin'
def write(name, records, full=False):
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Choice & Old & Current & Violations\\',r'\midrule']
    last=None
    for r in records:
        group=(r['dataset'],r['q'])
        if group!=last:
            d='CIFAR-10.1' if r['dataset']=='cifar101' else 'CIFAR-10.2'
            lines += [r'\multicolumn{4}{l}{'+d+', $q='+r['q']+r'\%$}\\']
            last=group
        f=lambda k:f"${float(r[k+'_mean']):.2f}\\pm{float(r[k+'_sd']):.2f}$"
        lines += [f"{names[r['policy']]} & {f('old')} & {f('current')} & {r['violations']}/10"+r'\\']
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')
target_rows=[]
margin=json.loads((ROOT/'new_experiments/score_frontier/fixed_margin/summary.json').read_text())
for dataset in ['cifar101','cifar102']:
    target_rows += [r for r in rows if r['dataset']==dataset and r['q']=='30' and r['policy'] in ['raw30','offset_usual','offset30']]
    item=next(r for r in margin if r['dataset']==dataset and r['policy']=='offset30')
    assert item['available']==10
    row={'dataset':dataset,'q':'30','policy':'offset30_margin5','violations':item['violations']}
    row.update({k+'_'+stat:item[k][field] for k in ['old','current'] for stat,field in [('mean','mean'),('sd','sample_sd')]})
    target_rows.append(row)
write('score_target30.tex',target_rows)
for dataset in ['cifar101','cifar102']:
    write('score_'+dataset+'_all.tex',[r for r in rows if r['dataset']==dataset])
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Choice & Available & Old & Current & Violations\\',r'\midrule']
for dataset,title in [('cifar101','CIFAR-10.1'),('cifar102','CIFAR-10.2')]:
    lines.append(r'\multicolumn{5}{l}{'+title+r'}\\')
    for r in margin:
        if r['dataset']!=dataset:continue
        f=lambda k:f"${r[k]['mean']:.2f}\\pm{r[k]['sample_sd']:.2f}$"
        lines.append(f"{names[r['policy']]} & {r['available']}/10 & {f('old')} & {f('current')} & {r['violations']}/{r['available']}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(ROOT/'tables/fixed_margin.tex').write_text('\n'.join(lines)+'\n')

objectives = json.loads(
    (ROOT / "new_experiments/validation_objective/run2/summary.json").read_text(encoding="utf-8")
)
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Test source & \shortstack{Validation\\old (\%)} & $\Delta$New & $\Delta$Old & $\Delta$Overall & Misses\\",
         r"\midrule"]
for item in objectives:
    source = {"cifar101": "CIFAR-10.1", "cifar102": "CIFAR-10.2"}[item["dataset"]]
    paired = item["paired_new_minus_overall_objective"]
    effects = [
        "$" + f"{row['mean']:+.2f}\\pm{row['sample_sd']:.2f}$"
        for row in (paired[metric + "_difference_pp"] for metric in ("new", "old", "all"))
    ]
    misses = [str(item["objectives"][name]["below30_count"])
              for name in ("new_accuracy", "overall_accuracy")]
    lines.append(" & ".join([source, str(item["validation_threshold_pct"]),
                            *effects, "/".join(misses)]) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "tables/validation_objective.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print('Generated score-adjustment tables from summary.csv')
