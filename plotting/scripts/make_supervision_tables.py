"""Tables from the fixed-history supervision experiment and held-out score analysis."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'new_experiments/cross_group_supervision'
data=json.loads((SRC/'summary.json').read_text())
names={'usual_er':'Old only, full','current_mask':'Old only, current ten',
       'mixed_full':'Mixed, full','mixed_restricted':'Mixed, current ten',
       'old_blend05':r'Old only, 5\% full','common_sgd':'Current-only'}
models=list(names)
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
       r'Condition & \shortstack{New\\all 100} & \shortstack{New\\current ten} & \shortstack{Old\\all 100} & All 100\\',r'\midrule']
for name in models:
    cells=[f"${data['methods'][name][k]['mean']:.2f}\\pm{data['methods'][name][k]['sample_sd']:.2f}$"
           for k in ['new_all100_pct','new_current10_pct','old_all100_pct','all100_pct']]
    lines.append(names[name]+' & '+' & '.join(cells)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(ROOT/'tables/supervision_raw.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
summary=json.loads((SRC/'score_analysis/summary.json').read_text())
def offset_table(targets,filename):
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & Available & Old & New & Violations\\',r'\midrule']
    for q in targets:
        lines.append(r'\multicolumn{5}{l}{$q='+str(q)+r'\%$; validation requires '+str(q+3)+r'\%}\\')
        for name in models:
            r=next(r for r in summary if r['model']==name and r['target_pct']==q)
            f=lambda key: '$'+f"{r[key]['mean']:.2f}\\pm{r[key]['sample_sd']:.2f}"+'$' if r['available_seeds']>1 else (str(r[key]['mean']) if r['available_seeds'] else '--')
            violations=f"{r['test_violations']}/{r['available_seeds']}" if r['available_seeds'] else '--'
            lines.append(names[name]+f" & {r['available_seeds']}/5 & {f('old_pct')} & {f('new_pct')} & {violations}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables'/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')
offset_table([20],'supervision_offset20.tex')
offset_table([5,10,15],'supervision_offset_low.tex')
offset_table([20,25,30],'supervision_offset_high.tex')
lines=[r'\begin{tabular}{rrrrr}',r'\toprule',r'Seed & New interaction & Competition interaction & Recovery above half & Old above ER\\',r'\midrule']
for i,seed in enumerate(data['seeds']):
    values=[data['restriction_interaction'][k]['values'][i] for k in ['new_all100_pct','competition_loss_pp']]
    values += [data['H2'][k]['values'][i] for k in ['new_minus_half_original_er_pp','old_minus_original_er_pp']]
    lines.append(str(seed)+' & '+' & '.join(f'${v:+.2f}$' for v in values)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(ROOT/'tables/supervision_seed_tests.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Generated supervision and held-out offset tables.')
