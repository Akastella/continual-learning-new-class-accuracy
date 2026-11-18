"""Regenerate editable vector figures from the existing study outputs.

Run with --project pointing to the original research checkout.
Every value comes from saved results or their existing generated tables.
"""
from pathlib import Path
import argparse, csv, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]/'figure_inputs')
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
root=a.project
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2})
colors=['#0072B2','#D55E00','#009E73']
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(a.output/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

h=json.loads((root/'results/r9_head_diagnostic/analysis.json').read_text())
units={(r['seed'],r['method'],r['view']):r for r in h['seed_units']}
seeds=list(range(201,211)); data=[]
for s in seeds:
    e=units[s,'er','all']; g=units[s,'sgd','all']; m=units[s,'er','local']; gm=units[s,'sgd','local']
    data.append([e['Y']-g['Y'],e['P']-g['P'],m['Y']-gm['Y']])
data=np.array(data)
with (a.output/'primary_data.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['seed','usual_update_ER_minus_SGD','current_only_ER_minus_SGD','current_classes_ER_minus_SGD'])
    w.writerows([[s,*v] for s,v in zip(seeds,data)])


# Longer training: seed means and SD already computed by the original analysis.
txt=(root/'paper_rebuttal/generated/four_task_budget_table.tex').read_text()
rows=[]
for line in txt.splitlines():
    if re.match(r'^(4|8|16) &',line):
        nums=re.findall(r'[+-]?\d+(?:\.\d+)?',line);rows.append([float(x) for x in nums])
arr=np.array(rows)
fig,axes=plt.subplots(1,2,figsize=(3.48,2.58));fig.subplots_adjust(left=.17,right=.98,bottom=.24,top=.76,wspace=.45)
for j,(lab,col,mark) in enumerate(zip(['Usual update','Current-only','Current-class prediction'],colors,['o','s','^'])):
    ax=axes[0 if j==0 else 1]
    ax.errorbar(arr[:,0],arr[:,1+2*j],yerr=arr[:,2+2*j],label=lab,color=col,marker=mark,ms=4,capsize=2)
for ax in axes:
    ax.axhline(0,color='#555',lw=.6);ax.set_xticks([4,8,16]);ax.grid(alpha=.12);ax.tick_params(labelsize=7.5)
axes[0].set_ylabel('ER minus SGD (pp)');axes[0].set_title('a  Usual update',fontsize=8.5)
axes[1].set_title('b  Other comparisons',fontsize=8.5)
fig.text(.57,.10,'Current-task epochs',ha='center',fontsize=8.5)
fig.legend(*axes[1].get_legend_handles_labels(),fontsize=8,frameon=False,loc='upper right',bbox_to_anchor=(.99,.975),borderaxespad=0)
save(fig,'longer_training')

# Final-task same-model retention, original and held-constant memory.
control=json.loads((root/'control_statistics.json').read_text())['summary']
ret=[]
for epoch in [8,16]:
    for mode in ['fixed','evolving']:
        record=control[f'epoch{epoch}_{mode}']
        ret.append([epoch,record['current']['mean'],record['current']['sd'],record['old']['mean'],record['old']['sd'],record['all']['mean'],record['all']['sd']])
with (a.output/'retention_points.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['epoch','memory','current_mean','old_mean','all_mean'])
    w.writerows([[v[0],('unchanged' if i%2==0 else 'evolving'),v[1],v[3],v[5]] for i,v in enumerate(ret)])
assert all(19 <= v[3] <= 41 and 60 <= v[1] <= 89 for v in ret), 'Every reported point must be visible'
fig,ax=plt.subplots(figsize=(3.48,2.58));fig.subplots_adjust(left=.17,right=.96,bottom=.21,top=.90)
ax.scatter([37.551],[64.615],s=40,c='black',marker='D',label='Before continuation (4 epochs)')
for fixed,col,marker in [(True,colors[0],'s'),(False,colors[1],'o')]:
    vals=[r for i,r in enumerate(ret) if (i%2==0)==fixed]
    xs=[37.551]+[r[3] for r in vals];ys=[64.615]+[r[1] for r in vals]
    ax.plot(xs,ys,color=col,marker=marker,ms=4,label='Memory contents unchanged' if fixed else 'Memory updated normally')
    for r in vals:ax.annotate(str(int(r[0])),(r[3],r[1]),xytext=(5,-2),textcoords='offset points',fontsize=8,color=col)
ax.set_xlabel('Old-class accuracy (%)');ax.set_ylabel('New-class accuracy (%)');ax.set_xlim(19,41);ax.set_ylim(60,89);ax.grid(alpha=.15)
ax.legend(fontsize=8,frameon=False,loc='lower left');save(fig,'retention_continuation')

profile={'unit':'seed; tasks averaged within seed','primary_n':10,'primary_missing':int(np.isnan(data).sum()),
 'primary_means':data.mean(0).tolist(),'primary_sample_SD':data.std(0,ddof=1).tolist(),
 'primary_min':data.min(0).tolist(),'primary_max':data.max(0).tolist(),
 'choice':'show every paired seed; use lines only for ordered training budgets; SD bars descriptive',
 'source_files':['results/r9_head_diagnostic/analysis.json','paper_rebuttal/generated/four_task_budget_table.tex','control_statistics.json']}
(a.output/'data_profile.json').write_text(json.dumps(profile,indent=2))
print(json.dumps(profile,indent=2))
