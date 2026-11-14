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
fig=plt.figure(figsize=(7.16,3.42));gs=fig.add_gridspec(1,2,width_ratios=[.9,1.15],left=.025,right=.98,bottom=.21,top=.89,wspace=.52)
ax=fig.add_subplot(gs[0]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(0,1.06,'A   Compare learning from each history',weight='bold',fontsize=10)
def box(x,y,w,hh,text,color='#f3f5f7'):
    ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=0.012,rounding_size=.015',facecolor=color,edgecolor='#bac2c8',linewidth=.6))
    ax.text(x+w/2,y+hh/2,text,ha='center',va='center',fontsize=8)
box(.16,.79,.67,.15,"Each method's own\npre-task model")
box(.01,.40,.46,.23,'Usual update\nNew + stored data\n(for replay)','#e7f1f8')
box(.54,.40,.44,.23,'Current-only SGD\nNew examples only','#fff0e7')
for x in [.24,.76]:ax.annotate('',(x,.645),(.49,.78),arrowprops={'arrowstyle':'->','lw':1,'color':'#687682'})
box(.01,.08,.46,.20,'Prediction candidates:\nall or current classes')
box(.54,.08,.44,.20,'Prediction candidates:\nall classes')
for x in [.24,.76]:ax.annotate('',(x,.29),(x,.39),arrowprops={'arrowstyle':'->','lw':1,'color':'#687682'})
ax.text(.5,-.08,'Test images: new classes.\nMeasure old accuracy separately.',ha='center',fontsize=8)
ax=fig.add_subplot(gs[1]);ax.set_title('B   ER minus SGD on new classes',loc='left',fontweight='bold',pad=12)
labels=['Usual training\nall 10 classes','Train on new data only\nall 10 classes','Usual model, unchanged\ncurrent 2 classes']
for j in range(3):
    y=2-j; jit=np.linspace(-.13,.13,10)
    ax.scatter(data[:,j],y+jit,color=colors[j],s=16,alpha=.72,linewidths=0)
    ax.scatter(data[:,j].mean(),y,marker='D',s=37,color=colors[j],edgecolor='white',linewidth=.6,zorder=5)
    ax.annotate(f'{data[:,j].mean():+.2f}',(data[:,j].mean(),y),xytext=(0,15),textcoords='offset points',ha='center',weight='bold',color=colors[j])
ax.axvline(0,color='#64707b',lw=.7,linestyle='--');ax.set_yticks([2,1,0],labels,fontsize=8.5)
ax.set_xlim(-31,8);ax.set_ylim(-.4,2.55);ax.set_xticks([-30,-20,-10,0]);ax.set_xlabel('Accuracy difference (percentage points)')
ax.grid(axis='x',alpha=.15)
fig.text(.72,.035,'Old accuracy: usual ER 43.56%, SGD 0%.\nCurrent-only: 0% for both histories.',ha='center',fontsize=8)
save(fig,'overview')

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
axes[0].set_ylabel('ER minus SGD (pp)');axes[0].set_title('Usual update',fontsize=8.5)
axes[1].set_title('Other comparisons',fontsize=8.5)
fig.text(.57,.10,'Current-task epochs',ha='center',fontsize=8.5)
fig.legend(*axes[1].get_legend_handles_labels(),fontsize=7.1,frameon=False,loc='upper right',bbox_to_anchor=(.99,.975),borderaxespad=0)
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
ax.legend(fontsize=6.8,frameon=False,loc='lower left');save(fig,'retention_continuation')

profile={'unit':'seed; tasks averaged within seed','primary_n':10,'primary_missing':int(np.isnan(data).sum()),
 'primary_means':data.mean(0).tolist(),'primary_sample_SD':data.std(0,ddof=1).tolist(),
 'primary_min':data.min(0).tolist(),'primary_max':data.max(0).tolist(),
 'choice':'show every paired seed; use lines only for ordered training budgets; SD bars descriptive',
 'source_files':['results/r9_head_diagnostic/analysis.json','paper_rebuttal/generated/four_task_budget_table.tex','control_statistics.json']}
(a.output/'data_profile.json').write_text(json.dumps(profile,indent=2))
print(json.dumps(profile,indent=2))
