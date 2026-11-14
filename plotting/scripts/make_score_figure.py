"""Plot existing decisions and their validation score tradeoff."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'new_experiments/score_frontier'
read=lambda name:list(csv.DictReader((src/name).open()))
metrics={(int(r['seed']),r['candidate']):r for r in read('all_candidate_group_metrics.csv') if r['dataset']=='validation'}
val={(int(r['seed']),r['candidate']):r for r in read('validation_candidates.csv') if r['q']=='30'}
curves={}
for r in read('complete_frontier.csv'):
    if r['dataset']=='validation' and r['candidate'] in ['er_standard','er_current_only_step_0628']:
        curves.setdefault((int(r['seed']),r['candidate']),[]).append((float(r['old']),float(r['current'])))
seeds=list(range(201,211)); names=['er_standard','sgd_standard','er_current_only_step_0628']
shared=[s for s in seeds if all(float(metrics[s,n]['old_within'])>=30 for n in ['er_standard','er_current_only_step_0628'])]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.linewidth':.7})
fig=plt.figure(figsize=(4.72,4.85))
ax=fig.add_axes([.37,.62,.59,.25])
for i,n in enumerate(names):
    y=2-i; jitter=np.linspace(-.06,.06,10)
    raw=[float(val[s,n]['raw_old']) for s in seeds]
    upper=[float(metrics[s,n]['old_within']) for s in seeds]
    ax.scatter(raw,y-.11+jitter,marker='o',s=13,c='#454545',alpha=.5,label='All classes' if i==0 else None)
    ax.scatter(upper,y+.11+jitter,marker='^',s=17,c='#0072B2',alpha=.6,label='Old classes only' if i==0 else None)
ax.set_yticks([2,1,0],['Usual ER','Usual SGD','ER after training\non new data only'],fontsize=8.4)
ax.set_xlim(-1,45);ax.set_ylim(-.4,2.45);ax.set_xticks([0,15,30,45]);ax.set_xlabel('Old-class accuracy (%)')
ax.axvline(30,ls=':',c='#666',lw=.8);ax.grid(axis='x',alpha=.15)
ax.legend(loc='lower left',bbox_to_anchor=(-.61,1.05),ncol=2,frameon=False,fontsize=8)
fig.text(.04,.96,'A   Old distinctions remain despite zero old accuracy',weight='bold',fontsize=8.7)
ax=fig.add_axes([.16,.13,.80,.32]); q=np.linspace(0,30,121)
records=[]
for n,color,style,marker,label in [('er_standard','#0072B2','-','o','Usual ER'),
 ('er_current_only_step_0628','#D55E00','--','s','ER after new-data-only training')]:
    arr=np.array([[max(v for old,v in curves[s,n] if old+1e-10>=target) for target in q] for s in shared])
    mean=arr.mean(0)
    ax.plot(q,mean,c=color,ls=style,lw=1.3,marker=marker,markevery=20,ms=4,label=label)
    records += [{'candidate':n,'target':float(x),'mean_current':float(y),'n':len(shared)} for x,y in zip(q,mean)]
ax.set_xlim(0,30);ax.set_ylim(0,100);ax.set_xticks([0,10,20,30]);ax.set_yticks([0,25,50,75,100]);ax.grid(alpha=.15)
ax.set_xlabel('Required old-class accuracy on validation (%)');ax.set_ylabel('New-class accuracy (%)')
ax.legend(loc='lower left',frameon=False,fontsize=7.6)
fig.text(.04,.52,'B   Remaining distinctions can require a large tradeoff',weight='bold',fontsize=9.2)
fig.text(.55,.025,f'Same {len(shared)} paired histories; validation images only',ha='center',fontsize=8)
for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'figures/score_tradeoff.{ext}',dpi=300,facecolor='white')
plt.close(fig)
Image.open(ROOT/'figures/score_tradeoff.png').convert('L').save(ROOT/'figures/score_tradeoff_gray.png')
(ROOT/'figures/score_tradeoff_data.json').write_text(json.dumps({'shared_seeds':shared,'curve':records},indent=2))
print('Common histories',shared)
