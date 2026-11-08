"""Descriptive seed summaries only; no hypothesis test or confidence interval."""
import numpy as np

def describe(values):
    return {'n':len(values),'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}

def summarize(raws):
    units=[];counts=[];retention=[]
    for (method,seed),raw in sorted(raws.items()):
        for epoch in [1,2,4]:
            ts=[]
            for t,row in enumerate(raw['tasks']):
                values={}
                for branch,key in [('fresh','F'),('probe','p'),('actual','a')]:
                    ev=row[branch]['trace'][epoch]['evaluation'];values[key]=100*ev['correct']/ev['n']
                    counts.append({'method':method,'seed':seed,'task':t,'epoch':epoch,'branch':branch,'correct':ev['correct'],'n':ev['n']})
                values.update(A=values['F']-values['a'],H=values['F']-values['p'],C=values['p']-values['a'])
                ts.append({'task':t,**values})
            units.append({'method':method,'seed':seed,'epoch':epoch,**{k:float(np.mean([x[k] for x in ts[1:]])) for k in ['F','p','a','A','H','C']},'per_task':ts})
        for t,row in enumerate(raw['tasks']):
            es=row['post_task_evaluation'];old=sum(v['correct'] for v in es[:t]);seen=sum(v['correct'] for v in es[:t+1]);allc=sum(v['correct'] for v in es)
            retention.append({'method':method,'seed':seed,'task':t,'epoch':4,'old_pct':100*old/(2000*t) if t else None,'seen_pct':100*seen/(2000*(t+1)),'all10_pct':allc/100,'correct_per_task':[v['correct'] for v in es],'n_per_task':2000})
    groups=[];paired=[]
    for epoch in [1,2,4]:
        for method in ['sgd','er','derpp']:
            rows=[x for x in units if x['epoch']==epoch and x['method']==method]
            if sorted(x['seed'] for x in rows)!=list(range(201,211)):raise ValueError('complete paired seed matrix required')
            groups.append({'method':method,'epoch':epoch,'metrics':{k:describe([r[k] for r in rows]) for k in ['F','p','a','A','H','C']}})
        for baseline in ['sgd','er']:
            vals=[]
            for seed in range(201,211):
                a=next(x for x in units if x['method']=='derpp' and x['epoch']==epoch and x['seed']==seed);b=next(x for x in units if x['method']==baseline and x['epoch']==epoch and x['seed']==seed)
                if a['F']!=b['F']:raise ValueError('paired fresh mismatch')
                vals.append({'seed':seed,**{k:a[k]-b[k] for k in ['F','p','a','A','H','C']}})
            paired.append({'contrast':'derpp_minus_'+baseline,'epoch':epoch,'seed_values':vals,'metrics':{k:describe([v[k] for v in vals]) for k in ['F','p','a','A','H','C']}})
    return {'seed_units':units,'current_integer_counts':counts,'groups':groups,'paired':paired,'actual_full_history_epoch4':retention,'scope':'All means equally weight tasks1..4 within seed then seeds; task0 also disclosed. Old/all scores are epoch4 actual checkpoints only. Descriptive SD, no CI or p-value.'}
