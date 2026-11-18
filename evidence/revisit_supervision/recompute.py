"""NumPy-only recomputation of the fixed replay-supervision control."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = (7, 9)
SEEDS = tuple(range(3101, 3106))
METHODS = ('old_restricted', 'mixed_restricted', 'revisit_current10', 'usual_er')
PAIRS = {'control_minus_mixed': ('revisit_current10','mixed_restricted'),
         'control_minus_old': ('revisit_current10','old_restricted'),
         'mixed_minus_old': ('mixed_restricted','old_restricted'),
         'control_minus_er': ('revisit_current10','usual_er')}


def write_json(path, values):
    path.write_text(json.dumps(values,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(numerators, denominator):
    values = np.asarray(numerators,dtype=np.float64)/denominator
    return {'n':len(numerators),'values':values.tolist(),
        'mean':sum(numerators)/(len(numerators)*denominator) if len(numerators) else None,
        'sample_sd':float(values.std(ddof=1)) if len(values)>1 else None,
        'negative_seeds':sum(value<0 for value in numerators),
        'zero_seeds':sum(value==0 for value in numerators),
        'positive_seeds':sum(value>0 for value in numerators),
        'nonnegative_seeds':sum(value>=0 for value in numerators)}


def calibration_mask(original_labels):
    mask = np.zeros(10000,dtype=bool)
    for original_class in range(100):
        ids = np.flatnonzero(original_labels==original_class)
        assert len(ids)==100
        generator = np.random.Generator(np.random.PCG64(2026110800+original_class))
        mask[generator.permutation(ids)[:20]] = True
    return mask


def measure(data, subset, bias=0.):
    task = int(data['task'])
    old,seen = task*10,task*10+10
    labels = data['mapped_labels']
    old_rows = subset & (labels<old)
    new_rows = subset & (labels>=old) & (labels<seen)
    prediction = np.where(data['current_score'].astype(np.float64)+bias > data['old_score'].astype(np.float64),
                          data['current_prediction'],data['old_prediction'])
    correct = prediction==labels
    old_n,new_n = int(old_rows.sum()),int(new_rows.sum())
    old_correct,new_correct = int(correct[old_rows].sum()),int(correct[new_rows].sum())
    current_correct = int((data['current_prediction'][new_rows]==labels[new_rows]).sum())
    within_old = int((data['old_prediction'][old_rows]==labels[old_rows]).sum())
    row = {'old_n':old_n,'new_n':new_n,'all_n':old_n+new_n,
        'old_correct':old_correct,'new_correct':new_correct,'all_correct':old_correct+new_correct,
        'current10_correct':current_correct,'competition_lost_correct':current_correct-new_correct,
        'within_old_correct':within_old,'old_pct':100*old_correct/old_n,'new_pct':100*new_correct/new_n,
        'all_pct':100*(old_correct+new_correct)/(old_n+new_n),'current10_pct':100*current_correct/new_n,
        'competition_pp':100*(current_correct-new_correct)/new_n,'within_old_pct':100*within_old/old_n}
    for old_task in range(task):
        take = subset & (labels//10==old_task)
        row[f'old_task{old_task}_n'] = int(take.sum())
        row[f'old_task{old_task}_correct'] = int(correct[take].sum())
        row[f'old_task{old_task}_pct'] = 100*int(correct[take].sum())/int(take.sum())
    return row


def calculate():
    raw,unadjusted,adjusted = [],[],[]
    identities = []
    for task in TASKS:
        old,seen = task*10,task*10+10
        for seed in SEEDS:
            reference = None
            for method in METHODS:
                with np.load(ROOT/'reader_data'/f't{task}_s{seed}_{method}.npz',allow_pickle=False) as archive:
                    data = {key:archive[key].copy() for key in archive.files}
                labels = data['mapped_labels']
                assert int(data['task'])==task and int(data['seed'])==seed and str(data['method'])==method
                assert labels.shape==(10000,) and np.array_equal(data['official_test_indices'],np.arange(10000))
                assert np.array_equal(data['class_order'][labels],data['original_labels'])
                assert np.array_equal(np.bincount(labels,minlength=100),np.full(100,100))
                if reference is None:
                    reference = [data[key] for key in ('mapped_labels','original_labels','class_order')]
                else:
                    assert all(np.array_equal(data[key],value) for key,value in zip(('mapped_labels','original_labels','class_order'),reference))
                predicted = np.where(data['current_score']>data['old_score'],data['current_prediction'],data['old_prediction'])
                assert np.array_equal(predicted,data['predictions_seen'])
                seen_rows = labels<seen
                calibration = calibration_mask(data['original_labels']) & seen_rows
                test = seen_rows & ~calibration
                assert int(calibration.sum())==20*seen and int(test.sum())==80*seen
                assert int((calibration & (labels<old)).sum())==20*old
                assert int((test & (labels<old)).sum())==80*old
                assert np.array_equal(np.bincount(labels[calibration],minlength=100),np.r_[np.full(seen,20),np.zeros(100-seen)])
                assert np.array_equal(np.bincount(labels[test],minlength=100),np.r_[np.full(seen,80),np.zeros(100-seen)])
                raw.append({'task':task,'seed':seed,'method':method,**measure(data,seen_rows)})
                for part,mask in (('calibration',calibration),('test',test)):
                    unadjusted.append({'task':task,'seed':seed,'method':method,'images':part,**measure(data,mask)})
                correct_old = calibration & (labels<old) & (data['old_prediction']==labels)
                thresholds = data['old_score'][correct_old].astype(np.float64)-data['current_score'][correct_old].astype(np.float64)
                required = old*20*23//100
                assert required==(322 if task==7 else 414)
                available = len(thresholds)>=required
                row = {'task':task,'seed':seed,'method':method,'q':20,'margin_pp':3,
                    'required_old_correct':required,'available':available,'bias':None,'test_violation':None}
                if available:
                    bias = float(np.sort(thresholds)[-required])
                    assert int((thresholds>=bias).sum())>=required and int((thresholds>bias).sum())<required
                    cal_scores,test_scores = measure(data,calibration,bias),measure(data,test,bias)
                    assert cal_scores['old_correct']>=required
                    row.update(bias=bias,test_violation=test_scores['old_correct']*100 < 20*test_scores['old_n'])
                    row.update({f'cal_{key}':value for key,value in cal_scores.items()})
                    row.update({f'test_{key}':value for key,value in test_scores.items()})
                adjusted.append(row)
                if method==METHODS[0]:
                    identities.append({'task':task,'seed':seed,'old_calibration_n':old*20,
                        'calibration_n':seen*20,'test_n':seen*80,'q20_required_old_correct':required,
                        'calibration_indices':data['official_test_indices'][calibration].tolist(),
                        'test_indices':data['official_test_indices'][test].tolist()})
    return raw,unadjusted,adjusted,identities


def summarize(raw, unadjusted, adjusted):
    lookup = {(row['task'],row['seed'],row['method']):row for row in raw}
    score_lookup = {(row['task'],row['seed'],row['method']):row for row in adjusted}
    boundaries = {}
    for task in TASKS:
        old,seen = task*10,task*10+10
        fields = {'new_pct':('new_correct',10),'current10_pct':('current10_correct',10),
            'competition_pp':('competition_lost_correct',10),'old_pct':('old_correct',old),
            'all_pct':('all_correct',seen),'within_old_pct':('within_old_correct',old)}
        methods = {method:{metric:stats([lookup[(task,seed,method)][field] for seed in SEEDS],denominator)
                    for metric,(field,denominator) in fields.items()} for method in METHODS}
        pairs = {name:{metric:stats([lookup[(task,seed,a)][field]-lookup[(task,seed,b)][field] for seed in SEEDS],denominator)
                  for metric,(field,denominator) in fields.items()} for name,(a,b) in PAIRS.items()}
        new_surplus,new_cancel,new_gain,competition_surplus,competition_recovery,competition_difference = [],[],[],[],[],[]
        seed_decisions = []
        for seed in SEEDS:
            a,b,c = (lookup[(task,seed,method)] for method in METHODS[:3])
            new_n = a['new_correct']+b['new_correct']-2*c['new_correct']
            comp_n = 2*c['competition_lost_correct']-a['competition_lost_correct']-b['competition_lost_correct']
            new_surplus.append(new_n)
            competition_surplus.append(comp_n)
            new_cancel.append(b['new_correct']-c['new_correct'])
            new_gain.append(b['new_correct']-a['new_correct'])
            competition_recovery.append(c['competition_lost_correct']-b['competition_lost_correct'])
            competition_difference.append(a['competition_lost_correct']-b['competition_lost_correct'])
            seed_decisions.append({'seed':seed,'new_surplus_pp':new_n/20,'competition_surplus_pp':comp_n/20,
                'new_requirement_met':new_n>=0,'competition_requirement_met':comp_n>=0,'both_met':new_n>=0 and comp_n>=0})
        new_rule,comp_rule = stats(new_surplus,20),stats(competition_surplus,20)
        new_pass = new_rule['mean']>=0 and new_rule['nonnegative_seeds']>=3
        comp_pass = comp_rule['mean']>=0 and comp_rule['nonnegative_seeds']>=3
        prediction = {'new_original_gain':stats(new_gain,10),'new_realized_cancellation':stats(new_cancel,10),
            'new_half_threshold_surplus':new_rule,'new_requirement_supported':new_pass,
            'competition_original_difference':stats(competition_difference,10),
            'competition_realized_recovery':stats(competition_recovery,10),
            'competition_half_threshold_surplus':comp_rule,'competition_requirement_supported':comp_pass,
            'joint_supported':new_pass and comp_pass,'seed_decisions':seed_decisions}
        score_fields = {'new_pct':('new_correct',8),'old_pct':('old_correct',old*.8),
            'all_pct':('all_correct',seen*.8),'current10_pct':('current10_correct',8),
            'within_old_pct':('within_old_correct',old*.8),
            **{f'old_task{index}_pct':(f'old_task{index}_correct',8) for index in range(task)}}
        q20 = []
        for method in METHODS:
            rows = [score_lookup[(task,seed,method)] for seed in SEEDS]
            available = [row for row in rows if row['available']]
            q20.append({'method':method,'total_seed_units':5,'available_seeds':[row['seed'] for row in available],
                'unavailable_seeds':[row['seed'] for row in rows if not row['available']],
                'test_violations':[row['seed'] for row in available if row['test_violation']],
                **{metric:stats([row['test_'+field] for row in available],denominator) for metric,(field,denominator) in score_fields.items()}})
        q20_pairs = {}
        for name,(a,b) in PAIRS.items():
            usable = [seed for seed in SEEDS if score_lookup[(task,seed,a)]['available'] and score_lookup[(task,seed,b)]['available']]
            q20_pairs[name] = {'total_seed_units':5,'paired_available_seeds':usable,
                'unavailable_seeds':[seed for seed in SEEDS if seed not in usable],
                'differences':{metric:stats([score_lookup[(task,seed,a)]['test_'+field]-score_lookup[(task,seed,b)]['test_'+field]
                    for seed in usable],denominator) for metric,(field,denominator) in score_fields.items()}}
        boundaries[str(task)] = {'boundary':task,'independent_seed_units':5,'methods':methods,
            'paired_raw_effects':pairs,'fixed_prediction':prediction,'q20_descriptive':q20,'q20_pairs':q20_pairs,
            'old_calibration_n':old*20,'calibration_n':seen*20,'test_n':seen*80,
            'q20_required_old_correct':old*20*23//100}
    return {'schema':'revisit_supervision_v1','seeds':list(SEEDS),'boundaries':boundaries,
        'split_seed_base':2026110800,'score_target':20,'score_margin_pp':3,
        'prediction_equality_meets_threshold':True,'raw_seed_units_per_boundary':5,
        'score_analysis':'descriptive; no prespecified adjusted ordering'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    raw,unadjusted,adjusted,identities = calculate()
    summary = summarize(raw,unadjusted,adjusted)
    output.mkdir(parents=True, exist_ok=False)
    integer_fields = [key for key in raw[0] if key in ('task','seed','method') or key.endswith('_n') or key.endswith('_correct')]
    extra = [key for row in raw for key in row if (key.endswith('_n') or key.endswith('_correct')) and key not in integer_fields]
    integer_fields += list(dict.fromkeys(extra))
    write_csv(output/'counts.csv',[{key:row[key] for key in integer_fields if key in row} for row in raw])
    write_csv(output/'seed_scores.csv',raw)
    write_csv(output/'unadjusted_scores.csv',unadjusted)
    write_csv(output/'q20_scores.csv',adjusted)
    write_json(output/'split_identities.json',identities)
    write_json(output/'summary.json',summary)
    print(json.dumps({'status':'complete','raw_endpoints':len(raw),'q20_attempts':len(adjusted),
        'fixed_prediction':{task:value['fixed_prediction']['joint_supported'] for task,value in summary['boundaries'].items()}}),flush=True)


if __name__=='__main__':
    main()
