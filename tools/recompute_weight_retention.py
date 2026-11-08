"""Recompute weight-control retention and validation-selected model comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 157, 314, 471, 628)
THRESHOLDS = (10, 20, 30)
SUBSETS = ('current_only', 'usual_and_current_only', 'all_candidates')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Different dictionary keys')
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Different list lengths')
        for left, right in zip(actual, expected):
            close(left, right)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if abs(float(actual) - float(expected)) > 1e-10:
            raise ValueError(f'Different number: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'Different value: {actual} != {expected}')


def score(predictions, labels, task):
    result = {}
    for name, mask in [('current', labels // 2 == task), ('old', labels // 2 < task),
                       ('seen', labels // 2 <= task), ('all', np.ones(len(labels), dtype=bool))]:
        total = int(mask.sum())
        correct = int((predictions[mask] == labels[mask]).sum())
        result.update({name+'_correct': correct, name+'_n': total,
                       name+'_pct': 100*correct/total if total else None})
    result['class_correct'] = [int(((predictions == labels) & (labels == c)).sum()) for c in range(10)]
    result['class_n'] = np.bincount(labels, minlength=10).tolist()
    return result


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None)


def membership(candidate):
    if '_current_only_step_' in candidate:
        return SUBSETS
    if candidate.endswith('_standard'):
        return SUBSETS[1:]
    return ('all_candidates',)


def recompute(root):
    directory = root/'evidence/weight_retention'
    export = read(directory/'EXPORT.json')
    for name, expected_hash in export['files'].items():
        if sha(directory/name) != expected_hash:
            raise ValueError('Changed input '+name)
    started = read(directory/'started.json')
    if sha(directory/'PROTOCOL.md') != started['protocol_sha256']:
        raise ValueError('Protocol differs from the pre-evaluation record')
    selection_record = read(directory/'selection_record.json')
    test_started = read(directory/'test_started.json')
    if datetime.fromisoformat(selection_record['created_utc']) >= datetime.fromisoformat(test_started['created_utc']):
        raise ValueError('Test evaluation preceded recorded selection')
    if sha(directory/'selection.json') != selection_record['selection_sha256']:
        raise ValueError('Changed selection record')
    official_reference = read(directory/'official_results.json')
    official_rows = []
    for endpoint in export['official_decisions']:
        with np.load(directory/endpoint['file'], allow_pickle=False) as raw:
            labels = raw['labels']
            predictions = raw['predictions']
            indices = raw['official_indices']
        if not np.array_equal(indices,np.arange(10000)) or not np.array_equal(np.bincount(labels,minlength=10),np.full(10,1000)):
            raise ValueError('Incomplete official test decision set')
        values = score(predictions,labels,endpoint['task'])
        original = next(row for row in official_reference['task_rows'] if all(row[k]==endpoint[k] for k in ('seed','task','candidate')))
        close(values,{key:original[key] for key in values})
        official_rows.append(dict(seed=endpoint['seed'],task=endpoint['task'],candidate=endpoint['candidate'],**values))
    for seed in range(201,211):
        for method in ('sgd','er'):
            raw=read(root/f'results/r9_followup/split_cifar10_classil_{method}_s{seed}/result.json')
            for task in range(1,5):
                entries=raw['tasks'][task]['post_task_evaluation']
                current=entries[task]['correct']; old=sum(item['correct'] for item in entries[:task])
                official_rows.append(dict(seed=seed,task=task,candidate=method+'_standard',
                                         current_pct=current/20,old_pct=old/(20*task),seen_pct=(old+current)/(20*(task+1))))
    seed_means=[]
    for seed in range(201,211):
        for candidate in ('sgd_standard','er_standard','er_current_weighted','er_current_weighted_unit_sum'):
            rows=[r for r in official_rows if r['seed']==seed and r['candidate']==candidate]
            if len(rows)!=4:
                raise ValueError('Missing task endpoint')
            seed_means.append(dict(seed=seed,candidate=candidate,**{metric:statistics.mean(r[metric+'_pct'] for r in rows) for metric in ('current','old','seen')}))
    close(seed_means,official_reference['seed_means'])
    official_summary={candidate:{metric:describe([r[metric] for r in seed_means if r['candidate']==candidate]) for metric in ('old','current','seen')} for candidate in sorted({r['candidate'] for r in seed_means})}
    close(official_summary,official_reference['summary'])
    validation=[]
    reference_validation=read(directory/'validation_results.json')
    for seed in range(201,211):
        for method in ('sgd','er'):
            path=root/f'evidence/retention_selection_original/split_cifar10_classil_{method}_s{seed}/validation.npz'
            with np.load(path,allow_pickle=False) as raw:
                for step in STEPS:
                    values=score(raw[f'logits_{step}'].argmax(1),raw['labels'],4)
                    validation.append(dict(seed=seed,candidate=f'{method}_current_only_step_{step:04d}',source='archived_validation',**values))
        for candidate in ('er_standard','sgd_standard','er_current_weighted','er_current_weighted_unit_sum'):
            path=directory/f'validation_predictions/s{seed}_{candidate}.npz'
            with np.load(path,allow_pickle=False) as raw:
                values=score(raw['logits'].argmax(1),raw['labels'],4)
            validation.append(dict(seed=seed,candidate=candidate,source='new_validation_inference',**values))
    close(validation,reference_validation['candidate_rows'])
    choices=[]
    for seed in range(201,211):
        for subset in SUBSETS:
            candidates=[r for r in validation if r['seed']==seed and subset in membership(r['candidate'])]
            for q in THRESHOLDS:
                eligible=[r for r in candidates if 100*r['old_correct']>=q*r['old_n']]
                chosen=min(eligible,key=lambda r:(-r['current_correct'],r['candidate'])) if eligible else None
                choices.append(dict(seed=seed,subset=subset,threshold=q,eligible_n=len(eligible),
                                    selected_candidate=chosen['candidate'] if chosen else None,
                                    validation_old_pct=chosen['old_pct'] if chosen else None,
                                    validation_current_pct=chosen['current_pct'] if chosen else None))
    close(choices,read(directory/'selection.json')['selections'])
    external=read(directory/'external_results.json')
    endpoints={}
    for endpoint in external['selected_endpoints']:
        path=directory/endpoint['file']
        if sha(path)!=endpoint['sha256']:
            raise ValueError('Changed selected-test evidence')
        with np.load(path,allow_pickle=False) as raw:
            values=score(raw['logits'].argmax(1),raw['labels'],4)
        close(values,endpoint['scores'])
        endpoints[endpoint['seed'],endpoint['candidate']]=values
    if set(endpoints)!={(r['seed'],r['selected_candidate']) for r in choices if r['selected_candidate'] is not None}:
        raise ValueError('Unexpected or missing test endpoint')
    selected=[]
    for chosen in choices:
        row=dict(chosen)
        if row['selected_candidate'] is None:
            row.update(test_old_pct=None,test_current_pct=None,test_seen_pct=None,test_violation=None)
        else:
            values=endpoints[row['seed'],row['selected_candidate']]
            row.update(test_old_pct=values['old_pct'],test_current_pct=values['current_pct'],test_seen_pct=values['seen_pct'],
                       test_violation=100*values['old_correct']<row['threshold']*values['old_n'])
        selected.append(row)
    close(selected,external['selection_rows'])
    groups=[]
    for subset in SUBSETS:
        for q in THRESHOLDS:
            rows=[r for r in selected if r['subset']==subset and r['threshold']==q and r['selected_candidate'] is not None]
            groups.append(dict(subset=subset,threshold=q,available_n=len(rows),total_n=10,
                               test_violations=sum(r['test_violation'] for r in rows),
                               current=describe([r['test_current_pct'] for r in rows]),
                               old=describe([r['test_old_pct'] for r in rows]),
                               seen=describe([r['test_seen_pct'] for r in rows]),
                               selected_counts={name:sum(r['selected_candidate']==name for r in rows) for name in sorted({r['selected_candidate'] for r in rows})}))
    close(groups,external['groups'])
    paired=[]
    for q in THRESHOLDS:
        seed_rows=[]
        for seed in range(201,211):
            full=next(r for r in selected if r['seed']==seed and r['threshold']==q and r['subset']=='all_candidates')
            usual=next(r for r in selected if r['seed']==seed and r['threshold']==q and r['subset']=='usual_and_current_only')
            seed_rows.append(dict(seed=seed,threshold=q,selected_all=full['selected_candidate'],
                                  selected_usual=usual['selected_candidate'],all_test_violation=full['test_violation'],
                                  **{metric+'_difference_pp':full['test_'+metric+'_pct']-usual['test_'+metric+'_pct'] for metric in ('old','current','seen')}))
        paired.append(dict(threshold=q,summary={metric:dict(mean=statistics.mean(r[metric+'_difference_pp'] for r in seed_rows),
                           sample_sd=statistics.stdev(r[metric+'_difference_pp'] for r in seed_rows)) for metric in ('old','current','seen')},seed_rows=seed_rows))
    close(paired,read(directory/'paired_expansion.json'))
    derpp_rows=[]
    for seed in range(201,211):
        raw=read(root/f'results/r9_derpp_challenge/split_cifar10_classil_derpp_s{seed}/result.json')
        old_values=[]; seen_values=[]
        for task in range(1,5):
            values=raw['tasks'][task]['post_task_evaluation']
            old_values.append(100*sum(r['correct'] for r in values[:task])/sum(r['n'] for r in values[:task]))
            seen_values.append(100*sum(r['correct'] for r in values[:task+1])/sum(r['n'] for r in values[:task+1]))
        derpp_rows.append(dict(seed=seed,old=statistics.mean(old_values),seen=statistics.mean(seen_values)))
    derpp=dict(per_seed=derpp_rows,summary={metric:dict(mean=statistics.mean(r[metric] for r in derpp_rows),
               sample_sd=statistics.stdev(r[metric] for r in derpp_rows)) for metric in ('old','seen')})
    close(derpp,read(directory/'derpp_retention.json'))
    return dict(status='PASS',official_control_endpoints=80,validation_candidates=300,
                selection_rules=90,selected_test_endpoints=len(endpoints),
                official_summary=official_summary,selection_summary=groups,paired_expansion=paired,derpp_retention=derpp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=recompute(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','official_control_endpoints','validation_candidates','selection_rules','selected_test_endpoints')}))


if __name__=='__main__':
    main()
