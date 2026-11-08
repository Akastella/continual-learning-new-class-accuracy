"""Evaluate existing loss-weight controls and select models using validation data."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys
import time

import numpy as np
import torch

STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 157, 314, 471, 628)
THRESHOLDS = (10, 20, 30)
SETS = ('current_only', 'usual_and_current_only', 'all_candidates')
SEEDS = tuple(range(201, 211))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def score(logits, labels, task):
    if logits.shape != (len(labels), 10) or not np.isfinite(logits).all():
        raise ValueError('Invalid FP32 outputs')
    predictions = logits.argmax(1)
    result = {}
    for name, mask in [('current', labels // 2 == task), ('old', labels // 2 < task),
                       ('seen', labels // 2 <= task), ('all', np.ones(len(labels), dtype=bool))]:
        total = int(mask.sum())
        correct = int((predictions[mask] == labels[mask]).sum())
        result[name + '_correct'] = correct
        result[name + '_n'] = total
        result[name + '_pct'] = 100 * correct / total if total else None
    result['class_correct'] = [int(((predictions == labels) & (labels == c)).sum()) for c in range(10)]
    result['class_n'] = np.bincount(labels, minlength=10).tolist()
    return result


def model_path(project, seed, candidate, task=4):
    if candidate == 'er_current_weighted_unit_sum':
        return project / f'results/r17_review_controls_v1/normalized_compensation/s{seed}_t{task}/endpoint.pt'
    method = 'er' if candidate.startswith('er_') else 'sgd'
    source = project / f'results/r9_followup/split_cifar10_classil_{method}_s{seed}'
    if candidate == 'er_current_weighted':
        return source / f'compensated_er_task_{task:03d}.pt'
    if candidate.endswith('_standard'):
        return source / f'post_task_{task:03d}.pt'
    step = int(candidate.rsplit('_', 1)[1])
    return project / f'results/r9_retention_trajectory/split_cifar10_classil_{method}_s{seed}/checkpoints/step_{step:06d}.pt'


@torch.inference_mode()
def infer(project, engine, seed, candidate, images, task=4):
    config = read(project / f'configs/r9_followup/split_cifar10_classil_er_s{seed}.json')
    path = model_path(project, seed, candidate, task)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    model = engine.local_model(config, config['init_seed'], 'cuda', state)
    model.eval()
    parts = []
    for start in range(0, len(images), 256):
        parts.append(model(images[start:start + 256].to('cuda')).cpu().numpy())
    outputs = np.concatenate(parts)
    state_digest = engine.state_hash(model)
    del model, state, parts
    torch.cuda.empty_cache()
    return outputs, {'checkpoint': path.relative_to(project).as_posix(),
                     'checkpoint_sha256': sha(path), 'state_sha256': state_digest}


def official_data(project):
    path = project / 'data/r9_official/cifar-10-batches-py/test_batch'
    raw = pickle.loads(path.read_bytes(), encoding='bytes')
    images = torch.from_numpy(raw[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)).clone()
    x = images.permute(0, 3, 1, 2).float() / 255
    x = (x - torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)) / torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1)
    return x, np.asarray(raw[b'labels'], dtype=np.int64), sha(path)


def evaluate_official(project, output, engine):
    images, original_labels, data_sha = official_data(project)
    rows, predictions = [], []
    for seed in SEEDS:
        order = np.random.Generator(np.random.PCG64(seed + 40000)).permutation(10)
        labels = np.argsort(order)[original_labels]
        normalized = read(project / f'results/r17_review_controls_v1/seed_{seed}.json')
        expected_normalized = {r['task']: r for r in normalized['normalized_compensation']['tasks']}
        expected_raw = read(project / f'results/r9_followup/split_cifar10_classil_er_s{seed}/result.json')
        for task in range(1, 5):
            for candidate in ('er_current_weighted', 'er_current_weighted_unit_sum'):
                logits, provenance = infer(project, engine, seed, candidate, images, task)
                scores = score(logits, labels, task)
                expected = expected_normalized[task]['correct'] if candidate.endswith('unit_sum') else expected_raw['tasks'][task]['compensated_er']['trace'][-1]['evaluation']['correct']
                if scores['current_correct'] != expected:
                    raise ValueError(f'Current-count mismatch at {seed=} {task=} {candidate=}')
                path = output / 'official_predictions' / f's{seed}_t{task}_{candidate}.npz'
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, logits=logits, labels=labels, official_indices=np.arange(10000, dtype=np.int64))
                row = dict(seed=seed, task=task, candidate=candidate, **scores)
                rows.append(row)
                predictions.append(dict(seed=seed, task=task, candidate=candidate, file=path.relative_to(output).as_posix(), sha256=sha(path), **provenance))
        # Baseline scores use previously exported predictions of these same endpoints.
        for method in ('sgd', 'er'):
            doc = read(project / f'results/r9_probe_retention/split_cifar10_classil_{method}_s{seed}.json')
            for record in doc['records']:
                if record['branch'] != 'actual':
                    continue
                task = record['task']
                counts = record['per_task']
                old_correct = sum(r['correct'] for r in counts[:-1])
                current_correct = counts[-1]['correct']
                rows.append(dict(seed=seed, task=task, candidate=method+'_standard',
                                 old_correct=old_correct, old_n=2000*task, old_pct=100*old_correct/(2000*task),
                                 current_correct=current_correct, current_n=2000, current_pct=current_correct/20,
                                 seen_correct=old_correct+current_correct, seen_n=2000*(task+1),
                                 seen_pct=100*(old_correct+current_correct)/(2000*(task+1))))
        print(json.dumps({'phase':'official', 'completed_seed':seed}), flush=True)
    seed_means = []
    for seed in SEEDS:
        for candidate in ('sgd_standard', 'er_standard', 'er_current_weighted', 'er_current_weighted_unit_sum'):
            selected = [r for r in rows if r['seed']==seed and r['candidate']==candidate]
            if len(selected) != 4:
                raise ValueError('Four noninitial tasks required')
            seed_means.append(dict(seed=seed, candidate=candidate, **{m:statistics.mean(r[m+'_pct'] for r in selected) for m in ('old','current','seen')}))
    groups = {candidate:{metric:summarize([r[metric] for r in seed_means if r['candidate']==candidate]) for metric in ('old','current','seen')} for candidate in sorted({r['candidate'] for r in seed_means})}
    contrasts = []
    for candidate in ('er_current_weighted','er_current_weighted_unit_sum'):
        for baseline in ('er_standard','sgd_standard'):
            contrast = dict(candidate=candidate, baseline=baseline)
            for metric in ('old','current','seen'):
                differences = [next(r[metric] for r in seed_means if r['seed']==seed and r['candidate']==candidate)-next(r[metric] for r in seed_means if r['seed']==seed and r['candidate']==baseline) for seed in SEEDS]
                contrast[metric] = summarize(differences)
            contrasts.append(contrast)
    result = dict(official_test_file_sha256=data_sha, task_rows=rows, seed_means=seed_means, summary=groups, contrasts=contrasts, predictions=predictions)
    write_json(output/'official_results.json', result)
    write_csv(output/'official_task_scores.csv', rows)
    write_csv(output/'official_seed_means.csv', seed_means)
    return result


def candidate_set(candidate):
    if '_current_only_step_' in candidate:
        return SETS
    if candidate.endswith('_standard'):
        return SETS[1:]
    return ('all_candidates',)


def external_validation(project, output, engine, data):
    images, original_labels, validation_ids, test_ids = data
    all_rows, provenance_rows = [], []
    for seed in SEEDS:
        order = np.random.Generator(np.random.PCG64(seed+40000)).permutation(10)
        labels = np.argsort(order)[original_labels]
        for method in ('sgd','er'):
            path = project / f'results/r9_retention_trajectory/split_cifar10_classil_{method}_s{seed}/validation.npz'
            with np.load(path,allow_pickle=False) as raw:
                if not np.array_equal(raw['indices'],validation_ids) or not np.array_equal(raw['labels'],labels[validation_ids]):
                    raise ValueError('Legacy validation split mismatch')
                for step in STEPS:
                    candidate = f'{method}_current_only_step_{step:04d}'
                    scores = score(raw[f'logits_{step}'],raw['labels'],4)
                    all_rows.append(dict(seed=seed,candidate=candidate,source='archived_validation',**scores))
            provenance_rows.append(dict(seed=seed,method=method,file=path.relative_to(project).as_posix(),sha256=sha(path)))
        for candidate in ('er_standard','sgd_standard','er_current_weighted','er_current_weighted_unit_sum'):
            logits, provenance = infer(project,engine,seed,candidate,images[validation_ids])
            scores = score(logits,labels[validation_ids],4)
            path = output/'validation_predictions'/f's{seed}_{candidate}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,logits=logits,labels=labels[validation_ids],indices=validation_ids)
            all_rows.append(dict(seed=seed,candidate=candidate,source='new_validation_inference',**scores))
            provenance_rows.append(dict(seed=seed,candidate=candidate,file=path.relative_to(output).as_posix(),sha256=sha(path),**provenance))
        print(json.dumps({'phase':'external_validation','completed_seed':seed}),flush=True)
    selections = []
    for seed in SEEDS:
        for subset in SETS:
            rows = [r for r in all_rows if r['seed']==seed and subset in candidate_set(r['candidate'])]
            if len(rows) != {'current_only':26,'usual_and_current_only':28,'all_candidates':30}[subset]:
                raise ValueError('Incomplete candidate set')
            for threshold in THRESHOLDS:
                eligible = [r for r in rows if 100*r['old_correct'] >= threshold*r['old_n']]
                chosen = min(eligible,key=lambda r:(-r['current_correct'],r['candidate'])) if eligible else None
                selections.append(dict(seed=seed,subset=subset,threshold=threshold,eligible_n=len(eligible),
                                       selected_candidate=chosen['candidate'] if chosen else None,
                                       validation_old_pct=chosen['old_pct'] if chosen else None,
                                       validation_current_pct=chosen['current_pct'] if chosen else None))
    write_json(output/'validation_results.json',dict(candidate_rows=all_rows,provenance=provenance_rows))
    write_csv(output/'validation_candidates.csv',all_rows)
    write_json(output/'selection.json',dict(created_utc=utc(),selections=selections))
    write_json(output/'selection_record.json',dict(created_utc=utc(),protocol_sha256=sha(output/'PROTOCOL.md'),
               validation_sha256=sha(output/'validation_results.json'),selection_sha256=sha(output/'selection.json'),
               rule='Validation current correct descending, full candidate name ascending, validation pooled old >= q.'))
    return selections, labels


def selected_test(project,output,engine,data,selections):
    images,original_labels,validation_ids,test_ids=data
    write_json(output/'test_started.json',dict(created_utc=utc(),selection_record_sha256=sha(output/'selection_record.json')))
    endpoints=[]
    for seed in SEEDS:
        order=np.random.Generator(np.random.PCG64(seed+40000)).permutation(10)
        labels=np.argsort(order)[original_labels]
        choices=sorted({r['selected_candidate'] for r in selections if r['seed']==seed and r['selected_candidate'] is not None})
        for candidate in choices:
            method='er' if candidate.startswith('er_') else 'sgd'
            key='actual' if candidate.endswith('_standard') else 'step_'+str(int(candidate.rsplit('_',1)[1])) if '_current_only_step_' in candidate else None
            cached=project/f'results/r9_retention_trajectory/split_cifar10_classil_{method}_s{seed}/test.npz'
            provenance={}
            source='new_test_inference'
            logits=None
            if key is not None:
                with np.load(cached,allow_pickle=False) as raw:
                    if not np.array_equal(raw['indices'],test_ids) or not np.array_equal(raw['labels'],labels[test_ids]):
                        raise ValueError('Legacy test split mismatch')
                    if 'logits_'+key in raw:
                        logits=raw['logits_'+key].copy()
                        provenance=dict(archived_file=cached.relative_to(project).as_posix(),archived_file_sha256=sha(cached))
                        source='archived_selected_test'
            if logits is None:
                logits,provenance=infer(project,engine,seed,candidate,images[test_ids])
            scores=score(logits,labels[test_ids],4)
            path=output/'selected_test_predictions'/f's{seed}_{candidate}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,logits=logits,labels=labels[test_ids],indices=test_ids)
            endpoints.append(dict(seed=seed,candidate=candidate,source=source,file=path.relative_to(output).as_posix(),sha256=sha(path),scores=scores,**provenance))
        print(json.dumps({'phase':'selected_test','completed_seed':seed,'unique_selected_candidates':choices}),flush=True)
    rows=[]
    for selected in selections:
        row=dict(selected)
        if row['selected_candidate'] is None:
            row.update(test_old_pct=None,test_current_pct=None,test_seen_pct=None,test_violation=None)
        else:
            endpoint=next(r for r in endpoints if r['seed']==row['seed'] and r['candidate']==row['selected_candidate'])
            scores=endpoint['scores']
            row.update(test_old_pct=scores['old_pct'],test_current_pct=scores['current_pct'],test_seen_pct=scores['seen_pct'],
                       test_violation=100*scores['old_correct']<row['threshold']*scores['old_n'])
        rows.append(row)
    groups=[]
    for subset in SETS:
        for threshold in THRESHOLDS:
            available=[r for r in rows if r['subset']==subset and r['threshold']==threshold and r['selected_candidate'] is not None]
            groups.append(dict(subset=subset,threshold=threshold,available_n=len(available),total_n=10,
                               test_violations=sum(r['test_violation'] for r in available),
                               current=summarize([r['test_current_pct'] for r in available]),
                               old=summarize([r['test_old_pct'] for r in available]),
                               seen=summarize([r['test_seen_pct'] for r in available]),
                               selected_counts={name:sum(r['selected_candidate']==name for r in available) for name in sorted({r['selected_candidate'] for r in available})}))
    result=dict(selection_rows=rows,groups=groups,selected_endpoints=endpoints)
    write_json(output/'external_results.json',result)
    write_csv(output/'selected_test_scores.csv',rows)
    write_csv(output/'selection_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)}|{metric+'_'+stat:r[metric][stat] for metric in ('old','current','seen') for stat in ('mean','sample_sd')} for r in groups])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    project,output=args.project.resolve(),args.output.resolve()
    if not (output/'PROTOCOL.md').is_file():
        raise ValueError('Write the protocol before executing inference')
    if (output/'started.json').exists():
        raise FileExistsError('This output already records an inference run')
    sys.path.insert(0,str(project))
    from experiments.r9_followup import engine
    from experiments.r9_retention_trajectory.evaluation import external_data
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    write_json(output/'started.json',dict(created_utc=utc(),protocol_sha256=sha(output/'PROTOCOL.md'),
               script_sha256=sha(Path(__file__)),batch_size=256,device=torch.cuda.get_device_name(),
               python=sys.version,numpy=np.__version__,torch=torch.__version__,training_updates=0))
    official=evaluate_official(project,output,engine)
    data=external_data(project/'data/r9_cifar101')
    stored=read(project/'configs/r9_retention_trajectory/split.json')
    if data[2].tolist()!=stored['validation_indices'] or data[3].tolist()!=stored['test_indices']:
        raise ValueError('Existing external split mismatch')
    selections,_=external_validation(project,output,engine,data)
    external=selected_test(project,output,engine,data,selections)
    write_json(output/'complete.json',dict(status='complete',created_utc=utc(),elapsed_seconds=time.perf_counter()-started,
               training_updates=0,official_endpoints=80,validation_candidates=300,selection_rows=90,
               selected_test_endpoints=len(external['selected_endpoints']),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
               official_results_sha256=sha(output/'official_results.json'),external_results_sha256=sha(output/'external_results.json')))
    print(json.dumps({'status':'complete','seconds':time.perf_counter()-started,'official_summary':official['summary'],'selection_summary':external['groups']}),flush=True)


if __name__=='__main__':
    main()
