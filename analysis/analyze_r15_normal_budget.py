"""Recompute the fixed R15 matrix from logits using only Python and NumPy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SEEDS = [3101, 3102, 3103, 3104, 3105]
TASKS = [6, 7, 8, 9]
METHODS = ['sgd', 'er']
BRANCHES = ['actual', 'probe', 'fresh']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint(path, seed, task):
    """Check identities and recompute counts, never trust saved predictions."""
    with np.load(path, allow_pickle=False) as z:
        logits = z['logits']
        indices = z['indices']
        original = z['original_labels']
        labels = z['mapped_labels']
        order = z['class_order']
    expected_order = np.random.Generator(np.random.PCG64(seed + 40000)).permutation(100)
    inverse = np.argsort(expected_order)
    if logits.shape != (10000, 100) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError(f'Invalid complete finite FP32 logits: {path}')
    for a in (indices, original, labels, order):
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f'Noninteger identity: {path}')
    if not np.array_equal(order, expected_order):
        raise ValueError(f'Class order differs: {path}')
    if indices.shape != (10000,) or not np.array_equal(np.sort(indices), np.arange(10000)):
        raise ValueError(f'Missing/duplicate official index: {path}')
    if original.shape != (10000,) or (original < 0).any() or (original >= 100).any():
        raise ValueError(f'Invalid original labels: {path}')
    if not np.array_equal(np.bincount(original, minlength=100), np.full(100, 100)):
        raise ValueError(f'Unbalanced labels: {path}')
    if not np.array_equal(labels, inverse[original]):
        raise ValueError(f'Incorrect remapping: {path}')
    pred = logits[:, :10*(task+1)].argmax(1)
    local = logits[:, 10*task:10*(task+1)].argmax(1) + 10*task
    current = labels // 10 == task
    old = labels // 10 < task
    seen = labels // 10 <= task
    count = lambda mask: {'correct': int(((pred == labels) & mask).sum()), 'n': int(mask.sum())}
    record = {'current': count(current), 'old': count(old), 'seen': count(seen),
              'current_masked': {'correct': int(((local == labels) & current).sum()), 'n': int(current.sum())},
              'old_tasks': [count(labels // 10 == t) for t in range(task)],
              'current_predictions_to_old': int(((pred < 10*task) & current).sum()),
              'sha256': sha(path),
              'identity_sha256': hashlib.sha256(np.stack([indices, original, labels]).astype('<i8').tobytes()).hexdigest(),
              'logits_sha256': hashlib.sha256(logits.tobytes()).hexdigest()}
    return record


def pct(count):
    return Fraction(100 * count['correct'], count['n'])


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (5,):
        raise ValueError('The predeclared analysis requires all five paired seeds')
    draws = np.array(list(itertools.product(range(5), repeat=5)), dtype=np.int64)
    means = x[draws].mean(1)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)),
            'approximate_bootstrap_percentile_95': np.quantile(means, [.025, .975]).tolist(),
            'bootstrap_resamples': 3125}


def validate_lineage(root, manifest):
    """Bind counts to the complete sealed campaign and its actual endpoints."""
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    manifest_hash = sha(root/'manifest.json')
    settings = manifest['settings']
    if settings['seeds'] != SEEDS or settings['methods'] != METHODS or settings['diagnostic_tasks'] != TASKS:
        raise ValueError('Manifest does not contain the full predeclared matrix')
    if manifest['sources_sha256'].get('analysis/analyze_r15_normal_budget.py') != sha(__file__):
        raise ValueError('Analysis source differs from frozen manifest')
    batch = read(root/'batch.json')
    if batch.get('schema') != 'r15_normal_budget_v1' or batch['status'] != 'complete':
        raise ValueError('Campaign has not completed')
    seal_path = root/'all_training_complete.json'
    seal = read(seal_path)
    if seal['manifest_sha256'] != manifest_hash:
        raise ValueError('Training seal belongs to a different manifest')
    expected = {f'seed{s}_{m}' for s in SEEDS for m in METHODS}
    if len(seal['jobs']) != 10 or {j['id'] for j in seal['jobs']} != expected:
        raise ValueError('Incomplete or duplicated training seal')
    if batch['jobs'] != seal['jobs']:
        raise ValueError('Batch/seal identities differ')
    if datetime.fromisoformat(manifest['frozen_utc']) >= datetime.fromisoformat(seal['utc']):
        raise ValueError('Training seal predates protocol freeze')
    bindings, trains = {}, {}
    for sealed in seal['jobs']:
        name = sealed['id']
        job = root/'jobs'/name
        marker = job/'training_complete.json'
        if sealed['training_complete_path'] != f'jobs/{name}/training_complete.json' or sha(marker) != sealed['sha256']:
            raise ValueError('Training marker has changed after sealing')
        trained = read(marker)
        trains[name] = trained
        if trained['manifest_sha256'] != manifest_hash or trained['test_access'] is not False:
            raise ValueError('Invalid training provenance')
        if name != f"seed{trained['seed']}_{trained['method']}":
            raise ValueError('Job identity mismatch')
        if sha(job/'epochs.jsonl') != trained['epochs_journal_sha256']:
            raise ValueError('Epoch journal no longer matches completed training')
        result = read(job/'result.json')
        if (result['status'] != 'complete' or result['manifest_sha256'] != manifest_hash
                or result['seed'] != trained['seed'] or result['method'] != trained['method']
                or result['global_training_seal_sha256'] != sha(seal_path)
                or result['test_payload_sha256'] != '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864'):
            raise ValueError('Evaluation does not bind to official data and sealed training')
        start = read(job/'evaluation/started.json')
        if (start['global_training_seal_sha256'] != sha(seal_path)
                or datetime.fromisoformat(start['utc']) < datetime.fromisoformat(seal['utc'])):
            raise ValueError('Evaluation started before all training was sealed')
        branches = {(r['branch'], r['task']): r for r in trained['branches']}
        expected_branches = {('actual', t) for t in range(10)} | {(b, t) for b in ('probe', 'fresh') for t in TASKS}
        if len(trained['branches']) != 18 or set(branches) != expected_branches:
            raise ValueError('Incomplete branch matrix')
        endpoints = {(e['branch'], e['task']): e for e in result['endpoints']}
        if len(result['endpoints']) != 18 or set(endpoints) != expected_branches:
            raise ValueError('Incomplete evaluation endpoints')
        for key, b in branches.items():
            cp = job/'checkpoints'/b['checkpoint']['path']
            e = endpoints[key]
            if (sha(cp) != b['checkpoint']['file_sha256'] or sha(cp) != e['checkpoint_sha256']
                    or b['end_state_sha256'] != e['state_sha256']):
                raise ValueError('Checkpoint/evaluation lineage mismatch')
            name_npz = f'{key[0]}_task_{key[1]:02d}.npz'
            if e['path'] != name_npz or sha(job/'evaluation'/name_npz) != e['sha256']:
                raise ValueError('Raw logits differ from the executed endpoint')
            if not (b.get('reused_actual') or b.get('reused_fresh')):
                epochs = 200 if key == ('actual', 0) else 70
                replay = 5000*epochs if trained['method'] == 'er' and key[0] == 'actual' and key[1] > 0 else 0
                if (b['epochs'] != epochs or b['optimizer_steps'] != 79*epochs
                        or b['current_presentations'] != 5000*epochs or b['replay_presentations'] != replay):
                    raise ValueError('Training budget/exposure differs from protocol')
        bindings[name] = {'training_complete_sha256': sha(marker), 'result_sha256': sha(job/'result.json')}
    for seed in SEEDS:
        get = lambda method, branch, task: next(b for b in trains[f'seed{seed}_{method}']['branches'] if b['branch'] == branch and b['task'] == task)
        first_sgd, first_er = get('sgd', 'actual', 0), get('er', 'actual', 0)
        if (first_sgd['start_state_sha256'] != first_er['start_state_sha256']
                or first_sgd['end_state_sha256'] != first_er['end_state_sha256']):
            raise ValueError('Task-zero method pairing failed')
        for task in TASKS:
            actual, probe = get('sgd', 'actual', task), get('sgd', 'probe', task)
            if not probe.get('reused_actual') or probe['start_state_sha256'] != actual['start_state_sha256'] or probe['end_state_sha256'] != actual['end_state_sha256']:
                raise ValueError('SGD operation reuse is not an identity')
            if get('er', 'probe', task)['start_state_sha256'] != get('er', 'actual', task)['start_state_sha256']:
                raise ValueError('ER actual/probe do not share the same pre-state')
            f, reused = get('sgd', 'fresh', task), get('er', 'fresh', task)
            if (not reused.get('reused_fresh') or reused['optimizer_steps_executed'] != 0
                    or f['start_state_sha256'] != first_sgd['start_state_sha256']
                    or reused['start_state_sha256'] != f['start_state_sha256']
                    or reused['end_state_sha256'] != f['end_state_sha256']
                    or reused['source_job'] != f'seed{seed}_sgd'
                    or reused['source_training_complete_sha256'] != bindings[f'seed{seed}_sgd']['training_complete_sha256']
                    or reused['source_checkpoint_sha256'] != f['checkpoint']['file_sha256']):
                raise ValueError('Fresh reuse has broken identity/provenance')
    return {'training_seal_sha256': sha(seal_path), 'batch_sha256': sha(root/'batch.json'), 'jobs': bindings,
            'scope': 'File/manifest/endpoint lineage verified; independent official-label and checkpoint forward audit is separate.'}


def analyze(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new analysis output directory')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'r15_normal_budget_v1':
        raise ValueError('Wrong manifest schema')
    lineage = validate_lineage(root, manifest)
    # The full fixed matrix is enforced below even if extra files are present.
    rows = []
    records = {}
    for seed in SEEDS:
        for method in METHODS:
            job = root / 'jobs' / f'seed{seed}_{method}'
            if not (job / 'training_complete.json').is_file():
                raise ValueError(f'No completed fixed training endpoint: {job}')
            for task in TASKS:
                identity = None
                for branch in BRANCHES:
                    path = job / 'evaluation' / f'{branch}_task_{task:02d}.npz'
                    rec = endpoint(path, seed, task)
                    if identity is not None and rec['identity_sha256'] != identity:
                        raise ValueError('Branch test rows/labels are not paired')
                    identity = rec['identity_sha256']
                    records[seed, method, task, branch] = rec
                    rows.append({'seed': seed, 'method': method, 'task': task, 'branch': branch,
                                 'path': path.relative_to(root).as_posix(), **rec})
        for task in TASKS:
            a = records[seed, 'sgd', task, 'actual']
            p = records[seed, 'sgd', task, 'probe']
            if a['logits_sha256'] != p['logits_sha256']:
                raise ValueError('SGD actual/probe operation identity violated')
            for branch in BRANCHES:
                if records[seed, 'sgd', task, branch]['identity_sha256'] != records[seed, 'er', task, branch]['identity_sha256']:
                    raise ValueError('Method test rows/labels are not paired')
            if records[seed, 'sgd', task, 'fresh']['logits_sha256'] != records[seed, 'er', task, 'fresh']['logits_sha256']:
                raise ValueError('Fresh pairing violated across methods')
    paired = []
    for seed in SEEDS:
        row = {'seed': seed}
        for name, branch, metric in [('d_actual', 'actual', 'current'),
                                     ('d_probe', 'probe', 'current'),
                                     ('d_masked', 'actual', 'current_masked')]:
            diffs = [pct(records[seed, 'er', t, branch][metric]) - pct(records[seed, 'sgd', t, branch][metric]) for t in TASKS]
            d = sum(diffs) / len(TASKS)
            row[name] = float(d)
            row[name + '_exact'] = str(d)
            row[name + '_per_task'] = [float(x) for x in diffs]
        row['joint_direction'] = bool(Fraction(row['d_actual_exact']) < 0 and Fraction(row['d_probe_exact']) > 0 and Fraction(row['d_masked_exact']) > 0)
        paired.append(row)
    joint = sum(x['joint_direction'] for x in paired)
    summaries = {name: stats([r[name] for r in paired]) for name in ('d_actual', 'd_probe', 'd_masked')}
    for name, sign in [('d_actual', -1), ('d_probe', 1), ('d_masked', 1)]:
        summaries[name]['direction_count'] = sum(sign*Fraction(r[name+'_exact']) > 0 for r in paired)
        summaries[name]['zero_count'] = sum(Fraction(r[name+'_exact']) == 0 for r in paired)
    endpoint_summaries = []
    for method in METHODS:
        for branch in BRANCHES:
            row = {'method': method, 'branch': branch}
            for metric in ('current', 'current_masked', 'old', 'seen'):
                values = [float(sum(pct(records[s, method, t, branch][metric]) for t in TASKS)/4) for s in SEEDS]
                row[metric] = {'per_seed': values, **stats(values)}
            endpoint_summaries.append(row)
    # One-sided binomial tail for the predeclared joint-direction indicator.
    from math import comb
    tail = sum(comb(5, k) for k in range(joint, 6)) / 32
    result = {'schema': 'r15_analysis_v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': sha(manifest_path), 'analysis_sha256': sha(__file__),
              'lineage': lineage,
              'seeds': SEEDS, 'tasks': TASKS, 'n_endpoints': len(rows),
              'paired_seed_contrasts': paired, 'summary': summaries,
              'endpoint_summary': endpoint_summaries,
              'joint_direction_seeds': joint, 'strict_replication_gate': joint == 5,
              'joint_direction_binomial_tail_p_half': tail,
              'binomial_scope': 'Seed-level joint-direction probability conditional on fixed dataset and implementation; not three mean tests.',
              'scope': 'New training runs on already accessed CIFAR-100 test; 5 paired seeds and fixed four late tasks. Bootstrap coverage is approximate.',
              'raw_endpoint_counts': rows}
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# R15 原始 logits 复算', '', f'完整 {len(rows)} 个终点；5 个配对seed，先平均task 6–9。', '',
             '| seed | actual ER−SGD | probe ER−SGD | masked actual ER−SGD | 联合方向 |', '|---|---:|---:|---:|---|']
    for r in paired:
        lines.append(f"| {r['seed']} | {r['d_actual']:.3f} | {r['d_probe']:.3f} | {r['d_masked']:.3f} | {r['joint_direction']} |")
    lines += ['', f'严格复制门：{joint}/5，{joint == 5}。联合方向指示的单侧二项尾概率：{tail:.5f}。',
              '该门不是新机制、部署效用或接收等级的证明。完整同模型保留率、每任务计数和近似bootstrap区间见analysis.json。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'n_endpoints': len(rows), 'joint_direction_seeds': joint, 'summary': summaries}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.output)
