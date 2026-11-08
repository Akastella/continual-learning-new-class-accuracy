"""Independent counts/selection audit; torch is used only to deserialize weights.

No experiment, runner, evaluation, or production analysis module is imported.
This verifies saved evidence, not the correctness of every SGD update. The
separate functional replay checks inference; endpoint equality checks lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'configs/r9_followup/manifest.json'
CANONICAL = 'configs/r9_retention_trajectory/manifest.json'
ORIGINAL_SHA = 'bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e'
OUTPUT = 'results/r9_retention_trajectory'
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 157, 314, 471, 628]
THRESHOLDS = [20, 10, 0]
SEEDS = list(range(201, 211))
METHODS = ['sgd', 'er']
SETTINGS = {
    'kind': 'posthoc_external_validation_selected_trajectory', 'seeds': SEEDS,
    'methods': METHODS, 'task': 4, 'steps': STEPS,
    'thresholds_pct': THRESHOLDS, 'primary_threshold_pct': 20,
    'split_seed_base': 20260915, 'per_class_validation': 100,
    'per_class_test': 100, 'limit_seconds': 3300,
    'total_optimizer_updates': 12560, 'validation_endpoints': 260,
    'candidates': 'all10', 'batch_eval': 512,
}
SOURCES = [
    'tools/run_r9_retention_trajectory.py', 'tools/freeze_r9_retention_trajectory.py',
    'tools/prepare_r9_retention_split.py',
    'experiments/r9_retention_trajectory/core.py',
    'experiments/r9_retention_trajectory/evaluation.py',
    'analysis/audit_r9_retention_trajectory.py',
    'analysis/replay_r9_retention_trajectory.py',
    'tests/test_r9_retention_selection.py',
    'tests/test_r9_retention_trajectory_core.py',
    'protocol/R9_RETENTION_TRAJECTORY.md', 'requirements-r9-windows-lock.txt',
]
EXTERNAL_HASHES = {
    'data/r9_cifar101/cifar10.1_v6_data.npy':
        '2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6',
    'data/r9_cifar101/cifar10.1_v6_labels.npy':
        'ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def safe(base, name):
    require(isinstance(name, str) and '\\' not in name, 'nonportable path')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts
            and ':' not in name and str(relative) == name, 'unsafe relative path ' + name)
    result = (base / name).resolve()
    require(result.is_relative_to(base.resolve()), 'path outside base')
    return result


def instant(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    require(value.tzinfo is not None, 'timezone missing')
    return value


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_arrays(state):
    require(isinstance(state, dict) and bool(state), 'empty model state')
    result = {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy()
        require(array.dtype == np.float32 and np.isfinite(array).all(),
                'nonfinite/non-FP32 state ' + name)
        result[name] = array
    return result


def state_digest(state):
    result = hashlib.sha256()
    for name, array in state_arrays(state).items():
        result.update(name.encode())
        result.update(bytes.fromhex(digest(array)))
    return result.hexdigest()


def equal_states(left, right):
    a, b = state_arrays(left), state_arrays(right)
    require(list(a) == list(b), 'state keys/order differ')
    for name in a:
        require(a[name].shape == b[name].shape and np.array_equal(a[name], b[name]),
                'checkpoint tensor differs: ' + name)
    return len(a)


def scores(logits, labels):
    require(logits.shape == (1000, 10) and logits.dtype == np.float32
            and np.isfinite(logits).all(), 'invalid prediction matrix')
    require(labels.shape == (1000,) and labels.dtype == np.int64
            and np.array_equal(np.unique(labels), np.arange(10))
            and np.array_equal(np.bincount(labels), np.full(10, 100)), 'label schema')
    hits = np.argmax(logits, axis=1) == labels
    counts = [int(hits[labels == c].sum()) for c in range(10)]
    old, current = sum(counts[:8]), sum(counts[8:])
    return {'class_correct': counts, 'class_n': [100] * 10,
            'old_correct': old, 'old_n': 800, 'current_correct': current,
            'current_n': 200, 'old_pct': old / 8, 'current_pct': current / 2,
            'all_pct': sum(counts) / 10,
            'old_task_pct': [sum(counts[i:i + 2]) / 2 for i in (0, 2, 4, 6)]}


def select(rows, threshold):
    # Deliberately independent loop; uses integer counts and no test quantities.
    best = None
    for row in rows:
        if 100 * row['scores']['old_correct'] < threshold * 800:
            continue
        if best is None or row['scores']['current_correct'] > best['scores']['current_correct']:
            best = row
        elif row['scores']['current_correct'] == best['scores']['current_correct'] \
                and row['step'] < best['step']:
            best = row
    return None if best is None else best['step']


def describe(values):
    n = len(values)
    return {'n': n, 'mean': float(np.mean(values)) if n else None,
            'sample_sd': float(np.std(values, ddof=1)) if n > 1 else None}


def metrics(rows):
    return {key: describe([r[key] for r in rows])
            for key in ('old_pct', 'current_pct', 'all_pct')}


def audit(directory, root=ROOT):
    # Import only here so importing pure counting helpers never loads torch.
    import torch

    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = read(directory / 'frozen_manifest.json')
    require(manifest['settings'] == SETTINGS, 'settings drift')
    manifest_sha = sha(directory / 'frozen_manifest.json')
    require(manifest_sha == sha(root / CANONICAL), 'canonical manifest mismatch')
    require(sha(root / ORIGINAL) == ORIGINAL_SHA, 'original manifest mismatch')
    original = read(root / ORIGINAL)
    runs = [r for r in original['runs'] if r['id'].startswith('split_cifar10_classil_')]
    wanted_ids = [f'split_cifar10_classil_{method}_s{seed}'
                  for seed in SEEDS for method in METHODS]
    require([r['id'] for r in runs] == wanted_ids, 'original 20-history matrix')
    wanted_inputs = {
        ORIGINAL, 'third_party/r9_cifar101_reference/SOURCE.json',
        'third_party/r9_cifar101_reference/README.md',
        'third_party/r9_cifar101_reference/LICENSE',
        'audit/r9/POSITIVE_RETENTION_DESIGN_REVIEW.md',
        'audit/r9/RETENTION_TRAJECTORY_RESOURCE.json', *EXTERNAL_HASHES,
        'configs/r9_retention_trajectory/split.json',
        *[f'data/r9_official/cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)],
    }
    for run in runs:
        wanted_inputs.add(run['config_path'])
        wanted_inputs.update(run['output'] + '/' + name for name in
                             ['pre_task_004.pt', 'probe_task_004.pt', 'post_task_004.pt',
                              'result.json', 'data_manifest.json', 'artifacts.json'])
    require(set(manifest['input_sha256']) == wanted_inputs, 'input closure')
    require(set(manifest['source_sha256']) == set(original['source_sha256']) | set(SOURCES),
            'source closure')
    for name, value in original['source_sha256'].items():
        require(manifest['source_sha256'][name] == value, 'original source changed')
    for group in ('input_sha256', 'source_sha256'):
        for name, value in manifest[group].items():
            require(sha(safe(root, name)) == value, 'frozen drift: ' + name)
    for name, value in EXTERNAL_HASHES.items():
        require(manifest['input_sha256'][name] == value, 'external version mismatch')
    for name, value in manifest['source_sha256'].items():
        require(sha(safe(directory / 'source', name)) == value, 'source snapshot drift')

    start, seal, test_start = [read(directory / name) for name in
                               ['started.json', 'selection_seal.json', 'test_started.json']]
    complete, watchdog = read(directory / 'complete.json'), read(directory / 'watchdog.json')
    require(start['manifest_sha256'] == complete['manifest_sha256'] == manifest_sha,
            'manifest lineage')
    chronology = [instant(manifest['frozen_utc']), instant(start['utc']),
                  instant(seal['created_utc']), instant(test_start['utc']),
                  instant(complete['finished_utc'])]
    require(chronology == sorted(chronology) and chronology[2] < chronology[3],
            'freeze/selection/test chronology')
    explicit_split = read(root / 'configs/r9_retention_trajectory/split.json')
    require(instant(explicit_split['created_utc']) <= chronology[0]
            and explicit_split['labels_sha256'] == EXTERNAL_HASHES['data/r9_cifar101/cifar10.1_v6_labels.npy']
            and explicit_split['model_forward_calls'] == 0 and explicit_split['images_decoded'] is False,
            'explicit split chronology and provenance')
    require(complete['status'] == watchdog['status'] == 'complete'
            and watchdog['exit_code'] == 0 and watchdog['limit_seconds'] == 3300
            and 0 < watchdog['elapsed_seconds'] <= 3300, 'completion/time budget')
    require((chronology[-1] - chronology[1]).total_seconds() <= watchdog['elapsed_seconds'] + 1,
            'wall-clock and watchdog disagree')
    require(complete['histories'] == 20 and complete['optimizer_updates'] == 12560
            and complete['validation_endpoints'] == 260, 'complete scope')
    seal_sha = sha(directory / 'selection_seal.json')
    require(test_start['selection_seal_sha256'] == seal_sha, 'test selection lineage')
    require(seal['selection_sha256'] == sha(directory / 'selection.json'), 'selection seal')
    choices = read(directory / 'selection.json')
    require(set(choices) == set(wanted_ids), 'selection history inventory')
    val_files = {rid + '/' + name for rid in wanted_ids
                 for name in ('validation.npz', 'trajectory.json')}
    require(set(seal['validation_files_sha256']) == val_files, 'seal coverage')
    for name, value in seal['validation_files_sha256'].items():
        require(sha(safe(directory, name)) == value, 'sealed validation changed')

    base_files = {'started.json', 'environment.json', 'frozen_manifest.json', 'split.npz',
                  'selection.json', 'selection_seal.json', 'test_started.json'}
    wanted_artifacts = base_files | {'source/' + n for n in manifest['source_sha256']}
    for rid in wanted_ids:
        wanted_artifacts.update(rid + '/' + n for n in
                                ('validation.npz', 'trajectory.json', 'test.npz', 'test.json'))
        wanted_artifacts.update(f'{rid}/checkpoints/step_{s:06d}.pt' for s in STEPS)
    artifacts = read(directory / 'artifacts.json')
    require(set(artifacts) == wanted_artifacts, 'artifact inventory')
    for name, value in artifacts.items():
        require(sha(safe(directory, name)) == value, 'artifact hash: ' + name)
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual_files == wanted_artifacts | {'artifacts.json', 'complete.json',
                                               'console.log', 'watchdog.json'}, 'output inventory')
    env = read(directory / 'environment.json')
    require(env['fp32'] is True and env['threads'] == 2 and env['evaluation_batch'] == 512
            and env['no_user_site'] is True, 'recorded environment flags')
    require(env['deterministic_algorithms'] is True and env['cudnn_benchmark'] is False
            and env['cudnn_deterministic'] is True and env['matmul_tf32'] is False
            and env['cudnn_tf32'] is False and env['cublas_workspace_config'] == ':4096:8',
            'recorded determinism and precision')
    lock = dict(line.split('==', 1) for line in
                (root / 'requirements-r9-windows-lock.txt').read_text(encoding='utf-8-sig').splitlines()
                if '==' in line and not line.startswith('#'))
    require(env['torch'] == lock['torch'] and env['numpy'] == lock['numpy'], 'recorded versions')
    for name in ('torchvision', 'pillow'):
        require(env[name] == lock[name], 'recorded ' + name)

    labels = np.load(root / 'data/r9_cifar101/cifar10.1_v6_labels.npy', allow_pickle=False)
    require(labels.shape == (2000,) and labels.dtype.kind in 'iu'
            and np.array_equal(np.unique(labels), np.arange(10))
            and np.array_equal(np.bincount(labels), np.full(10, 200)), 'official labels schema')
    labels = labels.astype(np.int64)
    images = np.load(root / 'data/r9_cifar101/cifar10.1_v6_data.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == (2000, 32, 32, 3) and images.dtype == np.uint8, 'official image schema')
    del images
    v, t = [], []
    for c in range(10):
        ids = np.flatnonzero(labels == c)
        ordered = np.random.Generator(np.random.PCG64(20260915 + c)).permutation(ids)
        v.extend(ordered[:100]); t.extend(ordered[100:])
    val_ids, test_ids = np.sort(v).astype(np.int64), np.sort(t).astype(np.int64)
    require(explicit_split['validation_indices'] == val_ids.tolist()
            and explicit_split['test_indices'] == test_ids.tolist(), 'explicit independent split')
    require(not np.intersect1d(val_ids, test_ids).size
            and np.array_equal(np.sort(np.r_[val_ids, test_ids]), np.arange(2000)), 'split disjointness')
    with np.load(directory / 'split.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'validation_indices', 'test_indices', 'original_labels'}, 'split keys')
        for key, expected in [('validation_indices', val_ids), ('test_indices', test_ids),
                              ('original_labels', labels)]:
            require(saved[key].dtype == np.int64 and np.array_equal(saved[key], expected), 'split ' + key)

    train_x, train_y, batch_hashes = [], [], {}
    for i in range(1, 6):
        name = f'data_batch_{i}'
        path = root / 'data/r9_official/cifar-10-batches-py' / name
        raw = pickle.loads(path.read_bytes(), encoding='latin1')
        train_x.append(raw['data']); train_y.extend(raw['labels']); batch_hashes[name] = sha(path)
    train_x = np.vstack(train_x).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    train_y = np.asarray(train_y, dtype=np.int64)
    require(train_x.shape == (50000, 32, 32, 3) and train_x.dtype == np.uint8
            and np.array_equal(np.bincount(train_y), np.full(10, 5000)), 'official training schema')
    official_image_hash, official_label_hash = digest(train_x), digest(train_y)

    seed_units, anchor_units, validation_units, endpoint_proofs = [], [], [], []
    total_test, total_updates, tensor_checks = 0, 0, 0
    for run in runs:
        rid = run['id']; rd = directory / rid
        cfg = read(root / run['config_path'])
        require(sha(root / run['config_path']) == run['config_sha256'], 'original config')
        require(rid == f"split_cifar10_classil_{cfg['method']}_s{cfg['seed']}"
                and cfg['seed'] in SEEDS and cfg['method'] in METHODS, 'run identity')
        require(cfg['class_order_seed'] == cfg['seed'] + 40000
                and cfg['order_seed'] == cfg['seed'] + 30000 and cfg['init_seed'] == cfg['seed'] + 10000
                and cfg['epochs'] == 4 and cfg['batch_size'] == 64 and cfg['lr'] == .01
                and cfg['max_train'] is None, 'original protocol constants')
        order = np.random.Generator(np.random.PCG64(cfg['class_order_seed'])).permutation(10)
        mapping = np.argsort(order); mapped = mapping[labels].astype(np.int64)
        dm = read(root / run['output'] / 'data_manifest.json')
        require(dm['provenance']['class_order_original_ids'] == order.tolist()
                and dm['provenance']['original_class_to_global_output'] == mapping.tolist(), 'original mapping')
        old_artifacts = read(root / run['output'] / 'artifacts.json')['files']
        for name in ('pre_task_004.pt', 'probe_task_004.pt', 'post_task_004.pt', 'result.json', 'data_manifest.json'):
            require(old_artifacts[name] == manifest['input_sha256'][run['output'] + '/' + name],
                    'original artifact lineage ' + name)
        original_result = read(root / run['output'] / 'result.json')['tasks'][4]['probe']
        row = read(rd / 'trajectory.json'); summary = row['summary']
        require((row['run_id'], row['seed'], row['method'], row['task'])
                == (rid, cfg['seed'], cfg['method'], 4), 'trajectory identity')
        require(row['original_terminal_tensor_exact'] is True, 'reported terminal mismatch')
        require((summary['start_step'], summary['stop_step'], summary['total_steps'],
                 summary['optimizer_steps_executed']) == (0, 628, 628, 628), 'trajectory update count')
        require(summary['batch_order_sha256'] == original_result['batch_order_sha256']
                and summary['final_state_sha256'] == original_result['final_state_sha256'], 'legacy trajectory identity')
        require(0 <= summary['callback_seconds'] <= summary['seconds'] <= 3300, 'trajectory timings')
        total_updates += summary['optimizer_steps_executed']
        ids = np.flatnonzero(np.isin(train_y, order[8:10])).astype(np.int64)
        normalized = train_x[ids].transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
        normalized = (normalized - np.asarray([.4914, .4822, .4465], np.float32)[None, :, None, None]) \
            / np.asarray([.2470, .2435, .2616], np.float32)[None, :, None, None]
        current_y = mapping[train_y[ids]].astype(np.int64)
        provenance = {'decoded_splits': ['train'], 'batch_sha256': batch_hashes,
                      'official_train_images_sha256': official_image_hash,
                      'official_train_labels_sha256': official_label_hash,
                      'class_order_original_ids': order.tolist(), 'task_id': 4,
                      'official_indices': ids.tolist(), 'images_sha256': digest(normalized),
                      'labels_sha256': digest(current_y), 'proxy_unique_train_images': 1000,
                      'proxy_forward_images_per_callback': 2000}
        require(row['train_provenance'] == provenance, 'independent training provenance')
        archived = dm['tasks'][4]['splits']['train']
        for key in ('official_indices', 'images_sha256', 'labels_sha256'):
            require(provenance[key] == archived[key], 'original training input ' + key)
        del normalized, current_y
        identity = {'config': cfg, 'task_id': 4, 'n': 10000,
                    'x_sha256': provenance['images_sha256'], 'y_sha256': provenance['labels_sha256'],
                    'order_sha256': summary['batch_order_sha256'], 'candidates': STEPS}
        require([r['step'] for r in row['validation']] == STEPS
                and [r['step'] for r in summary['checkpoints']] == STEPS, 'candidate completeness')
        state_hashes = {}
        for entry, recorded in zip(row['validation'], summary['checkpoints']):
            step = entry['step']; relative = f'{rid}/checkpoints/step_{step:06d}.pt'
            full_relative = OUTPUT + '/' + relative
            require(entry['checkpoint'] == recorded['path'] == full_relative, 'candidate checkpoint path')
            require(entry['checkpoint_sha256'] == artifacts[relative], 'candidate checkpoint hash')
            checkpoint = torch.load(rd / 'checkpoints' / f'step_{step:06d}.pt',
                                    map_location='cpu', weights_only=True)
            require(set(checkpoint) == {'step', 'identity', 'model', 'optimizer', 'state_sha256'}
                    and checkpoint['step'] == step and checkpoint['identity'] == identity, 'checkpoint identity')
            state_hashes[step] = state_digest(checkpoint['model'])
            require(state_hashes[step] == checkpoint['state_sha256'] == recorded['state_sha256'], 'state hash')
            opt = checkpoint['optimizer']
            require(not opt['state'] and len(opt['param_groups']) == 1, 'unexpected optimizer state')
            pg = opt['param_groups'][0]
            require(pg['lr'] == .01 and pg['momentum'] == 0 and pg['weight_decay'] == 0
                    and pg['dampening'] == 0 and pg['nesterov'] is False
                    and pg['maximize'] is False and pg['params'] == list(range(len(checkpoint['model']))),
                    'optimizer recipe')
            if step in (0, 628):
                anchor = 'pre_task_004.pt' if step == 0 else 'probe_task_004.pt'
                expected = torch.load(root / run['output'] / anchor, map_location='cpu', weights_only=True)
                tensor_checks += equal_states(checkpoint['model'], expected)
            del checkpoint
        require(state_hashes[628] == summary['final_state_sha256'], 'final state digest')
        require(sha(rd / 'validation.npz') == row['validation_npz_sha256'], 'validation NPZ lineage')
        validated = []
        with np.load(rd / 'validation.npz', allow_pickle=False) as arrays:
            require(set(arrays.files) == {'indices', 'labels'} | {'logits_' + str(s) for s in STEPS}, 'validation array inventory')
            require(arrays['indices'].dtype == np.int64 and np.array_equal(arrays['indices'], val_ids)
                    and arrays['labels'].dtype == np.int64 and np.array_equal(arrays['labels'], mapped[val_ids]), 'validation mapping')
            for item in row['validation']:
                measured = scores(arrays['logits_' + str(item['step'])], mapped[val_ids])
                require(measured == item['scores'], 'validation counts')
                validated.append({'step': item['step'], 'scores': measured})
                validation_units.append({'seed': cfg['seed'], 'method': cfg['method'], **validated[-1]})
        require(list(choices[rid]) == [str(q) for q in THRESHOLDS], 'threshold order/inventory')
        recomputed = {str(q): select(validated, q) for q in THRESHOLDS}
        require(choices[rid] == recomputed, 'validation-only selection mismatch')
        selected_steps = sorted({0, 628} | {s for s in recomputed.values() if s is not None})
        keys = ['step_' + str(s) for s in selected_steps] + ['actual']
        test_record = read(rd / 'test.json')
        require(test_record['run_id'] == rid and test_record['selection_seal_sha256'] == seal_sha
                and test_record['npz_sha256'] == sha(rd / 'test.npz'), 'test lineage')
        require([r['key'] for r in test_record['records']] == keys, 'test only selected plus fixed anchors')
        test_scores = {}
        with np.load(rd / 'test.npz', allow_pickle=False) as arrays:
            require(set(arrays.files) == {'indices', 'labels'} | {'logits_' + k for k in keys}, 'test array inventory')
            require(arrays['indices'].dtype == np.int64 and np.array_equal(arrays['indices'], test_ids)
                    and arrays['labels'].dtype == np.int64 and np.array_equal(arrays['labels'], mapped[test_ids]), 'test mapping')
            for item in test_record['records']:
                key = item['key']
                if key == 'actual':
                    expected_path = run['output'] + '/post_task_004.pt'
                    expected_sha = manifest['input_sha256'][expected_path]
                    actual = torch.load(root / expected_path, map_location='cpu', weights_only=True)
                    expected_state = state_digest(actual)
                else:
                    step = int(key.split('_')[1])
                    expected_path = f'{OUTPUT}/{rid}/checkpoints/step_{step:06d}.pt'
                    expected_sha = artifacts[f'{rid}/checkpoints/step_{step:06d}.pt']
                    expected_state = state_hashes[step]
                require(item['checkpoint'] == expected_path and item['checkpoint_sha256'] == expected_sha
                        and item['state_sha256'] == expected_state, 'test checkpoint lineage')
                test_scores[key] = scores(arrays['logits_' + key], mapped[test_ids])
                require(test_scores[key] == item['scores'], 'test counts')
                total_test += 1
        for threshold in THRESHOLDS:
            chosen = recomputed[str(threshold)]
            feasible = chosen is not None
            test_score = test_scores['step_' + str(chosen)] if feasible else None
            val_score = next(r['scores'] for r in validated if r['step'] == chosen) if feasible else None
            seed_units.append({'seed': cfg['seed'], 'method': cfg['method'], 'threshold_pct': threshold,
                               'validation_feasible': feasible, 'selected_step': chosen,
                               'validation': val_score, 'test': test_score,
                               'test_violation': (100 * test_score['old_correct'] < threshold * 800) if feasible else None,
                               'test_old_slack_pp': test_score['old_pct'] - threshold if feasible else None})
        for anchor, key in [('pre', 'step_0'), ('terminal', 'step_628'), ('actual', 'actual')]:
            anchor_units.append({'seed': cfg['seed'], 'method': cfg['method'],
                                 'anchor': anchor, 'test': test_scores[key]})
        endpoint_proofs.append({'run_id': rid, 'pre_exact': True, 'terminal_exact': True,
                                'terminal_state_sha256': state_hashes[628],
                                'validation_npz_sha256': sha(rd / 'validation.npz'),
                                'test_npz_sha256': sha(rd / 'test.npz')})
    require(total_updates == 12560 and len(validation_units) == 260
            and total_test == complete['test_endpoints'], 'matrix totals')

    groups, paired, anchor_groups = [], [], []
    for threshold in THRESHOLDS:
        for method in METHODS:
            all_rows = [r for r in seed_units if r['threshold_pct'] == threshold and r['method'] == method]
            require([r['seed'] for r in all_rows] == SEEDS, 'group seed coverage')
            usable = [r for r in all_rows if r['validation_feasible']]
            groups.append({'threshold_pct': threshold, 'method': method, 'total_seeds': 10,
                           'validation_feasible_n': len(usable),
                           'validation_feasible_seeds': [r['seed'] for r in usable],
                           'test_violation_n': sum(r['test_violation'] for r in usable),
                           'selected_and_test_satisfies_n': sum(not r['test_violation'] for r in usable),
                           'conditional_test_metrics': metrics([r['test'] for r in usable]),
                           'conditional_step': describe([r['selected_step'] for r in usable]),
                           'conditional_old_slack_pp': describe([r['test_old_slack_pp'] for r in usable])})
        pair_rows = []
        for seed in SEEDS:
            a, b = [next(r for r in seed_units if r['seed'] == seed and r['threshold_pct'] == threshold
                         and r['method'] == method) for method in METHODS]
            joint = a['validation_feasible'] and b['validation_feasible']
            pair_rows.append({'seed': seed, 'joint_validation_feasible': joint,
                              'sgd_validation_feasible': a['validation_feasible'],
                              'er_validation_feasible': b['validation_feasible'],
                              'sgd_test_violation': a['test_violation'], 'er_test_violation': b['test_violation'],
                              'er_minus_sgd_pp': {k: b['test'][k] - a['test'][k] for k in
                                                  ('old_pct', 'current_pct', 'all_pct')} if joint else None})
        available = [r for r in pair_rows if r['joint_validation_feasible']]
        paired.append({'threshold_pct': threshold, 'total_paired_seeds': 10,
                       'joint_validation_feasible_n': len(available),
                       'joint_validation_feasible_seeds': [r['seed'] for r in available],
                       'conditional_er_minus_sgd_pp': metrics([r['er_minus_sgd_pp'] for r in available]),
                       'seed_values': pair_rows, 'test_violations_excluded': False})
    for method in METHODS:
        for anchor in ('pre', 'terminal', 'actual'):
            rows = [r for r in anchor_units if r['method'] == method and r['anchor'] == anchor]
            require([r['seed'] for r in rows] == SEEDS, 'anchor seed coverage')
            anchor_groups.append({'method': method, 'anchor': anchor, 'n_seeds': 10,
                                  'test_metrics': metrics([r['test'] for r in rows])})
    return {'status': 'passed_independent_retention_trajectory_audit',
            'manifest_sha256': manifest_sha, 'auditor_sha256': sha(__file__),
            'selection_sha256': seal['selection_sha256'], 'selection_seal_sha256': seal_sha,
            'histories': 20, 'optimizer_updates': total_updates, 'validation_endpoints': 260,
            'test_endpoints': total_test, 'image_evaluations': (260 + total_test) * 1000,
            'endpoint_tensors_compared': tensor_checks, 'endpoint_proofs': endpoint_proofs,
            'seed_units': seed_units, 'groups': groups, 'paired': paired,
            'anchor_units': anchor_units, 'anchor_groups': anchor_groups,
            'validation_units': validation_units,
            'scope': 'Descriptive frozen-history, fixed-image-split, finite-grid checkpoint selection. '
                     'All test violations retained. No p-values, CIs, intrinsic-plasticity claim, '
                     'continuous-time retention guarantee, or verification of every SGD update. '
                     'NumPy counts/mappings/statistics; torch only for CPU checkpoint deserialization.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / OUTPUT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output already exists')
    require(not args.output.resolve().is_relative_to(args.input.resolve()), 'audit output must be outside raw directory')
    report = audit(args.input, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in
                      ('status', 'histories', 'validation_endpoints', 'test_endpoints')}))
