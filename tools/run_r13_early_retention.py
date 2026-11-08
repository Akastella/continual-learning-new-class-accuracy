"""Frozen policy sensitivity study: validation seals before new test inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

START = time.monotonic()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.r9_followup import engine
from experiments.r9_retention_trajectory import core as training
from experiments.r13_early_retention import core

MANIFEST = 'configs/r13_early_retention/manifest.json'
OUTPUT = 'results/r13_early_retention_v1'
ORIGINAL = 'configs/r9_followup/manifest.json'
OLD_MANIFEST = 'configs/r9_retention_trajectory/manifest.json'
OLD = 'results/r9_retention_trajectory'
OVERLAP = [0, 1, 2, 4, 8, 16]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def runs():
    rows = [r for r in read(ROOT / ORIGINAL)['runs']
            if r['id'].startswith('split_cifar10_classil_')]
    assert [r['id'] for r in rows] == [f'split_cifar10_classil_{m}_s{s}'
        for s in range(201, 211) for m in ['sgd', 'er']]
    return rows


def settings(image_key, label_key):
    return dict(kind='posthoc_policy_sensitivity_on_new_cifar102_images',
        seeds=list(range(201, 211)), methods=['sgd', 'er'], task=4,
        sparse_steps=list(core.SPARSE), dense_steps=list(core.DENSE),
        policies=list(core.POLICIES), constraints=list(core.CONSTRAINTS),
        validation_split='configs/r9_retention_trajectory/split.json',
        validation_data='data/r9_cifar101/cifar10.1_v6_data.npy',
        validation_labels='data/r9_cifar101/cifar10.1_v6_labels.npy',
        test_npz='data/r13_cifar102/cifar102_test.npz',
        test_images_key=image_key, test_labels_key=label_key,
        history_manifest=ORIGINAL, trajectory_manifest=OLD_MANIFEST,
        trajectory_root=OLD, output=OUTPUT, limit_seconds=3300,
        evaluation_batch=512, expected_optimizer_updates=320,
        reused_validation_endpoints=260, new_validation_endpoints=240,
        maximum_test_endpoints=140)


def input_names():
    names = {ORIGINAL, OLD_MANIFEST,
        'configs/r9_retention_trajectory/split.json',
        'data/r9_cifar101/cifar10.1_v6_data.npy',
        'data/r9_cifar101/cifar10.1_v6_labels.npy',
        'data/r13_cifar102/cifar102_test.npz',
        'third_party/r13_cifar102_reference/SOURCE.json',
        'third_party/r13_cifar102_reference/README.md',
        'refine-logs/r13-early-retention/DATA_PREFLIGHT.md',
        'refine-logs/r13-early-retention/DESIGN_REVIEW.md',
        'refine-logs/r13-early-retention/FEASIBILITY.md',
        'refine-logs/r13-early-retention/INVENTORY.json',
        'audit/r9/RETENTION_TRAJECTORY_INDEPENDENT.json',
        'audit/r9/RETENTION_TRAJECTORY_FORWARD.json',
        'audit/r13_early_retention/trainonly_resource/RESOURCE.json',
        OLD + '/artifacts.json', OLD + '/complete.json', OLD + '/watchdog.json'}
    names.update('data/r9_official/cifar-10-batches-py/data_batch_' + str(i)
                 for i in range(1, 6))
    for r in runs():
        names.add(r['config_path'])
        names.update(r['output'] + '/' + x for x in ['pre_task_004.pt',
            'probe_task_004.pt', 'post_task_004.pt', 'result.json', 'data_manifest.json'])
        names.update(OLD + '/' + r['id'] + '/' + x
                     for x in ['trajectory.json', 'validation.npz'])
        names.update(OLD + '/' + r['id'] + f'/checkpoints/step_{s:06d}.pt'
                     for s in core.SPARSE)
    return sorted(names)


def validate(manifest):
    s = manifest['settings']
    assert s == settings(s['test_images_key'], s['test_labels_key'])
    assert set(manifest['input_sha256']) == set(input_names())
    for group in ['source_sha256', 'input_sha256']:
        for name, h in manifest[group].items():
            p = (ROOT / name).resolve()
            if not p.is_relative_to(ROOT) or sha(p) != h:
                raise ValueError('Frozen source/input drift: ' + name)
    assert sha(ROOT / ORIGINAL) == 'bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e'
    assert sha(ROOT / OLD_MANIFEST) == '60ef2ac448e9e77c3f5285510b3a0ba937c198541f77aa73a6a3f74491261aa0'
    frozen = datetime.fromisoformat(manifest['frozen_utc'])
    assert frozen.tzinfo is not None and frozen <= datetime.now(timezone.utc)


def initialize_environment():
    if not sys.flags.no_user_site:
        raise RuntimeError('Use the locked Python with -s')
    locked = dict(x.split('==', 1) for x in (ROOT / 'requirements-r9-windows-lock.txt')
                  .read_text(encoding='utf-8-sig').splitlines()
                  if '==' in x and not x.startswith('#'))
    for name in ['torch', 'torchvision', 'numpy', 'pillow']:
        if importlib.metadata.version(name) != locked[name]:
            raise ValueError('Environment mismatch: ' + name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        threads=2, deterministic=True, tf32=False, batch_eval=512,
        no_user_site=True, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def normalize(images):
    assert images.dtype == np.uint8 and images.shape[1:] == (32, 32, 3)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    return ((x - torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)) /
            torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1))


def inference(model, x):
    model.eval()
    before = engine.state_hash(model)
    with torch.no_grad():
        result = np.concatenate([model(x[i:i+512].cuda()).cpu().numpy()
                                 for i in range(0, len(x), 512)])
    if engine.state_hash(model) != before:
        raise ValueError('Inference mutated model')
    return result


def mapped_labels(labels, cfg):
    order = np.random.default_rng(cfg['class_order_seed']).permutation(10)
    return np.argsort(order)[labels].astype(np.int64)


def state(path, kind):
    s = torch.load(path, map_location='cpu', weights_only=True)
    return s if kind == 'actual' else s['model']


def record(key, path, model_hash, origin, score):
    actual = key == 'actual'
    return dict(key=key, step=628 if actual else int(key.split('_')[1]),
        kind='actual' if actual else 'probe', checkpoint=path.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(path), state_sha256=model_hash, origin=origin, scores=score)


def worker():
    m = read(ROOT / MANIFEST)
    validate(m)
    out = ROOT / OUTPUT
    save(out / 'environment.json', initialize_environment())
    s = m['settings']
    split = read(ROOT / s['validation_split'])
    ids = np.array(split['validation_indices'], dtype=np.int64)
    pixels = np.load(ROOT / s['validation_data'], allow_pickle=False)
    original_labels = np.load(ROOT / s['validation_labels'], allow_pickle=False)
    vx = normalize(pixels)[ids]
    choices, validation_hashes = {}, {}
    updates = new_validation = reused_validation = 0
    for i, r in enumerate(runs(), 1):
        cfg = read(ROOT / r['config_path'])
        rd = out / r['id']
        rd.mkdir()
        old_rd = ROOT / OLD / r['id']
        old_trace = read(old_rd / 'trajectory.json')
        assert sha(old_rd / 'validation.npz') == old_trace['validation_npz_sha256']
        labels = mapped_labels(original_labels, cfg)[ids]
        arrays = {'indices': ids, 'labels': labels}
        records = []
        old_records = {v['step']: v for v in old_trace['validation']}
        with np.load(old_rd / 'validation.npz', allow_pickle=False) as old:
            assert np.array_equal(old['indices'], ids) and np.array_equal(old['labels'], labels)
            for step in core.SPARSE:
                key = 'step_' + str(step)
                arrays['logits_' + key] = old['logits_' + str(step)].copy()
                path = ROOT / old_records[step]['checkpoint']
                saved = torch.load(path, map_location='cpu', weights_only=True)
                rec = record(key, path, saved['state_sha256'], 'r9_reused',
                             core.scores(arrays['logits_' + key], labels))
                assert rec['scores'] == old_records[step]['scores']
                assert rec['checkpoint_sha256'] == old_records[step]['checkpoint_sha256']
                records.append(rec)
                reused_validation += 1
        tx, ty, proxy, provenance = training.load_train_only(ROOT / 'data', cfg, 4)
        del proxy
        archived = read(ROOT / r['output'] / 'data_manifest.json')['tasks'][4]['splits']['train']
        assert all(provenance[k] == archived[k] for k in ['official_indices', 'images_sha256', 'labels_sha256'])
        initial = torch.load(ROOT / r['output'] / 'pre_task_004.pt', map_location='cpu', weights_only=True)
        model = engine.local_model(cfg, cfg['init_seed'], 'cuda', initial)
        matched = []

        def callback(step, net, path):
            nonlocal new_validation
            if step in OVERLAP:
                expected = state(ROOT / old_records[step]['checkpoint'], 'probe')
                if not all(torch.equal(v.cpu(), expected[k]) for k, v in net.state_dict().items()):
                    raise ValueError('Old prefix state differs: ' + r['id'] + '/' + str(step))
                matched.append(step)
            else:
                key = 'step_' + str(step)
                z = inference(net, vx)
                arrays['logits_' + key] = z
                records.append(record(key, path, engine.state_hash(net), 'r13_new', core.scores(z, labels)))
                new_validation += 1

        summary = training.train_trajectory(model, tx, ty, cfg, 4, rd / 'checkpoints',
            list(core.DENSE), callback=callback, stop_step=16)
        assert matched == OVERLAP and summary['optimizer_steps_executed'] == 16
        assert summary['batch_order_sha256'] == old_trace['summary']['batch_order_sha256']
        for item in summary['checkpoints']:
            item['path'] = (rd / 'checkpoints' / item['path']).relative_to(ROOT).as_posix()
        updates += 16
        actual_path = ROOT / r['output'] / 'post_task_004.pt'
        model.load_state_dict(state(actual_path, 'actual'))
        z = inference(model, vx)
        arrays['logits_actual'] = z
        records.append(record('actual', actual_path, engine.state_hash(model), 'r13_new', core.scores(z, labels)))
        new_validation += 1
        records.sort(key=lambda row: (row['step'], row['kind'] == 'actual'))
        np.savez_compressed(rd / 'validation.npz', **arrays)
        save(rd / 'validation.json', dict(run_id=r['id'], records=records, npz_sha256=sha(rd / 'validation.npz')))
        save(rd / 'training.json', dict(summary=summary, matched_steps=matched, train_provenance=provenance))
        choices[r['id']] = {c: {p: core.choose(records, p, c) for p in core.POLICIES}
                            for c in core.CONSTRAINTS}
        for name in ['validation.npz', 'validation.json', 'training.json']:
            validation_hashes[(rd / name).relative_to(out).as_posix()] = sha(rd / name)
        del model, tx, ty, arrays
        print(json.dumps(dict(stage='validation', histories=i, updates=updates)), flush=True)
    assert (updates, new_validation, reused_validation) == (320, 240, 260)
    save(out / 'selection.json', choices)
    save(out / 'selection_seal.json', dict(created_utc=utc(), selection_sha256=sha(out / 'selection.json'),
        validation_files_sha256=validation_hashes, rule='protocol/R13_EARLY_RETENTION_PROTOCOL.md'))
    seal = sha(out / 'selection_seal.json')
    save(out / 'test_started.json', dict(utc=utc(), selection_seal_sha256=seal))
    with np.load(ROOT / s['test_npz'], allow_pickle=False) as data:
        test_x = normalize(data[s['test_images_key']])
        test_labels = data[s['test_labels_key']].astype(np.int64)
    endpoints = 0
    for i, r in enumerate(runs(), 1):
        rd = out / r['id']
        cfg = read(ROOT / r['config_path'])
        records = {v['key']: v for v in read(rd / 'validation.json')['records']}
        chosen = {'step_0', 'step_628', 'actual'} | {
            k for policy in choices[r['id']].values() for k in policy.values() if k is not None}
        assert len(chosen) <= 7
        labels = mapped_labels(test_labels, cfg)
        arrays = {'indices': np.arange(len(test_labels), dtype=np.int64), 'labels': labels}
        scored = []
        for key in sorted(chosen, key=lambda k: (records[k]['step'], records[k]['kind'] == 'actual')):
            rec = records[key]
            model = engine.local_model(cfg, cfg['init_seed'], 'cuda', state(ROOT / rec['checkpoint'], rec['kind']))
            assert engine.state_hash(model) == rec['state_sha256']
            z = inference(model, test_x)
            arrays['logits_' + key] = z
            scored.append(dict(rec, scores=core.scores(z, labels)))
            del model
        np.savez_compressed(rd / 'test.npz', **arrays)
        save(rd / 'test.json', dict(run_id=r['id'], records=scored, selection_seal_sha256=seal,
                                  npz_sha256=sha(rd / 'test.npz')))
        endpoints += len(scored)
        print(json.dumps(dict(stage='test', histories=i, endpoints=endpoints)), flush=True)
    assert endpoints <= 140
    validate(m)
    assert sha(out / 'selection_seal.json') == seal
    files = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
             if p.is_file() and p.name != 'console.log'}
    save(out / 'artifacts.json', files)
    save(out / 'complete.json', dict(status='complete', histories=20, optimizer_updates=updates,
        reused_validation_endpoints=reused_validation, new_validation_endpoints=new_validation,
        validation_endpoints=500, test_endpoints=endpoints, test_images=len(test_labels),
        artifacts_sha256=sha(out / 'artifacts.json'), manifest_sha256=sha(ROOT / MANIFEST), finished_utc=utc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except BaseException:
            save(ROOT / OUTPUT / 'failure.json', dict(utc=utc(), traceback=traceback.format_exc().replace(str(ROOT), '[REPO]')))
            raise
        return
    m = read(ROOT / MANIFEST)
    validate(m)
    out = ROOT / OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'started.json', dict(utc=utc(), manifest_sha256=sha(ROOT / MANIFEST)))
    shutil.copy2(ROOT / MANIFEST, out / 'frozen_manifest.json')
    for name in m['source_sha256']:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    env = dict(os.environ)
    for key in ['PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP']:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    status, code = 'failed', None
    with (out / 'console.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-s', '-X', 'utf8', str(Path(__file__).resolve()), '--worker'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            code = process.wait(timeout=max(1, 3300 - (time.monotonic() - START)))
            status = 'complete' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = 'timed_out'
    save(out / 'watchdog.json', dict(status=status, exit_code=code,
        elapsed_seconds=time.monotonic()-START, limit_seconds=3300))
    if status != 'complete':
        raise RuntimeError('Incomplete R13 matrix: ' + status)
    print(json.dumps(read(out / 'watchdog.json')))


if __name__ == '__main__':
    main()
