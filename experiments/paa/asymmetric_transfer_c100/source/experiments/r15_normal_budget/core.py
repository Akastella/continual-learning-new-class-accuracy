"""Long-budget CIFAR-100 engine; all test access is in the evaluation phase.

CUDA augmentation is a new local operator, not bitwise equivalent to PIL:
zero-pad/crop, flip, continuous FP32 brightness with clipping, normalize.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import shutil
import sys
import tarfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.r9_external import data
from experiments.r9_external.model import make_model

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'r15_normal_budget_v1'
SETTINGS = {
    'seeds': [3101, 3102, 3103, 3104, 3105], 'methods': ['sgd', 'er'],
    'tasks': 10, 'diagnostic_tasks': [6, 7, 8, 9], 'epochs_first': 200,
    'epochs_later': 70, 'probe_epochs': 70, 'batch_current': 64,
    'memory_per_old_class': 20, 'replay_allocation': 'fixed_1_to_1',
    'lr': .1, 'momentum': .9, 'nesterov': False,
    'weight_decay_first': .0005, 'weight_decay_later': .0002,
    'milestones_first': [60, 120, 170], 'milestones_later': [30, 50],
    'gamma': .1, 'reset_optimizer_each_branch': True,
    'new_head_rows_reset': True, 'dtype': 'float32', 'deterministic': True,
    'tf32': False, 'threads': 2, 'evaluation_batch': 256, 'memory_format': 'channels_last',
    'training_autocast': 'bfloat16', 'evaluation_autocast': 'disabled', 'gradient_scaler': False,
    'augmentation': 'cuda_zero_pad4_crop32_flip0.5_continuous_brightness63over255_clip01_normalize',
    'test_access': 'after_all_jobs_training_complete',
}
ARCHIVE_SHA256 = '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
TEST_SHA256 = '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864'
REQUIRED_SOURCES = {
    'experiments/r15_normal_budget/__init__.py', 'experiments/r15_normal_budget/core.py',
    'tools/run_r15_normal_budget.py', 'experiments/r9_external/data.py',
    'experiments/r9_external/model.py', 'third_party/pycil_reference/cifar_resnet.py',
    'third_party/pycil_reference/data.py', 'third_party/pycil_reference/LICENSE',
    'requirements-r9-windows-lock.txt',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_manifest(path):
    manifest = read_json(path)
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'frozen':
        raise ValueError('Expected frozen R15 manifest')
    if json.dumps(manifest.get('settings'), sort_keys=True) != json.dumps(SETTINGS, sort_keys=True):
        raise ValueError('Frozen settings differ from engine contract')
    frozen = datetime.fromisoformat(manifest['frozen_utc'])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError('Invalid freeze timestamp')
    sources = manifest.get('sources_sha256', {})
    if not REQUIRED_SOURCES.issubset(sources) or not any(p.startswith('protocol/') for p in sources):
        raise ValueError('Missing engine/reference/protocol source identity')
    for name, expected in sources.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or data.file_digest(source) != expected:
            raise ValueError('Frozen source hash mismatch: ' + name)
    if manifest.get('dataset', {}).get('archive_sha256') != ARCHIVE_SHA256:
        raise ValueError('Dataset identity differs')
    if type(manifest.get('job_timeout_seconds')) is not int or manifest['job_timeout_seconds'] <= 0:
        raise ValueError('Positive integer job timeout required')
    return manifest


def environment():
    if not sys.flags.no_user_site or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Use locked Python -s and CUBLAS_WORKSPACE_CONFIG=:4096:8')
    locked = dict(line.split('==', 1) for line in (ROOT/'requirements-r9-windows-lock.txt')
                  .read_text(encoding='utf-8-sig').splitlines() if '==' in line and not line.startswith('#'))
    for name in ('torch', 'torchvision', 'numpy', 'pillow'):
        if importlib.metadata.version(name) != locked[name]:
            raise RuntimeError('Environment lock mismatch: ' + name)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    return {'utc': utc(), 'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'deterministic': True, 'tf32': False, 'threads': 2, 'dtype': 'float32',
            'memory_format': SETTINGS['memory_format'],
            'training_autocast': SETTINGS['training_autocast'],
            'evaluation_autocast': SETTINGS['evaluation_autocast'],
            'gradient_scaler': False, 'no_user_site': True,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}


def augment(images, key):
    """Pure local-generator batched augmentation; no global RNG consumption."""
    if images.dtype != torch.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != (32, 32, 3):
        raise ValueError('Expected uint8 NHWC CIFAR images')
    n, device = len(images), images.device
    generator = torch.Generator(device=device).manual_seed(int(key))
    offsets = torch.randint(0, 9, (n, 2), generator=generator, device=device)
    flips = torch.rand(n, generator=generator, device=device) < .5
    brightness = 1 - 63/255 + torch.rand(n, generator=generator, device=device) * (126/255)
    x = images.permute(0, 3, 1, 2).to(torch.float32).div_(255)
    x = F.pad(x, (4, 4, 4, 4), value=0.)
    yy = offsets[:, 0, None, None] + torch.arange(32, device=device)[None, :, None]
    xx = offsets[:, 1, None, None] + torch.arange(32, device=device)[None, None, :]
    x = x.permute(0, 2, 3, 1)[torch.arange(n, device=device)[:, None, None], yy, xx]
    x = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    x = torch.where(flips[:, None, None, None], x.flip(-1), x)
    x.mul_(brightness[:, None, None, None]).clamp_(0., 1.)
    return x.sub_(torch.tensor(data.MEAN, device=device).view(1, 3, 1, 1)).div_(
        torch.tensor(data.STD, device=device).view(1, 3, 1, 1))


def augmentation_key(seed, task, epoch, batch, source):
    return data.event_seed(seed, 'r15_cuda_augmentation_v1', task, epoch, batch, source)


def reset_new_rows(model, initial, task):
    rows = slice(10*task, 10*(task+1))
    with torch.no_grad():
        model.head.weight[rows].copy_(initial['head.weight'][rows].to('cuda'))
        model.head.bias[rows].copy_(initial['head.bias'][rows].to('cuda'))


def save_checkpoint(path, model, metadata):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    digest = state_digest(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.pt.partial')
    torch.save({'model': state, 'state_sha256': digest, 'paramhash': digest, **metadata}, temporary)
    temporary.replace(path)
    return {'path': path.name, 'file_sha256': data.file_digest(path), 'state_sha256': digest, **metadata}


def restore(path, model):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if state_digest(saved['model']) != saved['state_sha256']:
        raise ValueError('Checkpoint state hash mismatch')
    model.load_state_dict(saved['model'])
    return saved


def train_branch(model, dataset, images_gpu, labels_gpu, seed, task, method, branch,
                 epochs, output, deadline, first_task=False):
    """Returns every epoch's loss/exposure/order identity; no evaluation data."""
    s = SETTINGS
    optimizer = torch.optim.SGD(model.parameters(), lr=s['lr'], momentum=s['momentum'],
                                weight_decay=s['weight_decay_first' if first_task else 'weight_decay_later'],
                                nesterov=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=s['milestones_first' if first_task else 'milestones_later'], gamma=s['gamma'])
    allocation = 'fixed' if method == 'er' else 'current'
    start_hash = state_digest(model.state_dict())
    started = time.perf_counter()
    records = []
    model.train()
    with (Path(output)/'epochs.jsonl').open('a', encoding='utf-8') as journal:
        for epoch in range(epochs):
            if time.monotonic() > deadline:
                raise TimeoutError('Train job deadline exceeded; partial run is not an endpoint')
            rows, plan = data.epoch_plan(dataset.labels, seed, task, epoch, 20, allocation)
            lr = optimizer.param_groups[0]['lr']
            sums = torch.zeros(2, device='cuda', dtype=torch.float64)
            nc = nr = 0
            keys = []
            before = time.perf_counter()
            for row in rows:
                if time.monotonic() > deadline:
                    raise TimeoutError('Train job deadline exceeded')
                batch_images, batch_labels = [], []
                for source in ('current', 'replay'):
                    ids = row[source+'_indices']
                    if not len(ids):
                        continue
                    ix = torch.as_tensor(ids, device='cuda')
                    key = augmentation_key(seed, task, epoch, row['step'], source)
                    keys.append(key)
                    batch_images.append(augment(images_gpu[ix], key))
                    batch_labels.append(labels_gpu[ix])
                x = torch.cat(batch_images) if len(batch_images) > 1 else batch_images[0]
                y = torch.cat(batch_labels) if len(batch_labels) > 1 else batch_labels[0]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=SETTINGS['training_autocast'] == 'bfloat16'):
                    z = model(x)[:, :10*(task+1)]
                    loss = F.cross_entropy(z, y)
                loss.backward()
                optimizer.step()
                sums[0] += loss.detach().double()*len(y)
                sums[1] += (z.detach().argmax(1) == y).sum()
                nc += len(row['current_indices']); nr += len(row['replay_indices'])
            scheduler.step()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(sums).all()):
                raise FloatingPointError('Nonfinite epoch loss/count; no endpoint accepted')
            gradients_finite = bool(torch.stack([torch.isfinite(p.grad).all()
                for p in model.parameters() if p.grad is not None]).all())
            if not gradients_finite:
                raise FloatingPointError('Nonfinite epoch-final gradients')
            row = {'seed': seed, 'method': method, 'branch': branch, 'task': task,
                'epoch': epoch+1, 'lr': lr, 'next_lr': optimizer.param_groups[0]['lr'],
                'optimizer_steps': len(rows), 'current_presentations': nc, 'replay_presentations': nr,
                'train_ce': float(sums[0].cpu())/(nc+nr), 'train_correct': int(sums[1].cpu()),
                'train_n': nc+nr, 'seconds': time.perf_counter()-before,
                'epoch_final_gradients_finite': gradients_finite,
                'current_order_sha256': plan['current_order_sha256'],
                'memory_indices_sha256': plan['memory_indices_sha256'],
                'replay_order_sha256': plan['replay_order_sha256'],
                'augmentation_keys_sha256': data.array_digest(np.asarray(keys, dtype=np.int64))}
            journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
            records.append(row)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite final parameters')
    return {'start_state_sha256': start_hash, 'end_state_sha256': state_digest(model.state_dict()),
        'epochs': epochs, 'optimizer_steps': sum(r['optimizer_steps'] for r in records),
        'current_presentations': sum(r['current_presentations'] for r in records),
        'replay_presentations': sum(r['replay_presentations'] for r in records),
        'wall_seconds': time.perf_counter()-started, 'epoch_records': records}


def load_gpu_train(seed):
    folder = ROOT/'data/r9_external'
    if data.file_digest(folder/data.ARCHIVE_NAME) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 mismatch')
    dataset = data.load_train(folder)
    _, inverse = data.class_mapping(seed)
    images = torch.from_numpy(dataset.images.copy()).to('cuda')
    labels = torch.as_tensor(inverse[dataset.labels], dtype=torch.long, device='cuda')
    return dataset, images, labels


def train_job(manifest_path, seed, method, output):
    manifest = validate_manifest(manifest_path)
    if seed not in SETTINGS['seeds'] or method not in SETTINGS['methods']:
        raise ValueError('Job outside frozen matrix')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output/'started.json', {'utc': utc(), 'seed': seed, 'method': method,
        'manifest_sha256': data.file_digest(manifest_path), 'test_access': False})
    try:
        write_json(output/'environment.json', environment())
        dataset, images, labels = load_gpu_train(seed)
        write_json(output/'train_provenance.json', dataset.provenance)
        model = make_model(seed+10000, 'cuda').to(memory_format=torch.channels_last)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_checkpoint(output/'checkpoints/initial.pt', model, {'branch': 'initial', 'task': -1, 'seed': seed})
        branches = []
        deadline = started + manifest['job_timeout_seconds']
        # Complete the actual history before auxiliary branches.
        for task in range(10):
            reset_new_rows(model, initial, task)
            save_checkpoint(output/f'checkpoints/pre_task_{task:02d}.pt', model,
                {'branch': 'pre', 'task': task, 'seed': seed, 'new_rows_reset': True})
            epochs = SETTINGS['epochs_first' if task == 0 else 'epochs_later']
            record = train_branch(model, dataset, images, labels, seed, task, method, 'actual',
                                  epochs, output, deadline, first_task=task == 0)
            record.update(branch='actual', task=task)
            record['checkpoint'] = save_checkpoint(output/f'checkpoints/actual_task_{task:02d}.pt',
                model, {'branch': 'actual', 'task': task, 'seed': seed, 'epochs': epochs})
            branches.append(record)
            write_json(output/'progress.json', {'utc': utc(), 'branches': branches, 'test_access': False})
        for task in SETTINGS['diagnostic_tasks']:
            if method == 'sgd':
                actual = next(r for r in branches if r['branch'] == 'actual' and r['task'] == task)
                pre = torch.load(output/f'checkpoints/pre_task_{task:02d}.pt', map_location='cpu', weights_only=True)
                if actual['start_state_sha256'] != pre['state_sha256']:
                    raise ValueError('SGD probe identity has different pre-state')
                branches.append({'branch': 'probe', 'task': task, 'reused_actual': True,
                    'identity_reason': 'same pre-state, current-only loss, order, augmentation, optimizer reset and schedule',
                    'start_state_sha256': actual['start_state_sha256'],
                    'end_state_sha256': actual['end_state_sha256'], 'checkpoint': actual['checkpoint'],
                    'optimizer_steps_executed': 0, 'logical_epochs': 70})
            else:
                restore(output/f'checkpoints/pre_task_{task:02d}.pt', model)
                record = train_branch(model, dataset, images, labels, seed, task, 'sgd', 'probe',
                    70, output, deadline)
                record.update(branch='probe', task=task)
                record['checkpoint'] = save_checkpoint(output/f'checkpoints/probe_task_{task:02d}.pt',
                    model, {'branch': 'probe', 'task': task, 'seed': seed, 'epochs': 70})
                branches.append(record)
            if method == 'er':
                # The preceding SGD job executed the identical fresh operator.
                # Content-addressed copying is explicit reuse, never new training.
                peer = output.parent/f'seed{seed}_sgd'
                peer_marker = peer/'training_complete.json'
                peer_completed = read_json(peer_marker)
                if (peer_completed['seed'] != seed or peer_completed['method'] != 'sgd'
                        or peer_completed['manifest_sha256'] != data.file_digest(manifest_path)):
                    raise ValueError('Fresh-reuse source belongs to another experiment')
                source_record = next(r for r in peer_completed['branches']
                                     if r['branch'] == 'fresh' and r['task'] == task)
                if source_record['start_state_sha256'] != state_digest(initial):
                    raise ValueError('Fresh-reuse initialization differs')
                source = peer/'checkpoints'/source_record['checkpoint']['path']
                if data.file_digest(source) != source_record['checkpoint']['file_sha256']:
                    raise ValueError('Fresh-reuse source checkpoint changed')
                saved = restore(source, model)
                if saved['state_sha256'] != source_record['end_state_sha256']:
                    raise ValueError('Fresh-reuse endpoint identity differs')
                destination = output/f'checkpoints/fresh_task_{task:02d}.pt'
                if destination.exists(): raise FileExistsError(destination)
                shutil.copy2(source, destination)
                if data.file_digest(destination) != source_record['checkpoint']['file_sha256']:
                    raise ValueError('Fresh-reuse copy changed bytes')
                record = {'branch': 'fresh', 'task': task, 'reused_fresh': True,
                    'optimizer_steps_executed': 0, 'logical_epochs': 70,
                    'start_state_sha256': source_record['start_state_sha256'],
                    'end_state_sha256': source_record['end_state_sha256'],
                    'checkpoint': source_record['checkpoint'],
                    'source_job': peer.name, 'source_training_complete_sha256': data.file_digest(peer_marker),
                    'source_checkpoint_sha256': data.file_digest(source),
                    'identity_reason': 'same frozen manifest, seed, initial state, current order and augmentation keys; actual SGD-job fresh endpoint reused'}
            else:
                model.load_state_dict(initial)
                record = train_branch(model, dataset, images, labels, seed, task, 'sgd', 'fresh',
                    70, output, deadline)
                record.update(branch='fresh', task=task)
                record['checkpoint'] = save_checkpoint(output/f'checkpoints/fresh_task_{task:02d}.pt',
                    model, {'branch': 'fresh', 'task': task, 'seed': seed, 'epochs': 70})
            branches.append(record)
            write_json(output/'progress.json', {'utc': utc(), 'branches': branches, 'test_access': False})
        validate_manifest(manifest_path)
        write_json(output/'training_complete.json', {'utc': utc(), 'seed': seed, 'method': method,
            'manifest_sha256': data.file_digest(manifest_path), 'branches': branches,
            'test_access': False, 'wall_seconds': time.monotonic()-started,
            'epochs_journal_sha256': data.file_digest(output/'epochs.jsonl'),
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()})
    except BaseException as error:
        write_json(output/'failure.json', {'utc': utc(), 'stage': 'train',
            'error': type(error).__name__+': '+str(error), 'wall_seconds': time.monotonic()-started})
        raise


def evaluate_job(manifest_path, output, training_seal):
    validate_manifest(manifest_path)
    output, training_seal = Path(output), Path(training_seal)
    seal = read_json(training_seal)
    if seal.get('manifest_sha256') != data.file_digest(manifest_path) or len(seal.get('jobs', [])) != 10:
        raise ValueError('Missing complete global training seal')
    expected_ids = {f'seed{s}_{m}' for s in SETTINGS['seeds'] for m in SETTINGS['methods']}
    if {j.get('id') for j in seal['jobs']} != expected_ids:
        raise ValueError('Global seal does not contain the exact frozen job matrix')
    for job in seal['jobs']:
        expected = f"jobs/{job['id']}/training_complete.json"
        if job['training_complete_path'] != expected:
            raise ValueError('Invalid training marker path in global seal')
        p = training_seal.parent/expected
        if data.file_digest(p) != job['sha256']:
            raise ValueError('Training marker changed after global seal')
        marker = read_json(p)
        if marker['manifest_sha256'] != data.file_digest(manifest_path):
            raise ValueError('Global seal mixes manifest identities')
    completed = read_json(output/'training_complete.json')
    if completed['manifest_sha256'] != data.file_digest(manifest_path):
        raise ValueError('Job belongs to another manifest')
    environment()
    directory = output/'evaluation'; directory.mkdir(exist_ok=False)
    write_json(directory/'started.json', {'utc': utc(), 'global_training_seal_sha256': data.file_digest(training_seal),
        'test_role': 'reused official CIFAR100 test; post-hoc longer-budget evaluation'})
    archive = ROOT/'data/r9_external'/data.ARCHIVE_NAME
    with tarfile.open(archive, 'r:gz') as tar:
        payload = tar.extractfile('cifar-100-python/test').read()
    if hashlib.sha256(payload).hexdigest() != TEST_SHA256:
        raise ValueError('Official test payload identity mismatch')
    raw = pickle.loads(payload, encoding='bytes')
    images = np.asarray(raw[b'data']).reshape(-1,3,32,32).transpose(0,2,3,1)
    original_labels = np.asarray(raw[b'fine_labels'], dtype=np.int64)
    if images.shape != (10000,32,32,3) or images.dtype != np.uint8:
        raise ValueError('Official test schema mismatch')
    seed = completed['seed']; order, inverse = data.class_mapping(seed)
    mapped = inverse[original_labels]
    model = make_model(seed+10000, 'cuda').to(memory_format=torch.channels_last)
    results = []
    for branch in completed['branches']:
        task, name = branch['task'], branch['branch']
        if name != 'actual' and task not in SETTINGS['diagnostic_tasks']:
            continue
        checkpoint = output/'checkpoints'/branch['checkpoint']['path']
        if data.file_digest(checkpoint) != branch['checkpoint']['file_sha256']:
            raise ValueError('Endpoint checkpoint file changed')
        if branch.get('reused_actual'):
            # Same state and complete test operator: preserve the actual NPZ bytes.
            source = directory/f'actual_task_{task:02d}.npz'
            actual = next(r for r in results if r['branch']=='actual' and r['task']==task)
            if data.file_digest(source) != actual['sha256']:
                raise ValueError('Actual inference artifact changed before identity reuse')
            path = directory/f'{name}_task_{task:02d}.npz'
            shutil.copy2(source,path)
            results.append({'branch':name,'task':task,'path':path.name,
                'sha256':data.file_digest(path),'checkpoint_sha256':data.file_digest(checkpoint),
                'state_sha256':branch['end_state_sha256'],'n':10000,'inference_reused_from':source.name})
            continue
        restore(checkpoint, model); model.eval()
        before = state_digest(model.state_dict())
        blocks = []
        with torch.no_grad():
            for offset in range(0,10000,256):
                x = torch.from_numpy(images[offset:offset+256].copy()).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255
                x = (x-torch.tensor(data.MEAN,device='cuda').view(1,3,1,1))/torch.tensor(data.STD,device='cuda').view(1,3,1,1)
                # Evaluation deliberately uses FP32 to avoid BF16 logit quantization.
                blocks.append(model(x).cpu().numpy())
        logits = np.concatenate(blocks).astype(np.float32)
        if not np.isfinite(logits).all() or state_digest(model.state_dict()) != before:
            raise ValueError('Invalid or mutating inference')
        path = directory/f'{name}_task_{task:02d}.npz'
        np.savez(path, logits=logits, indices=np.arange(10000,dtype=np.int64),
            original_labels=original_labels, mapped_labels=mapped, class_order=order,
            predictions_seen=logits[:,:10*(task+1)].argmax(1), task=np.int64(task))
        results.append({'branch': name, 'task': task, 'path': path.name,
            'sha256': data.file_digest(path), 'checkpoint_sha256': data.file_digest(checkpoint),
            'state_sha256': before, 'n': 10000})
    validate_manifest(manifest_path)
    write_json(output/'result.json', {'status': 'complete', 'utc': utc(), 'seed': seed,
        'method': completed['method'], 'test_payload_sha256': TEST_SHA256,
        'global_training_seal_sha256': data.file_digest(training_seal),
        'manifest_sha256': data.file_digest(manifest_path), 'endpoints': results})


def witness(output):
    """Official train-only two-epoch throughput and deterministic operator checks."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        env = environment()
        seed = 3199
        dataset, images, labels = load_gpu_train(seed)
        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        key = augmentation_key(seed,9,0,0,'current')
        a, b = augment(images[:64],key), augment(images[:64],key)
        if not torch.equal(a,b) or not torch.equal(before_cpu,torch.get_rng_state()) or not torch.equal(before_cuda,torch.cuda.get_rng_state()):
            raise ValueError('Augmentation is not paired or changed global RNG')
        floor = -torch.tensor(data.MEAN,device='cuda')/torch.tensor(data.STD,device='cuda')
        ceiling = (1-torch.tensor(data.MEAN,device='cuda'))/torch.tensor(data.STD,device='cuda')
        if not bool((a >= floor[None,:,None,None]-1e-6).all()) or not bool((a <= ceiling[None,:,None,None]+1e-6).all()):
            raise ValueError('Brightness clipping range violated')
        stages = []
        for task, method in ((0,'sgd'),(0,'er'),(9,'sgd'),(9,'er')):
            model = make_model(seed+10000,'cuda').to(memory_format=torch.channels_last)
            stage = train_branch(model,dataset,images,labels,seed,task,method,'engineering_witness',2,
                output,time.monotonic()+600,first_task=task==0)
            stages.append({'task':task,'method':method,**stage})
        if stages[0]['end_state_sha256'] != stages[1]['end_state_sha256']:
            raise ValueError('Task0 SGD/ER pairing failed')
        projected = {}
        for method in ('sgd','er'):
            first = next(r['wall_seconds']/2 for r in stages if r['task']==0 and r['method']==method)
            later = next(r['wall_seconds']/2 for r in stages if r['task']==9 and r['method']==method)
            current = next(r['wall_seconds']/2 for r in stages if r['task']==9 and r['method']=='sgd')
            # SGD executes four fresh; ER executes four probe and reuses SGD fresh.
            projected[method] = first*200+later*630+current*280
        write_json(output/'RESOURCE.json', {'status':'passed_train_only_witness','utc':utc(),
            'environment':env,'test_access':False,'engineering_seed':seed,
            'augmentation_paired':True,'augmentation_preserves_global_rng':True,
            'brightness_clip01_verified':True,'first_task_method_identity':True,
            'stages':stages,'projected_training_seconds_per_job':projected,
            'projected_training_seconds_five_seed_matrix':5*sum(projected.values()),
            'projection_caveat':'two-epoch extrapolation, includes warmup; no runtime guarantee or accuracy result',
            'wall_seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'source_sha256':{p:data.file_digest(ROOT/p) for p in sorted(REQUIRED_SOURCES)}})
    except BaseException as error:
        write_json(output/'failure.json', {'utc':utc(),'error':type(error).__name__+': '+str(error)})
        raise
