"""Unfrozen extension runner. No freeze operation; science requires a reviewed seal."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
FOLLOWUP = 'configs/r9_followup/manifest.json'
DERPP = 'configs/r9_derpp_challenge/manifest.json'
SETTINGS = {'kind': 'posthoc_descriptive_task4_budget_extension', 'seeds': list(range(201, 211)),
            'methods': ['sgd', 'er', 'derpp'], 'branches': ['actual', 'probe', 'shared_fresh'],
            'task': 4, 'start_epoch': 4, 'end_epoch': 16, 'endpoints': [4, 8, 16],
            'trajectories': 70, 'limit_seconds': 3300, 'optimizer_updates': 131880,
            'inference': 'descriptive_mean_sd_no_p_or_ci'}
BASE_SOURCES = ['tools/run_r9_budget_extension.py', 'experiments/r9_budget_extension/__init__.py',
                'experiments/r9_budget_extension/core.py', 'tests/test_r9_budget_extension.py',
                'experiments/r9_retention_trajectory/core.py', 'requirements-r9-windows-lock.txt']
FREEZE_SOURCES = BASE_SOURCES + [
    'protocol/R9_BUDGET_EXTENSION_PROTOCOL.md',
    'analysis/audit_r9_budget_extension.py',
    'analysis/replay_r9_budget_extension.py',
    'tests/test_r9_budget_extension_analysis.py',
]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def original_runs():
    rows = [r for r in read(ROOT/FOLLOWUP)['runs'] if r['id'].startswith('split_cifar10_classil_')]
    rows += read(ROOT/DERPP)['runs']
    if len(rows) != 30 or len({r['id'] for r in rows}) != 30:
        raise ValueError('Source matrix mismatch')
    return rows


def inputs(resource_path):
    names = {FOLLOWUP, DERPP, resource_path}
    for r in original_runs():
        names.update([r['config_path'], r['output']+'/artifacts.json'])
        names.update(r['output']+'/'+n for n in read(ROOT/r['output']/'artifacts.json')['files'])
    names.update('data/r9_official/cifar-10-batches-py/'+n for n in
                 ['batches.meta', 'test_batch', *[f'data_batch_{i}' for i in range(1, 6)]])
    return names


def validate_manifest(m):
    if m['settings'] != SETTINGS or m.get('engineering_ready') is not True:
        raise ValueError('Settings or readiness mismatch')
    if set(m['input_sha256']) != inputs(m['resource_path']):
        raise ValueError('Input ledger coverage')
    required = set(FREEZE_SOURCES) | set(read(ROOT/FOLLOWUP)['source_sha256'])
    if not required.issubset(m['source_sha256']):
        raise ValueError('Source ledger coverage')
    for group in ('source_sha256', 'input_sha256'):
        for n, h in m[group].items():
            p = (ROOT/n).resolve()
            if not p.is_relative_to(ROOT) or sha(p) != h:
                raise ValueError('Frozen drift '+n)
    for n, h in read(ROOT/FOLLOWUP)['source_sha256'].items():
        if sha(ROOT/n) != h:
            raise ValueError('Frozen parent drift '+n)
    resource = read(ROOT/m['resource_path'])
    if resource.get('status') != 'complete' or resource.get('no_test_access') is not True or resource['conservative_seconds'] >= 3300:
        raise ValueError('Resource witness fails budget')
    if resource.get('seed') in SETTINGS['seeds'] or resource.get('device') != 'cuda':
        raise ValueError('Witness must use independent seed and science device')
    if resource.get('source_sha256') != {n: sha(ROOT/n) for n in BASE_SOURCES}:
        raise ValueError('Witness predates current implementation')
    stamp = datetime.fromisoformat(m['frozen_utc'])
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Invalid freeze chronology')


def configure(device):
    import torch
    if not sys.flags.no_user_site:
        raise RuntimeError('Invoke with -s')
    locked = dict(line.split('==', 1) for line in (ROOT/'requirements-r9-windows-lock.txt').read_text(encoding='utf-8-sig').splitlines()
                  if '==' in line and not line.startswith('#'))
    for package in ('torch', 'torchvision', 'numpy', 'pillow'):
        if importlib.metadata.version(package) != locked[package]:
            raise ValueError('Environment drift '+package)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def checked_config(row):
    cfg = read(ROOT/row['config_path'])
    expected = {'benchmark': 'split_cifar10_classil', 'tasks': 5, 'epochs': 4, 'batch_size': 64,
                'lr': .01, 'width': 256, 'memory_size': 500, 'replay_batch': 32,
                'max_train': None, 'max_test': None}
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg['seed'] not in SETTINGS['seeds']:
        raise ValueError('Source configuration does not match extension')
    if cfg['method'] == 'derpp' and (cfg['der_alpha'] != .5 or cfg['der_beta'] != .5):
        raise ValueError('DER++ coefficients')
    if sha(ROOT/row['config_path']) != row['config_sha256']:
        raise ValueError('Original config drift')
    return cfg


def check_anchor(actual, expected):
    for k in ('n', 'correct', 'prediction_sha256'):
        if actual[k] != expected[k]:
            raise ValueError('Epoch-four anchor mismatch '+k)


def emit_epoch(out, rd, cfg, doc, tasks, device, record, active_model, active_memory, arrays=None, sc=None):
    import numpy as np
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_budget_extension import core
    epoch = record['epoch']; started = time.perf_counter()
    cp = rd/f'epoch_{epoch:03d}.pt'
    with cp.open('xb') as f:
        torch.save({'epoch': epoch, 'model': {k:v.detach().cpu().clone() for k,v in active_model.state_dict().items()},
                    'memory': copy.deepcopy(active_memory.state_dict()) if active_memory else None,
                    'config': cfg, 'state_sha256': engine.state_hash(active_model)}, f)
    if epoch in SETTINGS['endpoints']:
        if arrays is None:
            arrays, sc = core.inference(active_model, tasks, device)
        npz = rd/f'epoch_{epoch:03d}.npz'
        with npz.open('xb') as f:
            np.savez_compressed(f, **arrays)
        doc['endpoints'].append({'epoch': epoch, 'npz': npz.relative_to(out).as_posix(),
                                 'npz_sha256': sha(npz), 'scores': sc})
        current = sc['all10']['current']
    else:
        current = engine.evaluate(active_model, tasks[4], cfg['benchmark'], device)
    record.update(checkpoint=cp.relative_to(out).as_posix(), checkpoint_sha256=sha(cp),
                  current_evaluation=current, serialization_evaluation_seconds=time.perf_counter()-started)
    doc['epochs'].append(record)
    save(rd/f'epoch_{epoch:03d}.json', record)


def worker(manifest_path, out, deadline):
    import numpy as np
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_followup.provenance import validate_finished_run
    from experiments.r9_budget_extension import core
    m = read(manifest_path); validate_manifest(m); device = configure('cuda')
    save(out/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
         'cuda': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(),
         'no_user_site': True, 'deterministic_algorithms': True, 'tf32': False, 'threads': 2})
    parent_source = read(ROOT/FOLLOWUP)['source_sha256']; sources = {}
    for r in original_runs():
        cfg = checked_config(r)
        raw = validate_finished_run(ROOT/r['output'], cfg, parent_source)
        sources[(cfg['seed'], cfg['method'])] = (r, cfg, raw)
    completed = []; updates = 0
    for seed in SETTINGS['seeds']:
        base = sources[(seed, 'sgd')][1]
        tasks, provenance = engine.make_stream(base, ROOT/'data')
        if [len(t['train'][1]) for t in tasks] != [10000]*5 or [len(t['test'][1]) for t in tasks] != [2000]*5:
            raise ValueError('Full official task sizes required')
        fresh_states = []; fresh_orders = []; handles = []
        for method in SETTINGS['methods']:
            r, cfg, raw = sources[(seed, method)]
            dm = read(ROOT/r['output']/'data_manifest.json')
            if provenance != dm['provenance'] or [{k:v for k,v in t.items() if k not in ('train', 'test')} for t in tasks] != dm['tasks']:
                raise ValueError('Official stream drift')
            if any(cfg.get(k) != base.get(k) for k in ('init_seed', 'order_seed', 'class_order_seed')):
                raise ValueError('Unmatched source seeds')
            fresh_states.append(torch.load(ROOT/r['output']/'fresh_task_004.pt', map_location='cpu', weights_only=True))
            fresh_orders.append(raw['tasks'][4]['fresh']['batch_order_sha256'])
        if any(s.keys() != fresh_states[0].keys() or not all(torch.equal(v, fresh_states[0][k]) for k,v in s.items()) for s in fresh_states[1:]) or any(x != fresh_orders[0] for x in fresh_orders[1:]):
            raise ValueError('Cannot share unequal fresh anchors/orders')
        combinations = [(method, branch) for method in SETTINGS['methods'] for branch in ('actual', 'probe')] + [('shared', 'fresh')]
        for method, branch in combinations:
            effective = 'sgd' if method == 'shared' else method
            r, cfg, raw = sources[(seed, effective)]
            rid = f's{seed}_{method}_{branch}'; rd = out/'trajectories'/rid; rd.mkdir(parents=True)
            filename = {'actual': 'post', 'probe': 'probe', 'fresh': 'fresh'}[branch]+'_task_004.pt'
            path = ROOT/r['output']/filename
            state = torch.load(path, map_location='cpu', weights_only=True)
            model = engine.local_model(cfg, cfg['init_seed'], device, state)
            memory_path = ROOT/r['output']/'post_task_004_memory.pt'
            memory = core.restore_memory(torch.load(memory_path, map_location='cpu', weights_only=False)) if branch == 'actual' and method in ('er', 'derpp') else None
            anchor_sources = [sources[(seed, mm)] for mm in SETTINGS['methods']] if branch == 'fresh' else [(r, cfg, raw)]
            lineage = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)}
            if memory is not None:
                lineage[memory_path.relative_to(ROOT).as_posix()] = sha(memory_path)
            arrays, sc = core.inference(model, tasks, device)
            for ar, acfg, araw in anchor_sources:
                record = araw['tasks'][4]
                check_anchor(sc['all10']['current'], record[branch]['trace'][-1]['evaluation'])
                if engine.state_hash(model) != record[branch]['final_state_sha256']:
                    raise ValueError('Anchor tensor hash')
                if branch == 'actual':
                    for observed, expected in zip(sc['all10']['tasks'], record['post_task_evaluation']):
                        check_anchor(observed, expected)
                ap = ROOT/ar['output']/filename; lineage[ap.relative_to(ROOT).as_posix()] = sha(ap)
            doc = {'schema': 1, 'id': rid, 'seed': seed, 'method': method, 'branch': branch,
                   'source_run_ids': [ar['id'] for ar, _, _ in anchor_sources], 'lineage': lineage,
                   'anchor_validation': {'current_counts_prediction_and_state_exact': True,
                       'all_tasks_originally_available': branch == 'actual',
                       'old_probe_or_fresh_evaluation': 'new_re_evaluation' if branch != 'actual' else 'original_counts_and_predictions_exact'},
                   'epochs': [], 'endpoints': []}
            emit_epoch(out, rd, cfg, doc, tasks, device, {'epoch': 4, 'state_sha256': engine.state_hash(model), 'counts': dict(optimizer_steps=0, current_examples=0, replay_examples=0, forward_examples=0),
                  'batch_order_sha256': None, 'losses': None}, model, memory, arrays, sc)
            # All seven anchors for this seed are verified before any continuation.
            handles.append((rid, rd, cfg, method, branch, model, memory, doc))
        for rid, rd, cfg, method, branch, model, memory, doc in handles:
            def callback(record, active_model, active_memory):
                emit_epoch(out, rd, cfg, doc, tasks, device, record, active_model, active_memory)
            records = core.train_epochs(model, tasks[4], cfg, device, method if branch == 'actual' else 'sgd', memory,
                                        callback=callback, deadline=deadline)
            updates += sum(row['counts']['optimizer_steps'] for row in records)
            expected = [engine.digest_tensor(core.epoch_order(10000, cfg, 4, e)) for e in range(4, 16)]
            if [row['batch_order_sha256'] for row in records] != expected:
                raise ValueError('Epoch order mismatch')
            save(rd/'trajectory.json', doc); completed.append(rid)
            print(json.dumps({'completed_trajectories': len(completed), 'optimizer_updates': updates}), flush=True)
        del handles, tasks
    if len(completed) != 70 or len(set(completed)) != 70 or updates != SETTINGS['optimizer_updates']:
        raise ValueError('Incomplete or duplicate trajectory matrix')
    validate_manifest(m)
    save(out/'complete.json', {'status': 'complete', 'trajectories': completed, 'optimizer_updates': updates,
                              'manifest_sha256': sha(manifest_path), 'finished_utc': utc()})
    save(out/'campaign_artifacts.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
         if p.is_file() and p.name not in ('console.log', 'watchdog.json')})


def witness(out, device_name):
    """Independent seed, official TRAIN only; measurements are not scientific results."""
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_retention_trajectory.core import load_train_only
    from experiments.r9_budget_extension import core
    started = time.perf_counter(); out.mkdir(parents=True, exist_ok=False)
    device = configure(device_name); seed = 20260917
    cfg = {'benchmark': 'split_cifar10_classil', 'seed': seed, 'init_seed': seed+10000,
           'order_seed': seed+30000, 'class_order_seed': seed+40000, 'epochs': 4, 'tasks': 5,
           'batch_size': 64, 'lr': .01, 'width': 256, 'max_train': None, 'max_test': None,
           'memory_size': 500, 'replay_batch': 32, 'der_alpha': .5, 'der_beta': .5}
    x, y, proxy, provenance = load_train_only(ROOT/'data', cfg, 4)
    task = {'id': 4, 'train': (x, y)}; measures = {}
    for method in ('sgd', 'er', 'derpp'):
        model = engine.local_model(cfg, cfg['init_seed'], device)
        memory = core.Reservoir(500, seed+77001) if method != 'sgd' else None
        if memory is not None:
            with torch.no_grad():
                memory.add_batch(x[:500], y[:500], 4, model(x[:500].to(device)).cpu())
        records = core.train_epochs(model, task, cfg, device, method, memory, 4, 5)
        t = time.perf_counter()
        with (out/f'{method}_checkpoint.pt').open('xb') as f:
            torch.save({'model': model.cpu().state_dict(), 'memory': memory.state_dict() if memory else None}, f)
        serialization = time.perf_counter()-t; model.to(device).eval()
        t = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(proxy), 512):
                z = model(proxy[s:s+512].to(device)).cpu()
                if not torch.isfinite(z).all(): raise FloatingPointError('Witness inference')
        if device.type == 'cuda': torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-t
        measures[method] = {'epoch_seconds': records[0]['training_seconds'], 'serialization_seconds': serialization,
                            'proxy_2000_forward_seconds': inference_seconds, 'counts': records[0]['counts']}
    # 50 SGD, 10 ER, 10 DER++ continuations; 13 saved epochs and 3 all-five endpoints.
    training = 12*(50*measures['sgd']['epoch_seconds']+10*measures['er']['epoch_seconds']+10*measures['derpp']['epoch_seconds'])
    serialization = 70*13*max(v['serialization_seconds'] for v in measures.values())
    evaluation = 70*(3*5+10)*max(v['proxy_2000_forward_seconds'] for v in measures.values())
    projected = training+serialization+evaluation
    report = {'status': 'complete', 'no_test_access': True, 'seed': seed, 'device': str(device),
              'torch': torch.__version__, 'config': cfg, 'train_provenance': provenance,
              'measurements': measures, 'projection': {'training_seconds': training, 'serialization_seconds': serialization,
                  'evaluation_seconds': evaluation, 'raw_seconds': projected, 'safety_multiplier': 1.5,
                  'fixed_overhead_seconds': 180}, 'conservative_seconds': projected*1.5+180,
              'actual_seconds': time.perf_counter()-started,
              'scope': 'Independent train-only one full continuation epoch per operator; train forward proxy; no accuracy or test access',
              'source_sha256': {n:sha(ROOT/n) for n in BASE_SOURCES if (ROOT/n).exists()}}
    save(out/'report.json', report)
    print(json.dumps({'report': str(out/'report.json'), 'conservative_seconds': report['conservative_seconds']}))


def kill_tree(child):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run', 'witness'])
    p.add_argument('--manifest'); p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--worker', action='store_true'); p.add_argument('--deadline', type=float)
    args = p.parse_args(); out = Path(args.output).resolve()
    if args.action == 'witness':
        witness(out, args.device); return
    if not args.manifest: p.error('run requires --manifest')
    manifest_path = Path(args.manifest).resolve()
    if args.worker:
        if args.deadline is None: p.error('worker requires deadline')
        try:
            worker(manifest_path, out, args.deadline)
        except BaseException:
            save(out/'failure.json', {'utc': utc(), 'traceback': traceback.format_exc()}); raise
        return
    started = time.monotonic(); deadline = started+3300
    m = read(manifest_path); validate_manifest(m)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'started.json', {'utc': utc(), 'manifest_sha256': sha(manifest_path), 'limit_seconds': 3300})
    shutil.copy2(manifest_path, out/'frozen_manifest.json')
    for n in m['source_sha256']:
        dest = out/'source'/n; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/n, dest)
    env = os.environ.copy()
    for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP'): env.pop(k, None)
    env.update(PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/'console.log').open('xb') as log:
        command = [sys.executable, '-s', '-X', 'utf8', str(Path(__file__).resolve()), 'run', '--worker',
                   '--manifest', str(manifest_path), '--output', str(out), '--deadline', str(deadline)]
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
        try:
            code = child.wait(timeout=max(.01, deadline-time.monotonic())); status = 'complete' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            kill_tree(child); code = None; status = 'timed_out'
    save(out/'watchdog.json', {'status': status, 'exit_code': code, 'limit_seconds': 3300,
                              'elapsed_seconds': time.monotonic()-started, 'entire_process_tree_terminated_on_timeout': True})
    if status != 'complete': raise RuntimeError('Incomplete matrix; retain partial artifacts; no analysis')


if __name__ == '__main__':
    main()
