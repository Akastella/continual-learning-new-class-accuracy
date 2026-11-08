"""Fixed four-task budget recheck; explicit manifest seal, no automatic freezing."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from tools import run_r9_budget_extension as parent

sha, read, save, utc = parent.sha, parent.read, parent.save, parent.utc
FOLLOWUP = parent.FOLLOWUP
PRIOR = 'results/r9_budget_extension_v1'
SETTINGS = {'kind': 'posthoc_four_task_budget_recheck', 'seeds': list(range(201, 211)),
            'tasks': [1, 2, 3, 4], 'methods': ['sgd', 'er'], 'branches': ['actual', 'probe'],
            'start_epoch': 4, 'end_epoch': 16, 'endpoints': [4, 8, 16],
            'logical_trajectories': 160, 'new_trajectories': 120, 'reused_trajectories': 40,
            'new_optimizer_updates': 226080, 'limit_seconds': 3300}
BASE_SOURCES = ['tools/run_r12_four_task_budget.py', 'tests/test_r12_four_task_budget.py',
                'tools/run_r9_budget_extension.py', 'experiments/r9_budget_extension/core.py',
                'experiments/r9_retention_trajectory/core.py', 'requirements-r9-windows-lock.txt']
FREEZE_SOURCES = BASE_SOURCES + ['protocol/R12_FOUR_TASK_BUDGET_PROTOCOL.md',
    'analysis/audit_r12_four_task_budget.py', 'analysis/replay_r12_four_task_budget.py',
    'tests/test_r12_four_task_budget_analysis.py']


def original_runs():
    rows = [r for r in read(ROOT/FOLLOWUP)['runs']
            if r['id'].startswith('split_cifar10_classil_')]
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
        raise ValueError('Source matrix mismatch')
    return rows


def prior_paths(seed, method, branch):
    base = f'{PRIOR}/trajectories/s{seed}_{method}_{branch}'
    return [base+'/trajectory.json', *[base+f'/epoch_{e:03d}.json' for e in range(4, 17)],
            *[base+f'/epoch_{e:03d}.{ext}' for e in SETTINGS['endpoints'] for ext in ('pt', 'npz')]]


def inputs(resource_path):
    names = {FOLLOWUP, resource_path, f'{PRIOR}/frozen_manifest.json',
             f'{PRIOR}/campaign_artifacts.json', f'{PRIOR}/complete.json', f'{PRIOR}/watchdog.json'}
    for row in original_runs():
        names.update([row['config_path'], row['output']+'/artifacts.json'])
        names.update(row['output']+'/'+n for n in read(ROOT/row['output']/'artifacts.json')['files'])
    for seed in SETTINGS['seeds']:
        for method in SETTINGS['methods']:
            for branch in SETTINGS['branches']:
                names.update(prior_paths(seed, method, branch))
    names.update('data/r9_official/cifar-10-batches-py/'+n for n in
                 ['batches.meta', 'test_batch', *[f'data_batch_{i}' for i in range(1, 6)]])
    return names


def validate_manifest(m):
    if m.get('settings') != SETTINGS or m.get('engineering_ready') is not True:
        raise ValueError('Settings/readiness mismatch')
    if set(m['input_sha256']) != inputs(m['resource_path']):
        raise ValueError('Input ledger coverage mismatch')
    required = set(FREEZE_SOURCES) | set(read(ROOT/FOLLOWUP)['source_sha256'])
    if not required.issubset(m['source_sha256']):
        raise ValueError('Source ledger coverage mismatch')
    for group in ('source_sha256', 'input_sha256'):
        for name, expected in m[group].items():
            path = (ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != expected:
                raise ValueError('Frozen drift '+name)
    for name, expected in read(ROOT/FOLLOWUP)['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Parent source drift '+name)
    resource = read(ROOT/m['resource_path'])
    if (resource.get('status') != 'complete' or resource.get('no_test_access') is not True
            or resource.get('device') != 'cuda' or resource.get('seed') in SETTINGS['seeds']
            or resource['conservative_seconds'] >= SETTINGS['limit_seconds']):
        raise ValueError('Resource witness fails budget')
    if resource.get('source_sha256') != {n: sha(ROOT/n) for n in BASE_SOURCES}:
        raise ValueError('Witness predates implementation')
    stamp = datetime.fromisoformat(m['frozen_utc'])
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Invalid freeze chronology')
    done, watchdog = read(ROOT/PRIOR/'complete.json'), read(ROOT/PRIOR/'watchdog.json')
    if done.get('status') != 'complete' or watchdog.get('status') != 'complete' or watchdog.get('exit_code') != 0:
        raise ValueError('Prior task-four campaign is incomplete')
    if done['manifest_sha256'] != sha(ROOT/PRIOR/'frozen_manifest.json'):
        raise ValueError('Prior manifest binding mismatch')
    prior_sources = read(ROOT/PRIOR/'frozen_manifest.json')['source_sha256']
    for name in BASE_SOURCES[2:]:
        if prior_sources.get(name) != sha(ROOT/name):
            raise ValueError('Reused operator differs from frozen prior '+name)
    inventory = read(ROOT/PRIOR/'campaign_artifacts.json')
    for seed in SETTINGS['seeds']:
        for method in SETTINGS['methods']:
            for branch in SETTINGS['branches']:
                for name in prior_paths(seed, method, branch):
                    relative = Path(name).relative_to(PRIOR).as_posix()
                    if inventory.get(relative) != sha(ROOT/name):
                        raise ValueError('Prior artifact binding mismatch '+name)


def configure(device_name):
    expected = (ROOT/'.venv-r9-locked/Scripts/python.exe').resolve()
    if Path(sys.executable).resolve() != expected or not sys.flags.no_user_site:
        raise RuntimeError('Use the locked project interpreter with -s')
    return parent.configure(device_name)


def raw_inference(model, tasks, device):
    """All five tasks, no endpoint semantics delegated to hardcoded task-four scores."""
    import numpy as np
    import torch
    model.eval(); logits = []; labels = []; indices = []; tids = []
    with torch.no_grad():
        for task in tasks:
            x, y = task['test']; chunks = []
            for start in range(0, len(y), 512):
                chunks.append(model(x[start:start+512].to(device)).cpu())
            logits.append(torch.cat(chunks).numpy()); labels.append(y.numpy())
            indices.extend(task['splits']['test']['official_indices'])
            tids.extend([task['id']]*len(y))
    arrays = {'logits': np.concatenate(logits), 'labels': np.concatenate(labels),
              'official_indices': np.asarray(indices, dtype=np.int64),
              'task_ids': np.asarray(tids, dtype=np.int64)}
    if arrays['logits'].shape != (len(arrays['labels']), 10) or not np.isfinite(arrays['logits']).all():
        raise ValueError('Invalid logits')
    return arrays


def anchor_counts(arrays, tid):
    import torch
    from experiments.r9_followup import engine
    mask = arrays['task_ids'] == tid
    pred = torch.as_tensor(arrays['logits'][mask]).argmax(1)
    y = torch.as_tensor(arrays['labels'][mask])
    return {'n': len(y), 'correct': int((pred == y).sum()), 'prediction_sha256': engine.digest_tensor(pred)}


def validate_anchor(arrays, state_hash, raw, tid, branch):
    record = raw['tasks'][tid]
    parent.check_anchor(anchor_counts(arrays, tid), record[branch]['trace'][-1]['evaluation'])
    if state_hash != record[branch]['final_state_sha256']:
        raise ValueError('Anchor state mismatch')
    if branch == 'actual':
        for t, expected in enumerate(record['post_task_evaluation']):
            parent.check_anchor(anchor_counts(arrays, t), expected)


def emit_epoch(out, rd, cfg, doc, tasks, device, record, model, memory, arrays=None):
    import numpy as np
    import torch
    from experiments.r9_followup import engine
    epoch = record['epoch']
    if epoch in SETTINGS['endpoints']:
        cp = rd/f'epoch_{epoch:03d}.pt'
        with cp.open('xb') as stream:
            torch.save({'epoch': epoch, 'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'memory': copy.deepcopy(memory.state_dict()) if memory else None,
                        'config': cfg, 'state_sha256': engine.state_hash(model)}, stream)
        if arrays is None:
            arrays = raw_inference(model, tasks, device)
        npz = rd/f'epoch_{epoch:03d}.npz'
        with npz.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        endpoint = {'epoch': epoch, 'npz': npz.relative_to(out).as_posix(), 'npz_sha256': sha(npz),
                    'checkpoint': cp.relative_to(out).as_posix(), 'checkpoint_sha256': sha(cp)}
        doc['endpoints'].append(endpoint)
        record.update(checkpoint=endpoint['checkpoint'], checkpoint_sha256=endpoint['checkpoint_sha256'])
        record['current_evaluation'] = anchor_counts(arrays, doc['task'])
    doc['epochs'].append(record)
    save(rd/f'epoch_{epoch:03d}.json', record)


def reuse_task4(out, rd, doc, raw):
    import numpy as np
    import torch
    from experiments.r9_followup import engine
    names = prior_paths(doc['seed'], doc['method'], doc['branch'])
    previous = read(ROOT/names[0])
    if (previous['seed'], previous['method'], previous['branch']) != (doc['seed'], doc['method'], doc['branch']):
        raise ValueError('Reuse identity mismatch')
    doc['source_trajectory_json'] = names[0]
    doc['lineage'].update({name: sha(ROOT/name) for name in names})
    base = (ROOT/names[0]).parent
    for original in previous['epochs']:
        record = copy.deepcopy(original); epoch = record['epoch']
        if record != read(base/f'epoch_{epoch:03d}.json'):
            raise ValueError('Prior embedded/per-epoch record mismatch')
        record.pop('checkpoint', None); record.pop('checkpoint_sha256', None)
        if epoch in SETTINGS['endpoints']:
            for suffix in ('pt', 'npz'):
                shutil.copy2(base/f'epoch_{epoch:03d}.{suffix}', rd/f'epoch_{epoch:03d}.{suffix}')
            cp, npz = rd/f'epoch_{epoch:03d}.pt', rd/f'epoch_{epoch:03d}.npz'
            endpoint = {'epoch': epoch, 'npz': npz.relative_to(out).as_posix(), 'npz_sha256': sha(npz),
                        'checkpoint': cp.relative_to(out).as_posix(), 'checkpoint_sha256': sha(cp)}
            doc['endpoints'].append(endpoint)
            record.update(checkpoint=endpoint['checkpoint'], checkpoint_sha256=endpoint['checkpoint_sha256'])
            if epoch == 4:
                state = torch.load(cp, map_location='cpu', weights_only=False)
                h = engine.state_hash(engine.local_model(state['config'], state['config']['init_seed'], torch.device('cpu'), state['model']))
                with np.load(npz, allow_pickle=False) as arr:
                    validate_anchor(arr, h, raw, 4, doc['branch'])
        doc['epochs'].append(record)
        save(rd/f'epoch_{epoch:03d}.json', record)
    if [r['epoch'] for r in doc['epochs']] != list(range(4, 17)):
        raise ValueError('Reused epoch coverage')


def worker(manifest_path, out, deadline):
    import numpy as np
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_followup.provenance import validate_finished_run
    from experiments.r9_budget_extension import core
    manifest = read(manifest_path); validate_manifest(manifest); device = configure('cuda')
    save(out/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(),
        'no_user_site': True, 'deterministic_algorithms': True, 'tf32': False, 'threads': 2})
    parent_source = read(ROOT/FOLLOWUP)['source_sha256']; sources = {}
    for row in original_runs():
        cfg = parent.checked_config(row)
        raw = validate_finished_run(ROOT/row['output'], cfg, parent_source)
        sources[(cfg['seed'], cfg['method'])] = row, cfg, raw
    completed = []; updates = 0; reused = 0
    for seed in SETTINGS['seeds']:
        base = sources[(seed, 'sgd')][1]
        tasks, provenance = engine.make_stream(base, ROOT/'data')
        if [len(t['train'][1]) for t in tasks] != [10000]*5 or [len(t['test'][1]) for t in tasks] != [2000]*5:
            raise ValueError('Full official tasks required')
        for method in SETTINGS['methods']:
            row, cfg, raw = sources[(seed, method)]
            dm = read(ROOT/row['output']/'data_manifest.json')
            if provenance != dm['provenance'] or [{k:v for k,v in t.items() if k not in ('train','test')} for t in tasks] != dm['tasks']:
                raise ValueError('Data stream drift')
            if any(cfg[k] != base[k] for k in ('init_seed', 'order_seed', 'class_order_seed')):
                raise ValueError('Unpaired configuration')
            for tid in SETTINGS['tasks']:
                for branch in SETTINGS['branches']:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Whole matrix deadline')
                    rid = f's{seed}_t{tid}_{method}_{branch}'
                    rd = out/'trajectories'/rid; rd.mkdir(parents=True)
                    filename = ('post' if branch == 'actual' else 'probe')+f'_task_{tid:03d}.pt'
                    path = ROOT/row['output']/filename
                    doc = {'schema': 1, 'id': rid, 'seed': seed, 'task': tid, 'method': method, 'branch': branch,
                           'origin': 'reused_r9_task4' if tid == 4 else 'new_continuation',
                           'source_run_ids': [row['id']], 'lineage': {path.relative_to(ROOT).as_posix(): sha(path)},
                           'anchor_validation': {'current_counts_prediction_and_state_exact': True,
                               'all_tasks_originally_available': branch == 'actual',
                               'other_task_evaluation': 'original_counts_and_predictions_exact' if branch == 'actual' else
                                   ('reused_r9_new_re_evaluation' if tid == 4 else 'new_r12_re_evaluation')},
                           'epochs': [], 'endpoints': []}
                    mp = ROOT/row['output']/f'post_task_{tid:03d}_memory.pt'
                    if method == 'er' and branch == 'actual':
                        doc['lineage'][mp.relative_to(ROOT).as_posix()] = sha(mp)
                    if tid == 4:
                        reuse_task4(out, rd, doc, raw); reused += 1
                    else:
                        state = torch.load(path, map_location='cpu', weights_only=True)
                        model = engine.local_model(cfg, cfg['init_seed'], device, state)
                        memory = core.restore_memory(torch.load(mp, map_location='cpu', weights_only=False)) if method == 'er' and branch == 'actual' else None
                        arrays = raw_inference(model, tasks, device)
                        validate_anchor(arrays, engine.state_hash(model), raw, tid, branch)
                        emit_epoch(out, rd, cfg, doc, tasks, device,
                            {'epoch': 4, 'state_sha256': engine.state_hash(model),
                             'counts': dict(optimizer_steps=0, current_examples=0, replay_examples=0, forward_examples=0),
                             'batch_order_sha256': None, 'losses': None}, model, memory, arrays)
                        def callback(record, active_model, active_memory):
                            emit_epoch(out, rd, cfg, doc, tasks, device, record, active_model, active_memory)
                        records = core.train_epochs(model, tasks[tid], cfg, device,
                            method if branch == 'actual' else 'sgd', memory, callback=callback, deadline=deadline)
                        expected = [engine.digest_tensor(core.epoch_order(10000, cfg, tid, e)) for e in range(4, 16)]
                        if [r['batch_order_sha256'] for r in records] != expected:
                            raise ValueError('Continuation order mismatch')
                        updates += sum(r['counts']['optimizer_steps'] for r in records)
                        del model, memory
                    save(rd/'trajectory.json', doc); completed.append(rid)
                    print(json.dumps({'completed_trajectories': len(completed), 'new_optimizer_updates': updates, 'reused_trajectories': reused}), flush=True)
    if len(completed) != 160 or len(set(completed)) != 160 or updates != 226080 or reused != 40:
        raise ValueError('Incomplete matrix')
    validate_manifest(manifest)
    save(out/'complete.json', {'status': 'complete', 'trajectories': completed, 'new_optimizer_updates': updates,
        'reused_trajectories': reused, 'manifest_sha256': sha(manifest_path), 'finished_utc': utc()})
    save(out/'campaign_artifacts.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
        if p.is_file() and p.name not in ('console.log', 'watchdog.json')})


def witness(out, device_name):
    """Independent seed; official TRAIN tensors only, no scientific endpoint."""
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_retention_trajectory.core import load_train_only
    from experiments.r9_budget_extension import core
    started = time.perf_counter(); out.mkdir(parents=True, exist_ok=False)
    device = configure(device_name); seed = 2026091701
    cfg = {'benchmark': 'split_cifar10_classil', 'seed': seed, 'init_seed': seed+10000,
           'order_seed': seed+30000, 'class_order_seed': seed+40000, 'epochs': 4, 'tasks': 5,
           'batch_size': 64, 'lr': .01, 'width': 256, 'max_train': None, 'max_test': None,
           'memory_size': 500, 'replay_batch': 32}
    measures = {}
    for tid in (1, 2, 3):
        x, y, proxy, provenance = load_train_only(ROOT/'data', cfg, tid)
        for method in ('sgd', 'er'):
            model = engine.local_model(cfg, cfg['init_seed'], device)
            memory = core.Reservoir(500, seed+77001) if method == 'er' else None
            if memory:
                with torch.no_grad():
                    memory.add_batch(x[:500], y[:500], tid, model(x[:500].to(device)).cpu())
            records = core.train_epochs(model, {'id': tid, 'train': (x,y)}, cfg, device, method, memory, 4, 5)
            t = time.perf_counter()
            with (out/f't{tid}_{method}_checkpoint.pt').open('xb') as stream:
                torch.save({'model': {k:v.cpu() for k,v in model.state_dict().items()},
                            'memory': memory.state_dict() if memory else None}, stream)
            serialization = time.perf_counter()-t; model.eval(); t = time.perf_counter()
            with torch.no_grad():
                for start in range(0, len(proxy), 512):
                    if not torch.isfinite(model(proxy[start:start+512].to(device))).all():
                        raise FloatingPointError('Witness inference')
            if device.type == 'cuda': torch.cuda.synchronize()
            measures[f't{tid}_{method}'] = {'epoch_seconds': records[0]['training_seconds'],
                'serialization_seconds': serialization, 'proxy_2000_forward_seconds': time.perf_counter()-t,
                'counts': records[0]['counts']}
    training = 12*10*sum(3*measures[f't{tid}_sgd']['epoch_seconds']+measures[f't{tid}_er']['epoch_seconds'] for tid in (1,2,3))
    serialization = 120*3*max(v['serialization_seconds'] for v in measures.values())
    evaluation = 120*3*5*max(v['proxy_2000_forward_seconds'] for v in measures.values())
    projected = training+serialization+evaluation
    save(out/'report.json', {'status':'complete', 'seed':seed, 'device':str(device), 'no_test_access':True,
        'measurements':measures, 'projection':{'training_seconds':training,'serialization_seconds':serialization,
        'evaluation_seconds':evaluation,'raw_seconds':projected,'safety_multiplier':1.5,'fixed_overhead_seconds':300},
        'conservative_seconds':1.5*projected+300,'actual_seconds':time.perf_counter()-started,
        'scope':'Three task IDs, full train-only epoch per SGD/ER operator; all 120 new trajectories, 40 verified reuses included by fixed overhead',
        'source_sha256':{n:sha(ROOT/n) for n in BASE_SOURCES}})
    print(json.dumps({'report':str(out/'report.json'),'conservative_seconds':1.5*projected+300}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'witness'])
    parser.add_argument('--manifest'); parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--deadline', type=float)
    args = parser.parse_args(); out = Path(args.output).resolve()
    if args.action == 'witness':
        witness(out, args.device); return
    if not args.manifest: parser.error('run requires --manifest')
    manifest = Path(args.manifest).resolve()
    if args.worker:
        if args.deadline is None: parser.error('worker requires deadline')
        try: worker(manifest, out, args.deadline)
        except BaseException:
            save(out/'failure.json', {'utc':utc(), 'traceback':traceback.format_exc()}); raise
        return
    started = time.monotonic(); deadline = started+SETTINGS['limit_seconds']
    m = read(manifest); validate_manifest(m)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'started.json', {'utc':utc(),'manifest_sha256':sha(manifest),'limit_seconds':3300})
    shutil.copy2(manifest,out/'frozen_manifest.json')
    for name in m['source_sha256']:
        dest = out/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/'console.log').open('xb') as log:
        command = [sys.executable,'-s','-X','utf8',str(Path(__file__).resolve()),'run','--worker',
                   '--manifest',str(manifest),'--output',str(out),'--deadline',str(deadline)]
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
        try:
            code = child.wait(timeout=max(.01,deadline-time.monotonic())); status = 'complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            parent.kill_tree(child); code=None; status='timed_out'
    save(out/'watchdog.json',{'status':status,'exit_code':code,'limit_seconds':3300,
        'elapsed_seconds':time.monotonic()-started,'entire_process_tree_terminated_on_timeout':True})
    if status != 'complete': raise RuntimeError('Incomplete matrix; no endpoint analysis')


if __name__ == '__main__':
    main()
