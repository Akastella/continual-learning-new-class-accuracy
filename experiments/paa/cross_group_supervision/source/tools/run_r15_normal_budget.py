"""R15 train-only witness or frozen matrix; test phase starts after all histories."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from experiments.r15_normal_budget import core


def launch(command, log, timeout):
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with Path(log).open('xb') as stream:
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
            raise
    if code:
        raise RuntimeError(f'Child exited {code}; see {log}')


def matrix(manifest_path, output):
    manifest = core.validate_manifest(manifest_path)
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir(); (output/'jobs').mkdir()
    (output/'manifest.json').write_bytes(manifest_path.read_bytes())
    for name, expected in manifest['sources_sha256'].items():
        destination=output/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
        if core.data.file_digest(destination)!=expected:
            raise ValueError('Source changed while copying frozen snapshot: '+name)
    state={'schema':core.SCHEMA,'status':'training','started_utc':core.utc(),'jobs':[]}
    began=time.monotonic()
    core.write_json(output/'batch.json',state)
    try:
        for seed in core.SETTINGS['seeds']:
            for method in core.SETTINGS['methods']:
                name=f'seed{seed}_{method}'
                directory=output/'jobs'/name
                command=[sys.executable,'-s','-X','utf8',str(Path(__file__).resolve()),
                    '--worker','train','--manifest',str(manifest_path),'--output',str(directory),
                    '--seed',str(seed),'--method',method]
                launch(command,output/'logs'/f'{name}_train.log',manifest['job_timeout_seconds']+60)
                completed=directory/'training_complete.json'
                if not completed.is_file(): raise RuntimeError('No training completion marker')
                state['jobs'].append({'id':name,'training_complete_path':completed.relative_to(output).as_posix(),
                    'sha256':core.data.file_digest(completed)})
                core.write_json(output/'batch.json',state)
                print(json.dumps({'job':name,'stage':'training_complete','elapsed':time.monotonic()-began}),flush=True)
        # Reject broken pairing before any test payload is decoded.
        for seed in core.SETTINGS['seeds']:
            pair = [core.read_json(output/'jobs'/f'seed{seed}_{m}'/'training_complete.json')
                    for m in core.SETTINGS['methods']]
            first = [next(r for r in p['branches'] if r['branch']=='actual' and r['task']==0)
                     for p in pair]
            if first[0]['end_state_sha256'] != first[1]['end_state_sha256']:
                raise ValueError('Task0 method identity failed')
            for task in core.SETTINGS['diagnostic_tasks']:
                fresh = [next(r for r in p['branches'] if r['branch']=='fresh' and r['task']==task)
                         for p in pair]
                if (fresh[0]['start_state_sha256'] != fresh[1]['start_state_sha256'] or
                        fresh[0]['end_state_sha256'] != fresh[1]['end_state_sha256']):
                    raise ValueError('Fresh paired identity failed')
        # All ten full histories and auxiliary branches are complete before test access.
        seal=output/'all_training_complete.json'
        core.write_json(seal,{'utc':core.utc(),'manifest_sha256':core.data.file_digest(manifest_path),'jobs':state['jobs']})
        state['status']='evaluating';core.write_json(output/'batch.json',state)
        for job in state['jobs']:
            directory=output/'jobs'/job['id']
            command=[sys.executable,'-s','-X','utf8',str(Path(__file__).resolve()),'--worker','evaluate',
                '--manifest',str(manifest_path),'--output',str(directory),'--training-seal',str(seal)]
            launch(command,output/'logs'/f"{job['id']}_evaluate.log",1800)
            if not (directory/'result.json').exists(): raise RuntimeError('No evaluation result')
        state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        state.update(finished_utc=core.utc(),wall_seconds=time.monotonic()-began)
        core.write_json(output/'batch.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--witness',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--worker',choices=['train','evaluate'],help=argparse.SUPPRESS)
    p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--method',choices=['sgd','er'])
    p.add_argument('--training-seal',type=Path)
    args=p.parse_args();args.output=args.output.resolve()
    if args.witness:
        core.witness(args.output)
    else:
        if args.manifest is None:p.error('--manifest is required for frozen execution')
        args.manifest=args.manifest.resolve()
        if args.run:matrix(args.manifest,args.output)
        elif args.worker=='train':core.train_job(args.manifest,args.seed,args.method,args.output)
        else:
            if args.training_seal is None:p.error('--training-seal is required')
            core.evaluate_job(args.manifest,args.output,args.training_seal.resolve())


if __name__=='__main__':
    main()
