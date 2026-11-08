"""Ten DER++ histories under one fixed 55-minute watchdog; frozen engine reuse."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
START=time.monotonic()
MANIFEST='configs/r9_derpp_challenge/manifest.json'
OUTPUT='results/r9_derpp_challenge'
ORIGINAL='configs/r9_followup/manifest.json'
ORIGINAL_SHA='bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e'
RESOURCE='audit/r9/derpp_resource_20260916/report.json'
SETTINGS={'kind':'posthoc_descriptive_derpp_challenge','seeds':list(range(201,211)),
          'benchmark':'split_cifar10_classil','method':'derpp','der_alpha':.5,'der_beta':.5,
          'tasks':5,'epochs':4,'batch_size':64,'memory_size':500,'replay_batch':32,
          'branches':['actual','probe','fresh'],'release_current_weight_control':False,
          'limit_seconds':3300,'histories':10,'optimizer_updates':94200,
          'training_forward_images':8008960,'primary_tasks':[1,2,3,4],
          'primary_epoch':4,'reported_epochs':[1,2,4],'inference':'descriptive_mean_sd_no_p_or_ci'}
SOURCES=['tools/run_r9_derpp_challenge.py','tools/freeze_r9_derpp_challenge.py',
         'tests/test_r9_derpp_challenge.py','protocol/R9_DERPP_CHALLENGE.md',
         'analysis/audit_r9_derpp_challenge.py','analysis/summarize_r9_derpp_challenge.py',
         'analysis/replay_r9_derpp_challenge.py','tests/test_r9_derpp_analysis.py',
         'requirements-r9-windows-lock.txt']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def utc():return datetime.now(timezone.utc).isoformat()


def planned_runs():
    return [{'id':f'split_cifar10_classil_derpp_s{s}',
             'config_path':f'configs/r9_derpp_challenge/split_cifar10_classil_derpp_s{s}.json',
             'output':f'{OUTPUT}/split_cifar10_classil_derpp_s{s}'} for s in range(201,211)]


def comparators():return [r for r in read(ROOT/ORIGINAL)['runs'] if r['id'].startswith('split_cifar10_classil_')]


def expected_config(seed):
    old=read(ROOT/f'configs/r9_followup/split_cifar10_classil_er_s{seed}.json')
    return {**old,'run_type':'posthoc_descriptive_derpp_challenge','method':'derpp',
            'der_alpha':.5,'der_beta':.5,'release_current_weight_control':False}


def inputs():
    names={ORIGINAL,RESOURCE,'audit/r9/measure_derpp_challenge.py',
           'third_party/r9_mammoth_reference/SOURCE.json','third_party/r9_mammoth_reference/derpp.py',
           'third_party/r9_mammoth_reference/LICENSE'}
    for r in comparators():
        names.add(r['config_path']);names.add(r['output']+'/artifacts.json')
        names.update(r['output']+'/'+n for n in read(ROOT/r['output']/'artifacts.json')['files'])
    names.update('data/r9_official/cifar-10-batches-py/'+n for n in ['batches.meta','test_batch',*[f'data_batch_{i}' for i in range(1,6)]])
    return names


def validate_manifest(m):
    if m['settings']!=SETTINGS or m['run_count']!=10 or m.get('engineering_ready') is not True:
        raise ValueError('Campaign settings/readiness mismatch')
    if [{k:r[k] for k in ['id','config_path','output']} for r in m['runs']]!=planned_runs():
        raise ValueError('Run matrix mismatch')
    if m['comparators']!=comparators():raise ValueError('Comparator matrix mismatch')
    if sha(ROOT/ORIGINAL)!=ORIGINAL_SHA:raise ValueError('Original freeze changed')
    expected_sources={**read(ROOT/ORIGINAL)['source_sha256'],**{n:sha(ROOT/n) for n in SOURCES}}
    if m['source_sha256']!=expected_sources:raise ValueError('Source coverage/drift')
    if set(m['input_sha256'])!=inputs():raise ValueError('Input coverage drift')
    for name,h in m['input_sha256'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=h:raise ValueError('Input drift '+name)
    for name,h in read(ROOT/ORIGINAL)['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen engine drift '+name)
    for r,seed in zip(m['runs'],SETTINGS['seeds']):
        if sha(ROOT/r['config_path'])!=r['config_sha256'] or read(ROOT/r['config_path'])!=expected_config(seed):
            raise ValueError('New config drift')
    when=datetime.fromisoformat(m['frozen_utc'])
    if when.tzinfo is None or when>datetime.now(timezone.utc):raise ValueError('Freeze time invalid')


def verify_old_runs():
    from experiments.r9_followup.provenance import validate_finished_run
    source=read(ROOT/ORIGINAL)['source_sha256']
    for r in comparators():validate_finished_run(ROOT/r['output'],read(ROOT/r['config_path']),source)


def worker():
    from experiments.r9_followup import engine
    from experiments.r9_followup.provenance import validate_finished_run
    import torch
    m=read(ROOT/MANIFEST);validate_manifest(m);verify_old_runs()
    if not sys.flags.no_user_site:raise RuntimeError('Invoke Python with -s')
    locked=dict(line.split('==',1) for line in (ROOT/'requirements-r9-windows-lock.txt').read_text(encoding='utf-8-sig').splitlines() if '==' in line and not line.startswith('#'))
    for package in ['torch','torchvision','numpy','pillow']:
        if importlib.metadata.version(package)!=locked[package]:raise ValueError('Environment drift '+package)
    source=read(ROOT/ORIGINAL)['source_sha256'];out=ROOT/OUTPUT;records=[]
    for r in m['runs']:
        cfg=read(ROOT/r['config_path'])
        result=engine.execute(cfg,str(ROOT/'data'),str(ROOT/r['output']))
        validate_finished_run(ROOT/r['output'],cfg,source)
        totals={k:sum(t[b]['counts'][k] for t in result['tasks'] for b in ['actual','probe','fresh'])
                for k in ['optimizer_steps','current_examples','replay_examples','forward_examples']}
        if totals!={'optimizer_steps':9420,'current_examples':600000,'replay_examples':200896,'forward_examples':800896}:
            raise ValueError('Complete three-branch counts mismatch')
        old=ROOT/f"results/r9_followup/split_cifar10_classil_er_s{cfg['seed']}"
        for tid in range(5):
            a=torch.load(ROOT/r['output']/f'fresh_task_{tid:03d}.pt',map_location='cpu',weights_only=True)
            b=torch.load(old/f'fresh_task_{tid:03d}.pt',map_location='cpu',weights_only=True)
            if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) for k in a):raise ValueError('Matched fresh differs from original')
        records.append({'id':r['id'],'counts':totals,'fresh_matches_old_er':True,
                        'artifacts_sha256':sha(ROOT/r['output']/'artifacts.json')})
        print(json.dumps({'completed_histories':len(records),'total_histories':10}),flush=True)
    validate_manifest(m)
    if sum(r['counts']['optimizer_steps'] for r in records)!=94200:raise ValueError('Matrix training count mismatch')
    save(out/'complete.json',{'status':'training_complete_pending_independent_analysis','finished_utc':utc(),
         'manifest_sha256':sha(ROOT/MANIFEST),'runs':records,'histories':10,
         'optimizer_updates':94200,'training_forward_images':8008960})
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['console.log']}
    save(out/'campaign_artifacts.json',files)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:
        try:worker()
        except BaseException:
            save(ROOT/OUTPUT/'failure.json',{'utc':utc(),'traceback':traceback.format_exc().replace(str(ROOT),'[REPO]')});raise
        return
    m=read(ROOT/MANIFEST);validate_manifest(m);out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False)
    save(out/'started.json',{'utc':utc(),'manifest_sha256':sha(ROOT/MANIFEST),'limit_seconds':3300})
    shutil.copy2(ROOT/MANIFEST,out/'frozen_manifest.json')
    for name in m['source_sha256']:
        dest=out/'campaign_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP']:env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/'console.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-s','-X','utf8',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:code=child.wait(timeout=max(1,3300-(time.monotonic()-START)));status='complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:child.kill();child.wait();code=None;status='timed_out'
    save(out/'watchdog.json',{'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-START,'limit_seconds':3300})
    if status!='complete':raise RuntimeError('DER++ matrix incomplete; retain partial evidence')
    print(json.dumps(read(out/'watchdog.json')))


if __name__=='__main__':main()
