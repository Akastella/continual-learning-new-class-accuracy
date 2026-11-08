"""Freeze/run the R9 decisive follow-up; preserve all evidence and failures."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.r9_followup.provenance import validate_finished_run,validate_manifest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'experiments/r9_followup').glob('*.py'))}


def freeze():
    path=ROOT/'configs/r9_followup/manifest.json'
    if path.exists():raise FileExistsError('Already frozen')
    runs=[]
    for benchmark in ['split_cifar10_classil','split_cifar10_taskaware']:
        for seed in range(201,211):
            for method in ['sgd','er']:
                cfg={'run_type':'prospective_followup_after_discovery','benchmark':benchmark,'method':method,'seed':seed,
                     'init_seed':seed+10000,'order_seed':seed+30000,'class_order_seed':seed+40000,
                     'tasks':5,'epochs':4,'batch_size':64,'lr':.01,'width':256,'max_train':None,'max_test':None,
                     'memory_size':500,'replay_batch':32,'release_current_weight_control':True,'threads':2,'device':'cuda'}
                key=f'{benchmark}_{method}_s{seed}';cfgpath=path.parent/f'{key}.json'
                cfgpath.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
                runs.append({'id':key,'config_path':str(cfgpath.relative_to(ROOT)).replace('\\','/'),'config_sha256':sha(cfgpath),'output':'results/r9_followup/'+key})
    manifest={'frozen_utc':datetime.now(timezone.utc).isoformat(),'run_count':len(runs),'runs':runs,'source_sha256':sources(),
              'plan_path':'protocol/R9_FOLLOWUP_PROTOCOL.md','plan_sha256':sha(ROOT/'protocol/R9_FOLLOWUP_PROTOCOL.md'),'manager_sha256':sha(Path(__file__)),
              'analysis_sha256':{name:sha(ROOT/name) for name in ['analysis/summarize_r9_followup.py','analysis/summarize_r9_pilot.py','analysis/r9_statistics.py']}}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'runs':len(runs),'manifest_sha256':sha(path)}))


def run():
    path=ROOT/'configs/r9_followup/manifest.json';manifest=json.loads(path.read_text());validate_manifest(manifest)
    ledger=ROOT/'audit/r9/FOLLOWUP_CAMPAIGN_PROCESS.json';records=[]
    for rec in manifest['runs']:
        if manifest['source_sha256']!=sources() or manifest['plan_sha256']!=sha(ROOT/manifest['plan_path']) or manifest['manager_sha256']!=sha(Path(__file__)):
            raise RuntimeError('Source/plan/manager changed after freeze')
        cfgpath=ROOT/rec['config_path'];out=ROOT/rec['output']
        if sha(cfgpath)!=rec['config_sha256']:raise RuntimeError('Config changed')
        cfg=json.loads(cfgpath.read_text())
        if (out/'result.json').exists():
            validate_finished_run(out,cfg,manifest['source_sha256']);print(json.dumps({'id':rec['id'],'state':'validated_existing'}),flush=True);continue
        if (out/'run_started.json').exists():raise RuntimeError('Partial run needs investigation')
        out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','experiments.r9_followup.engine','--config',str(cfgpath),'--data-root','data','--output',str(out)]
        print(json.dumps({'id':rec['id'],'state':'starting'}),flush=True)
        with (out/'console.log').open('w',encoding='utf-8') as f:
            p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            process={'pid':p.pid,'command':command,'launched_utc':datetime.now(timezone.utc).isoformat()}
            (out/'process.json').write_text(json.dumps(process,indent=2),encoding='utf-8');code=p.wait()
        records.append({'id':rec['id'],'returncode':code,'finished_utc':datetime.now(timezone.utc).isoformat()})
        ledger.write_text(json.dumps({'status':'running' if code==0 else 'failed','runs_finished_this_invocation':records},indent=2),encoding='utf-8')
        if code:raise RuntimeError(f'Failed run {rec["id"]}')
        validate_finished_run(out,cfg,manifest['source_sha256'])
        print(json.dumps({'id':rec['id'],'state':'finished'}),flush=True)
    ledger.write_text(json.dumps({'status':'complete','runs_finished_this_invocation':records,'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    print('FOLLOWUP_MATRIX_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
