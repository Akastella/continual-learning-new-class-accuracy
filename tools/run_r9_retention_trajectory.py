"""Fixed validation-selected trajectories; all selection seals precede test inference."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,shutil,subprocess,traceback,platform,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
START=time.monotonic()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.r9_followup import engine
from experiments.r9_retention_trajectory import core,evaluation
MANIFEST='configs/r9_retention_trajectory/manifest.json';OUTPUT='results/r9_retention_trajectory';ORIGINAL='configs/r9_followup/manifest.json'
SETTINGS={'kind':'posthoc_external_validation_selected_trajectory','seeds':list(range(201,211)),'methods':['sgd','er'],'task':4,'steps':[0,1,2,4,8,16,32,64,128,157,314,471,628],'thresholds_pct':[20,10,0],'primary_threshold_pct':20,'split_seed_base':20260915,'per_class_validation':100,'per_class_test':100,'limit_seconds':3300,'total_optimizer_updates':12560,'validation_endpoints':260,'candidates':'all10','batch_eval':512}
SOURCES=['tools/run_r9_retention_trajectory.py','tools/freeze_r9_retention_trajectory.py','tools/prepare_r9_retention_split.py','experiments/r9_retention_trajectory/core.py','experiments/r9_retention_trajectory/evaluation.py','analysis/audit_r9_retention_trajectory.py','analysis/replay_r9_retention_trajectory.py','tests/test_r9_retention_selection.py','tests/test_r9_retention_trajectory_core.py','protocol/R9_RETENTION_TRAJECTORY.md','requirements-r9-windows-lock.txt']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def runs():return [r for r in read(ROOT/ORIGINAL)['runs'] if r['id'].startswith('split_cifar10_classil_')]
def inputs():
    names={ORIGINAL,'third_party/r9_cifar101_reference/SOURCE.json','third_party/r9_cifar101_reference/README.md','third_party/r9_cifar101_reference/LICENSE','data/r9_cifar101/cifar10.1_v6_data.npy','data/r9_cifar101/cifar10.1_v6_labels.npy','audit/r9/POSITIVE_RETENTION_DESIGN_REVIEW.md','audit/r9/RETENTION_TRAJECTORY_RESOURCE.json'}
    for r in runs():
        names.add(r['config_path'])
        names.update(r['output']+'/'+n for n in ['pre_task_004.pt','probe_task_004.pt','post_task_004.pt','result.json','data_manifest.json','artifacts.json'])
    names.update('data/r9_official/cifar-10-batches-py/data_batch_'+str(i) for i in range(1,6))
    names.add('configs/r9_retention_trajectory/split.json')
    return names
def validate(m):
    if m['settings']!=SETTINGS or set(m['input_sha256'])!=inputs() or not set(SOURCES).issubset(m['source_sha256']):raise ValueError('freeze coverage drift')
    if sha(ROOT/ORIGINAL)!='bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e':raise ValueError('original freeze drift')
    for group in ['input_sha256','source_sha256']:
        for n,h in m[group].items():
            p=(ROOT/n).resolve()
            if not p.is_relative_to(ROOT) or sha(p)!=h:raise ValueError('frozen drift '+n)
    dt=datetime.fromisoformat(m['frozen_utc'])
    if dt.tzinfo is None or dt>datetime.now(timezone.utc):raise ValueError('freeze chronology')
def inference(model,x):
    model.eval();zs=[]
    with torch.no_grad():
        for s in range(0,len(x),512):zs.append(model(x[s:s+512].cuda()).cpu().numpy())
    return np.concatenate(zs)
def worker():
    m=read(ROOT/MANIFEST);validate(m);out=ROOT/OUTPUT
    if not sys.flags.no_user_site:raise RuntimeError('Use -s')
    locked=dict(x.split('==',1) for x in (ROOT/'requirements-r9-windows-lock.txt').read_text(encoding='utf-8-sig').splitlines() if '==' in x and not x.startswith('#'))
    for name in ['torch','torchvision','numpy','pillow']:
        if importlib.metadata.version(name)!=locked[name]:raise ValueError('environment '+name)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'torchvision':importlib.metadata.version('torchvision'),'pillow':importlib.metadata.version('pillow'),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'fp32':True,'threads':2,'evaluation_batch':512,'no_user_site':True,'deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True,'matmul_tf32':False,'cudnn_tf32':False,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
    x,labels,val_ids,test_ids=evaluation.external_data(ROOT/'data/r9_cifar101')
    split=read(ROOT/'configs/r9_retention_trajectory/split.json')
    if val_ids.tolist()!=split['validation_indices'] or test_ids.tolist()!=split['test_indices']:raise ValueError('explicit split drift')
    np.savez_compressed(out/'split.npz',validation_indices=val_ids,test_indices=test_ids,original_labels=labels)
    choices={};validation_hashes={};total_steps=0
    for i,r in enumerate(runs(),1):
        cfg=read(ROOT/r['config_path']);raw=read(ROOT/r['output']/'result.json');rd=out/r['id'];rd.mkdir()
        tx,ty,proxy,provenance=core.load_train_only(ROOT/'data',cfg,4);del proxy
        dm=read(ROOT/r['output']/'data_manifest.json');archived=dm['tasks'][4]['splits']['train']
        if any(provenance[k]!=archived[k] for k in ['official_indices','images_sha256','labels_sha256']):raise ValueError('original current train mismatch')
        if core.candidate_steps(len(ty),cfg)!=SETTINGS['steps']:raise ValueError('step grid')
        mapped=np.argsort(np.random.default_rng(cfg['class_order_seed']).permutation(10))[labels].astype(np.int64)
        arrays={'indices':val_ids,'labels':mapped[val_ids]};rows=[]
        def callback(step,model,path):
            z=inference(model,x[val_ids]);arrays['logits_'+str(step)]=z
            rows.append({'step':step,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),'scores':evaluation.scores(z,mapped[val_ids])})
        model=engine.local_model(cfg,cfg['init_seed'],'cuda',torch.load(ROOT/r['output']/'pre_task_004.pt',map_location='cpu',weights_only=True))
        summary=core.train_trajectory(model,tx,ty,cfg,4,rd/'checkpoints',SETTINGS['steps'],callback=callback)
        for item in summary['checkpoints']:item['path']=(rd/'checkpoints'/Path(item['path']).name).relative_to(ROOT).as_posix()
        expected=torch.load(ROOT/r['output']/'probe_task_004.pt',map_location='cpu',weights_only=True)
        if not all(torch.equal(v.cpu(),expected[k]) for k,v in model.state_dict().items()):raise ValueError('legacy terminal tensor mismatch')
        if summary['batch_order_sha256']!=raw['tasks'][4]['probe']['batch_order_sha256'] or summary['final_state_sha256']!=raw['tasks'][4]['probe']['final_state_sha256']:raise ValueError('original trajectory identity')
        total_steps+=summary['optimizer_steps_executed'];np.savez_compressed(rd/'validation.npz',**arrays)
        save(rd/'trajectory.json',{'run_id':r['id'],'seed':cfg['seed'],'method':cfg['method'],'task':4,'summary':summary,'train_provenance':provenance,'validation':rows,'validation_npz_sha256':sha(rd/'validation.npz'),'original_terminal_tensor_exact':True})
        choices[r['id']]={str(q):evaluation.choose(rows,q) for q in SETTINGS['thresholds_pct']}
        for name in ['validation.npz','trajectory.json']:validation_hashes[(rd/name).relative_to(out).as_posix()]=sha(rd/name)
        del model,tx,ty,arrays
        print(json.dumps({'stage':'validation','completed_histories':i,'optimizer_updates':total_steps}),flush=True)
    if total_steps!=12560:raise ValueError('training matrix incomplete')
    save(out/'selection.json',choices)
    save(out/'selection_seal.json',{'created_utc':utc(),'selection_sha256':sha(out/'selection.json'),'validation_files_sha256':validation_hashes,'rule':'max validation current subject to validation old threshold; ties smallest step; empty null'})
    seal_hash=sha(out/'selection_seal.json');save(out/'test_started.json',{'utc':utc(),'selection_seal_sha256':seal_hash})
    test_endpoints=0
    for i,r in enumerate(runs(),1):
        cfg=read(ROOT/r['config_path']);rd=out/r['id'];mapped=np.argsort(np.random.default_rng(cfg['class_order_seed']).permutation(10))[labels].astype(np.int64)
        arrays={'indices':test_ids,'labels':mapped[test_ids]};records=[]
        chosen=sorted({0,628}|{s for s in choices[r['id']].values() if s is not None})
        targets=[('step_'+str(s),rd/'checkpoints'/f'step_{s:06d}.pt',True) for s in chosen]+[('actual',ROOT/r['output']/'post_task_004.pt',False)]
        for key,path,nested in targets:
            state=torch.load(path,map_location='cpu',weights_only=True);state=state['model'] if nested else state
            model=engine.local_model(cfg,cfg['init_seed'],'cuda',state);before=engine.state_hash(model);z=inference(model,x[test_ids]);arrays['logits_'+key]=z
            if engine.state_hash(model)!=before:raise ValueError('test mutation')
            records.append({'key':key,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),'state_sha256':before,'scores':evaluation.scores(z,mapped[test_ids])});del model
        np.savez_compressed(rd/'test.npz',**arrays)
        save(rd/'test.json',{'run_id':r['id'],'records':records,'selection_seal_sha256':seal_hash,'npz_sha256':sha(rd/'test.npz')});test_endpoints+=len(records)
        print(json.dumps({'stage':'test','completed_histories':i}),flush=True)
    validate(m)
    if sha(out/'selection_seal.json')!=seal_hash:raise ValueError('selection seal changed')
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['console.log']}
    save(out/'artifacts.json',files)
    save(out/'complete.json',{'status':'complete','histories':20,'optimizer_updates':total_steps,'validation_endpoints':260,'test_endpoints':test_endpoints,'manifest_sha256':sha(ROOT/MANIFEST),'finished_utc':utc()})
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:
        try:worker()
        except BaseException:
            save(ROOT/OUTPUT/'failure.json',{'utc':utc(),'traceback':traceback.format_exc().replace(str(ROOT),'[REPO]')});raise
        return
    m=read(ROOT/MANIFEST);validate(m);out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False)
    save(out/'started.json',{'utc':utc(),'manifest_sha256':sha(ROOT/MANIFEST)})
    shutil.copy2(ROOT/MANIFEST,out/'frozen_manifest.json')
    for name in m['source_sha256']:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP']:env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    status='failed';code=None
    with (out/'console.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-s','-X','utf8',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:code=process.wait(timeout=max(1,3300-(time.monotonic()-START)));status='complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:process.kill();process.wait();status='timed_out'
    save(out/'watchdog.json',{'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-START,'limit_seconds':3300})
    if status!='complete':raise RuntimeError('Incomplete trajectory matrix')
    print(json.dumps(read(out/'watchdog.json')))
if __name__=='__main__':main()
