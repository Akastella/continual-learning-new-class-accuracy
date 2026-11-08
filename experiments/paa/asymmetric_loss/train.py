"""Transfer the author's asymmetric CE rule to the existing compact-CNN protocol."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT
sys.path.insert(0, str(ROOT / 'source'))
from r9_followup import engine

ARTIFACT = ROOT.parent.parent / 'artifact'
EXISTING_RESULTS = Path(r'E:\Projects\iclr 27\results\r9_followup')
DEADLINE = float('inf')


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    engine.atomic_json(Path(path), obj)


def check_time():
    if time.perf_counter() > DEADLINE:
        raise TimeoutError('Prespecified 55 minute wall-clock limit reached')


def mask_logits(logits, present, seen):
    """Exact author rule: present plus [max(seen), number_outputs)."""
    keep = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    keep[present] = True
    keep[max(seen):] = True
    return logits.masked_fill(~keep[None, :], -1e9), keep


@torch.no_grad()
def evaluate(model, task, device):
    model.eval()
    x, y = task['test']
    full_correct = pair_correct = competition_count = pair_wrong = 0
    full_loss = 0.
    for start in range(0, len(y), 512):
        check_time()
        bx, by = x[start:start+512].to(device), y[start:start+512].to(device)
        logits = model(bx)
        pred_full = logits.argmax(1)
        classes = torch.tensor(task['classes'], device=device)
        pred_pair = classes[logits[:, classes].argmax(1)]
        full_ok, pair_ok = pred_full == by, pred_pair == by
        full_correct += int(full_ok.sum())
        pair_correct += int(pair_ok.sum())
        competition_count += int((pair_ok & ~full_ok).sum())
        pair_wrong += int((~pair_ok).sum())
        full_loss += float(F.cross_entropy(logits, by, reduction='sum'))
    if pair_correct - full_correct != competition_count:
        raise AssertionError('Restriction must preserve every correct full prediction')
    return {'n':len(y), 'all10_correct':full_correct, 'current2_correct':pair_correct,
            'all10_accuracy_pct':100*full_correct/len(y),
            'current2_accuracy_pct':100*pair_correct/len(y),
            'competition_errors':competition_count, 'current_pair_errors':pair_wrong,
            'all10_cross_entropy':full_loss/len(y)}


def train(model, task, cfg, device, mode='sgd', memory=None, seen=None):
    x,y = task['train']; tid=task['id']
    opt=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=0.,weight_decay=0.)
    seen = set() if seen is None else seen
    counts={'optimizer_steps':0,'current_examples':0,'replay_examples':0,'forward_examples':0}
    orders=[]; losses=[]; started=time.perf_counter(); mask_sizes={}
    for epoch in range(cfg['epochs']):
        model.train()
        ids=torch.randperm(len(y),generator=torch.Generator().manual_seed(cfg['order_seed']*100000+tid*100+epoch))
        orders.append(engine.digest_tensor(ids))
        inc_sum=re_sum=total_sum=0.; steps=0
        for start in range(0,len(y),cfg['batch_size']):
            check_time()
            ix=ids[start:start+cfg['batch_size']]
            bx,by=x[ix].to(device),y[ix].to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(bx)
            arrival_logits=logits.detach().clone()
            present=by.unique(); seen.update(int(v) for v in present.cpu().tolist())
            if mode=='er_ace_style':
                incoming,keep=mask_logits(logits,present,seen)
                key=','.join(str(v) for v in torch.where(keep)[0].cpu().tolist())
                mask_sizes[key]=mask_sizes.get(key,0)+1
            else:
                incoming=logits
            inc_loss=F.cross_entropy(incoming,by); loss=inc_loss
            replay_loss=None
            counts['forward_examples']+=len(by)
            if memory is not None:
                draw=memory.sample(cfg['replay_batch'],device)
                if draw is not None:
                    rx,ry,rt,rz=draw
                    replay_loss=F.cross_entropy(model(rx),ry)
                    if mode=='er_concat':
                        loss=(len(by)*inc_loss+len(ry)*replay_loss)/(len(by)+len(ry))
                    elif mode in ('er_equal_sum','er_ace_style'):
                        loss=inc_loss+replay_loss
                    else:
                        raise ValueError(mode)
                    counts['replay_examples']+=len(ry);counts['forward_examples']+=len(ry)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} seed{cfg["seed"]} task{tid} epoch{epoch}')
            loss.backward();opt.step()
            counts['optimizer_steps']+=1;counts['current_examples']+=len(by)
            if memory is not None:
                memory.add_batch(bx,by,tid,arrival_logits)
            inc_sum+=float(inc_loss.detach());re_sum+=float(replay_loss.detach()) if replay_loss is not None else 0.
            total_sum+=float(loss.detach());steps+=1
        losses.append({'epoch':epoch+1,'mean_step_incoming_loss':inc_sum/steps,
                       'mean_step_replay_loss':re_sum/steps,'mean_step_total_loss':total_sum/steps})
    if device.type=='cuda':torch.cuda.synchronize()
    return {'counts':counts,'current_orders':orders,'seconds':time.perf_counter()-started,
            'epoch_losses':losses,'allowed_output_steps':mask_sizes,'seen_classes':sorted(seen)}


def cpu_loss_checks():
    z=torch.arange(10,dtype=torch.float64)[None,:].requires_grad_()
    masked,keep=mask_logits(z,torch.tensor([2,3]),{0,1,2,3})
    F.cross_entropy(masked,torch.tensor([2])).backward()
    assert keep.tolist()==[False,False]+[True]*8
    assert z.grad[0,:2].eq(0).all() and z.grad[0,4:].gt(0).all()
    _,last=mask_logits(z,torch.tensor([8,9]),set(range(10)))
    assert last.tolist()==[False]*8+[True,True]
    _,missing=mask_logits(z,torch.tensor([2]),{0,1,2,3})
    assert missing.tolist()==[False,False]+[True]*8
    dump(ROOT/'loss_rule_checks.json',{'created_utc':now(),'status':'passed',
         'earlier_seen_absent_outputs_have_zero_current_gradient':True,
         'future_outputs_have_nonzero_current_negative_gradient':True,
         'highest_seen_label_is_retained_even_if_absent':True})


def verify_reference(tasks,cfg,device):
    small={**cfg,'epochs':1}
    task={**tasks[0],'train':(tasks[0]['train'][0][:128],tasks[0]['train'][1][:128])}
    a=engine.local_model(cfg,cfg['init_seed'],device)
    b=engine.local_model(cfg,cfg['init_seed'],device)
    ma=engine.Reservoir(500,cfg['seed']+77001)
    mb=engine.Reservoir(500,cfg['seed']+77001)
    reference=engine.train_branch(a,task,small,device,'er',ma)
    candidate=train(b,task,small,device,'er_concat',mb)
    same=all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    assert same and reference['counts']==candidate['counts']
    assert reference['batch_order_sha256']==candidate['current_orders']
    dump(ROOT/'reference_check.json',{'created_utc':now(),'status':'passed',
         'description':'Two minibatches, one epoch: local unmasked concatenated loss matches original engine exactly',
         'parameter_tensors_equal':same,'counts':candidate['counts']})


def config_for(seed):
    p=ARTIFACT/'configs'/'r9_followup'/f'split_cifar10_classil_er_s{seed}.json'
    cfg=json.loads(p.read_text())
    expected={'benchmark':'split_cifar10_classil','seed':seed,'init_seed':10000+seed,
              'order_seed':30000+seed,'class_order_seed':40000+seed,'tasks':5,'epochs':4,
              'batch_size':64,'lr':.01,'width':256,'memory_size':500,'replay_batch':32}
    assert all(cfg[k]==v for k,v in expected.items())
    return cfg


def main():
    global DEADLINE, ROOT, ARTIFACT, EXISTING_RESULTS
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--existing-results',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args()
    ARTIFACT=args.artifact_root.resolve();EXISTING_RESULTS=args.existing_results.resolve()
    ROOT=args.output_root.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    cpu_loss_checks()
    if args.check_only:return
    if (ROOT/'run_started.json').exists():raise FileExistsError('Existing run: inspect instead of overwriting')
    started=time.perf_counter();DEADLINE=started+55*60
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda')
    meta={'started_utc':now(),'pid':os.getpid(),'max_minutes':55,'python':platform.python_version(),
          'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name(),
          'run_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(ROOT/'run_started.json',meta);print(json.dumps({'started':meta}),flush=True)
    records=[];raw=[]
    def record(seed,method,tid,endpoint,eval_tid,ev):
        raw.append({'seed':seed,'method':method,'task':tid,'endpoint':endpoint,
                    'evaluation_task':eval_tid,**ev})
    try:
        for seed in range(201,206):
            cfg=config_for(seed)
            tasks,provenance=engine.make_stream(cfg,args.data_root)
            dump(ROOT/f'data_seed_{seed}.json',{'config':cfg,'provenance':provenance,
                 'tasks':[{k:v for k,v in t.items() if k not in ('train','test')} for t in tasks]})
            if seed==201:verify_reference(tasks,cfg,device)
            for method,old_name in [('sgd','sgd'),('er_concat','er')]:
                old=EXISTING_RESULTS/f'split_cifar10_classil_{old_name}_s{seed}'
                results=json.loads((old/'result.json').read_text())
                for k in ('seed','init_seed','order_seed','class_order_seed','tasks','epochs','batch_size','lr','width'):
                    assert cfg[k]==results['config'][k],(seed,method,k)
                for tid,task in enumerate(tasks):
                    rr={'seed':seed,'method':method,'task':tid,'source':'existing independent history',
                        'usual_counts':results['tasks'][tid]['actual']['counts'],
                        'current_only_counts':results['tasks'][tid]['probe']['counts']}
                    for endpoint,filename in [('usual',f'post_task_{tid:03d}.pt'),('current_only',f'probe_task_{tid:03d}.pt')]:
                        state=torch.load(old/filename,map_location='cpu',weights_only=True)
                        model=engine.local_model(cfg,cfg['init_seed'],device,state)
                        ev=evaluate(model,task,device);rr[endpoint]=ev
                        key='actual' if endpoint=='usual' else 'probe'
                        assert ev['all10_correct']==results['tasks'][tid][key]['trace'][-1]['evaluation']['correct']
                        record(seed,method,tid,endpoint,tid,ev)
                        if endpoint=='usual':
                            rr['earlier_tasks']=[]
                            for et in range(tid):
                                ee=evaluate(model,tasks[et],device);rr['earlier_tasks'].append(ee)
                                assert ee['all10_correct']==results['tasks'][tid]['post_task_evaluation'][et]['correct']
                                record(seed,method,tid,'usual_old',et,ee)
                        del model
                    records.append(rr)
                print(json.dumps({'seed':seed,'method':method,'status':'existing endpoints scored'}),flush=True)
            for method in ('er_equal_sum','er_ace_style'):
                out=ROOT/'runs'/f'{method}_s{seed}';out.mkdir(parents=True,exist_ok=False)
                model=engine.local_model(cfg,cfg['init_seed'],device)
                memory=engine.Reservoir(cfg['memory_size'],seed+77001);seen=set()
                for tid,task in enumerate(tasks):
                    check_time();pre=copy.deepcopy(model.state_dict())
                    torch.save(pre,out/f'pre_task_{tid:03d}.pt')
                    current=engine.local_model(cfg,cfg['init_seed'],device,pre)
                    c_log=train(current,task,cfg,device)
                    c_eval=evaluate(current,task,device)
                    torch.save(current.state_dict(),out/f'current_only_task_{tid:03d}.pt');del current
                    a_log=train(model,task,cfg,device,method,memory,seen)
                    a_eval=evaluate(model,task,device)
                    assert c_log['current_orders']==a_log['current_orders']
                    rr={'seed':seed,'method':method,'task':tid,'source':'new independent history',
                        'usual_counts':a_log['counts'],'current_only_counts':c_log['counts'],
                        'usual':a_eval,'current_only':c_eval,'usual_training':a_log,
                        'current_only_training':c_log,'earlier_tasks':[]}
                    record(seed,method,tid,'usual',tid,a_eval);record(seed,method,tid,'current_only',tid,c_eval)
                    for et in range(tid):
                        ee=evaluate(model,tasks[et],device);rr['earlier_tasks'].append(ee)
                        record(seed,method,tid,'usual_old',et,ee)
                    torch.save(model.state_dict(),out/f'post_task_{tid:03d}.pt')
                    dump(out/f'task_{tid:03d}.json',rr);records.append(rr)
                    dump(ROOT/'partial_records.json',records)
                    print(json.dumps({'seed':seed,'method':method,'task':tid,'usual':a_eval,
                                      'current_only':c_eval,'seconds':a_log['seconds']+c_log['seconds']}),flush=True)
                torch.save(memory.state_dict(),out/'final_memory.pt');del model,memory
            dump(ROOT/f'seed_{seed}.json',[r for r in records if r['seed']==seed])
        dump(ROOT/'records.json',records)
        with (ROOT/'raw_counts.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
        dump(ROOT/'execution.json',{**meta,'finished_utc':now(),'seconds':time.perf_counter()-started,
                                   'status':'complete','method_task_records':len(records),'raw_count_rows':len(raw)})
        print(json.dumps({'status':'complete','seconds':time.perf_counter()-started}),flush=True)
    except Exception as exc:
        dump(ROOT/'partial_records.json',records)
        dump(ROOT/'failure.json',{'failed_utc':now(),'type':type(exc).__name__,'message':str(exc),
                                'seconds':time.perf_counter()-started,'completed_records':len(records)})
        raise


if __name__=='__main__':main()
