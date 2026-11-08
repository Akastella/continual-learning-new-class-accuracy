"""Actual algorithm, current-only history probe and matched fresh control.

Every branch is forked from a saved pre-task parameter state. The history probe
uses current data and plain SGD; its endpoint is not substituted for algorithm
performance. Replay objectives and sample/step budgets are logged explicitly.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
from .data import make_stream, digest_tensor
from .provenance import write_artifact_manifest


def validate_config(config):
    allowed={'target_remapping','permuted_mnist','stationary_mnist','split_cifar10_classil','split_cifar10_taskaware','split_cifar10_shared2'}
    if config.get('benchmark') not in allowed: raise ValueError('Unsupported benchmark')
    if config.get('method') not in {'sgd','er','derpp'}: raise ValueError('Unsupported method')
    for key in ('seed','init_seed','stream_seed','order_seed','data_seed','class_order_seed'):
        value=config.get(key,config.get('seed'))
        if type(value) is not int or value<0: raise ValueError(f'Invalid {key}')
    for key in ('tasks','epochs','batch_size'):
        if type(config.get(key)) is not int or config[key]<1: raise ValueError(f'Invalid {key}')
    for key in ('width','threads','memory_size','replay_batch','max_train','max_test'):
        value=config.get(key)
        if value is not None and (type(value) is not int or value<1): raise ValueError(f'Invalid {key}')
    if not np.isfinite(config['lr']) or config['lr']<=0: raise ValueError('Invalid learning rate')
    if config.get('momentum',0.)!=0.: raise ValueError('This protocol supports zero momentum only')
    if config['benchmark'].startswith('split_cifar10') and config['tasks']>5: raise ValueError('CIFAR10 allows at most five class pairs')
    if config['method'] in {'er','derpp'} and any(config.get(k) is None for k in ('memory_size','replay_batch')): raise ValueError('Replay capacity and batch required')
    if config.get('release_current_weight_control',False):
        if config['method'] not in {'sgd','er'} or any(config.get(k) is None for k in ('memory_size','replay_batch')):
            raise ValueError('Release-control campaign requires SGD/ER and explicit memory/batch sizes')
    for key in ('der_alpha','der_beta'):
        if not np.isfinite(config.get(key,.1)) or config.get(key,.1)<0: raise ValueError(f'Invalid {key}')


class Network(nn.Module):
    def __init__(self, benchmark, width=256):
        super().__init__()
        self.cifar=benchmark.startswith('split_cifar10')
        if self.cifar:
            self.backbone=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(2048,width),nn.ReLU())
        else:
            self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(784,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.head=nn.Linear(width,2 if benchmark=='split_cifar10_shared2' else 10)

    def forward(self,x):
        return self.head(self.backbone(x))


def selected_logits(model,x,task_ids,benchmark):
    z=model(x)
    if benchmark=='split_cifar10_taskaware':
        if isinstance(task_ids,int):
            return z[:,2*task_ids:2*task_ids+2]
        inds=task_ids[:,None]*2+torch.arange(2,device=z.device)[None,:]
        return z.gather(1,inds)
    return z


def target_labels(y,benchmark):
    return y%2 if benchmark=='split_cifar10_taskaware' else y


def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode()); h.update(bytes.fromhex(digest_tensor(t)))
    return h.hexdigest()


def local_model(config,seed,device,state=None):
    # Creating an auxiliary branch never advances global training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        m=Network(config['benchmark'],config.get('width',256))
    if state is not None: m.load_state_dict(state)
    return m.to(device)


@torch.no_grad()
def evaluate(model,task,benchmark,device):
    model.eval()
    x,y=task['test']; correct=0; loss_sum=0.; predictions=[]
    for start in range(0,len(y),512):
        bx,by=x[start:start+512].to(device),target_labels(y[start:start+512].to(device),benchmark)
        z=selected_logits(model,bx,task['id'],benchmark)
        pred=z.argmax(1)
        correct+=int((pred==by).sum()); loss_sum+=float(F.cross_entropy(z,by,reduction='sum'))
        predictions.append(pred.cpu())
    return {'n':len(y),'correct':correct,'accuracy_pct':100*correct/len(y),
            'cross_entropy':loss_sum/len(y),'prediction_sha256':digest_tensor(torch.cat(predictions))}


class Reservoir:
    """Uniform streaming reservoir with fixed total capacity; stores arrival-time logits."""
    def __init__(self,capacity,seed):
        self.capacity=capacity
        self.storage_rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
        self.sample_rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
        self.seen=0; self.rows=[]

    def add_batch(self,x,y,task_id,logits):
        for i in range(len(y)):
            self.seen+=1
            j=len(self.rows) if len(self.rows)<self.capacity else int(self.storage_rng.integers(self.seen))
            if j<self.capacity:
                row=(x[i].detach().cpu().clone(),y[i].detach().cpu().clone(),task_id,logits[i].detach().cpu().clone())
                if j==len(self.rows): self.rows.append(row)
                else: self.rows[j]=row

    def sample(self,n,device):
        if not self.rows: return None
        ids=self.sample_rng.choice(len(self.rows),size=min(n,len(self.rows)),replace=False)
        rows=[self.rows[i] for i in ids]
        return (torch.stack([r[0] for r in rows]).to(device),torch.stack([r[1] for r in rows]).to(device),
                torch.tensor([r[2] for r in rows],device=device),torch.stack([r[3] for r in rows]).to(device))

    def state_dict(self):
        return {'capacity':self.capacity,'seen':self.seen,'rows':self.rows,
                'storage_rng':self.storage_rng.bit_generator.state,'sample_rng':self.sample_rng.bit_generator.state}


def train_branch(model,task,config,device,method='sgd',memory=None,order_seed=None):
    bench=config['benchmark']; x,y=task['train']; tid=task['id']
    opt=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config.get('momentum',0.),weight_decay=0.)
    trace=[{'epoch':0,'evaluation':evaluate(model,task,bench,device)}]
    counts={'optimizer_steps':0,'current_examples':0,'replay_examples':0,'forward_examples':0}
    orders=[]; started=time.perf_counter()
    weight_only=config.get('_er_current_weight_only',False)
    current_extra=config.get('_er_current_extra',False)
    if current_extra:counts['extra_current_examples']=0
    virtual_memory_len=config.get('_er_memory_len_at_start',0)
    extra_rng=torch.Generator().manual_seed(config.get('order_seed',config['seed'])*100000+tid*100+990001)
    current_factors=[]
    if (weight_only or current_extra) and (method!='sgd' or memory is not None): raise ValueError('Release controls require current-only SGD')
    if weight_only and current_extra:raise ValueError('Release controls are separate operators')
    for epoch in range(config['epochs']):
        model.train()
        g=torch.Generator().manual_seed((config.get('order_seed',config['seed']) if order_seed is None else order_seed)*100000+tid*100+epoch)
        ids=torch.randperm(len(y),generator=g); orders.append(digest_tensor(ids))
        for start in range(0,len(y),config['batch_size']):
            ix=ids[start:start+config['batch_size']]
            bx,by=x[ix].to(device),y[ix].to(device)
            opt.zero_grad(set_to_none=True)
            z=selected_logits(model,bx,tid,bench)
            arrival_logits=z.detach().clone()
            loss=F.cross_entropy(z,target_labels(by,bench))
            factor=len(by)/(len(by)+min(config['replay_batch'],virtual_memory_len)) if weight_only else 1.
            loss=loss*factor
            counts['forward_examples']+=len(by)
            if current_extra:
                nr=min(config['replay_batch'],virtual_memory_len)
                if nr:
                    # Fresh current examples replace the replay draw. The frozen
                    # current batch order and number of optimizer steps stay fixed.
                    extra_ids=torch.randint(len(y),(nr,),generator=extra_rng)
                    ex,ey=x[extra_ids].to(device),y[extra_ids].to(device)
                    extra_loss=F.cross_entropy(selected_logits(model,ex,tid,bench),target_labels(ey,bench))
                    loss=(len(by)*loss+nr*extra_loss)/(len(by)+nr)
                    factor=len(by)/(len(by)+nr)
                    counts['extra_current_examples']+=nr;counts['forward_examples']+=nr
            if memory is not None:
                r=memory.sample(config['replay_batch'],device)
                if r is not None:
                    rx,ry,rt,rz=r
                    out=selected_logits(model,rx,rt,bench)
                    if method=='er':
                        # Mean CE of concatenated current/replay batch, as in
                        # the reference ER algorithm. Do not silently give the
                        # smaller replay batch the same total weight as current.
                        replay_loss=F.cross_entropy(out,target_labels(ry,bench))
                        if config.get('_er_restore_current_weight',False):
                            loss=loss+len(ry)/(len(by)+len(ry))*replay_loss
                        else:
                            loss=(len(by)*loss+len(ry)*replay_loss)/(len(by)+len(ry))
                            factor=len(by)/(len(by)+len(ry))
                    elif method=='derpp':
                        loss=loss+config.get('der_alpha',.1)*F.mse_loss(out,rz)
                        # Separate reservoir draw for labelled replay, as in DER++.
                        r2=memory.sample(config['replay_batch'],device)
                        rx2,ry2,rt2,_=r2
                        loss=loss+config.get('der_beta',.5)*F.cross_entropy(selected_logits(model,rx2,rt2,bench),target_labels(ry2,bench))
                        counts['replay_examples']+=len(ry2); counts['forward_examples']+=len(ry2)
                    else: raise ValueError(method)
                    counts['replay_examples']+=len(ry); counts['forward_examples']+=len(ry)
            current_factors.append(factor)
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss {tid=} {epoch=}')
            loss.backward(); opt.step()
            counts['optimizer_steps']+=1; counts['current_examples']+=len(by)
            if memory is not None:
                # Each example is an arrival in the explicitly multi-epoch stream.
                memory.add_batch(bx,by,tid,arrival_logits)
            if weight_only or current_extra:virtual_memory_len=min(config['memory_size'],virtual_memory_len+len(by))
        trace.append({'epoch':epoch+1,'evaluation':evaluate(model,task,bench,device)})
    if device.type=='cuda': torch.cuda.synchronize()
    return {'trace':trace,'counts':counts,'batch_order_sha256':orders,'seconds':time.perf_counter()-started,
            'operator':{'method':method,'er_current_weight_only':weight_only,'er_current_extra':current_extra,
                        'extra_examples_source':'current_task' if current_extra else ('reservoir' if memory is not None else None),
                        'er_restore_current_weight':config.get('_er_restore_current_weight',False),
                        'initial_virtual_memory_rows':config.get('_er_memory_len_at_start',0) if weight_only or current_extra else None,
                        'core_current_factor_min':min(current_factors),'core_current_factor_max':max(current_factors),
                        'core_current_factor_sha256':digest_tensor(torch.tensor(current_factors,dtype=torch.float64))},
            'final_state_sha256':state_hash(model)}


def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)


def execute(config,data_root,output):
    validate_config(config)
    output=Path(output)
    if (output/'result.json').exists(): raise FileExistsError('Completed result exists; choose a new run directory.')
    if (output/'run_started.json').exists(): raise FileExistsError('Partial run exists; inspect it and choose a new run directory.')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config.get('threads',2))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    source_dir=Path(__file__).parent
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob('*.py')}
    meta={'config':config,'source_sha256':sources,'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                       'device':str(device),'device_name':torch.cuda.get_device_name() if device.type=='cuda' else platform.processor()}}
    # Exclusive creation claims ownership, even if two launchers race.
    with (output/'run_started.json').open('x',encoding='utf-8') as f:
        json.dump(meta,f,indent=2,allow_nan=False)
    for p in source_dir.glob('*.py'):
        dest=output/'source'/p.name; dest.parent.mkdir(exist_ok=True); dest.write_bytes(p.read_bytes())
    tasks,provenance=make_stream(config,data_root)
    atomic_json(output/'data_manifest.json',{'provenance':provenance,'tasks':[{k:v for k,v in t.items() if k not in ('train','test')} for t in tasks]})
    init_seed=config.get('init_seed',config['seed'])
    model=local_model(config,init_seed,device)
    initial=copy.deepcopy(model.state_dict()); torch.save(initial,output/'initial.pt')
    memory=Reservoir(config['memory_size'],config['seed']+77001) if config['method'] in ('er','derpp') else None
    records=[]
    for t,task in enumerate(tasks):
        pre=copy.deepcopy(model.state_dict()); torch.save(pre,output/f'pre_task_{t:03d}.pt')
        if memory is not None: torch.save(memory.state_dict(),output/f'pre_task_{t:03d}_memory.pt')
        pre_hash=state_hash(model)
        pre_eval=[evaluate(model,other,config['benchmark'],device) for other in tasks]
        fresh=local_model(config,init_seed,device,initial)
        fresh_log=train_branch(fresh,task,config,device)
        torch.save(fresh.state_dict(),output/f'fresh_task_{t:03d}.pt')
        probe=local_model(config,init_seed,device,pre)
        probe_log=train_branch(probe,task,config,device)
        torch.save(probe.state_dict(),output/f'probe_task_{t:03d}.pt')
        extras={}
        if config.get('release_current_weight_control',False):
            virtual_prior_rows=min(config['memory_size'],sum(len(z['train'][1])*config['epochs'] for z in tasks[:t]))
            weighted_cfg={**config,'_er_current_weight_only':True,'_er_memory_len_at_start':virtual_prior_rows}
            for branch,state in [('weighted_fresh',initial),('weighted_probe',pre)]:
                controlled=local_model(config,init_seed,device,state)
                extras[branch]=train_branch(controlled,task,weighted_cfg,device)
                torch.save(controlled.state_dict(),output/f'{branch}_task_{t:03d}.pt')
                if extras[branch]['batch_order_sha256']!=fresh_log['batch_order_sha256']:raise AssertionError('Control order mismatch')
            extra_cfg={**config,'_er_current_extra':True,'_er_memory_len_at_start':virtual_prior_rows}
            for branch,state in [('compute_fresh',initial),('compute_probe',pre)]:
                controlled=local_model(config,init_seed,device,state)
                extras[branch]=train_branch(controlled,task,extra_cfg,device)
                torch.save(controlled.state_dict(),output/f'{branch}_task_{t:03d}.pt')
            if config['method']=='er':
                controlled=local_model(config,init_seed,device,pre)
                extras['compensated_er']=train_branch(controlled,task,{**config,'_er_restore_current_weight':True},device,'er',copy.deepcopy(memory))
                torch.save(controlled.state_dict(),output/f'compensated_er_task_{t:03d}.pt')
        actual_log=train_branch(model,task,config,device,config['method'],memory)
        for branch,log in extras.items():
            if log['batch_order_sha256']!=actual_log['batch_order_sha256']:raise AssertionError('Control current order mismatch')
            if branch in ('compute_fresh','compute_probe') and config['method']=='er':
                if any(log['counts'][k]!=actual_log['counts'][k] for k in ('optimizer_steps','current_examples','forward_examples')) or log['counts']['extra_current_examples']!=actual_log['counts']['replay_examples']:
                    raise AssertionError('Control training-example budget mismatch')
            if branch=='compensated_er' and log['counts']!=actual_log['counts']:raise AssertionError('Compensated ER budget mismatch')
        if extras:
            if extras['compute_fresh']['counts']!=extras['compute_probe']['counts']:raise AssertionError('Compute controls differ in budget')
            if t==0:
                for prefix in ('weighted','compute'):
                    if extras[prefix+'_fresh']['final_state_sha256']!=extras[prefix+'_probe']['final_state_sha256']:raise AssertionError('Task0 control initialization differs')
        post_eval=[evaluate(model,other,config['benchmark'],device) for other in tasks]
        fresh_acc=fresh_log['trace'][-1]['evaluation']['accuracy_pct']
        probe_acc=probe_log['trace'][-1]['evaluation']['accuracy_pct']
        actual_acc=actual_log['trace'][-1]['evaluation']['accuracy_pct']
        if fresh_log['batch_order_sha256']!=probe_log['batch_order_sha256'] or fresh_log['batch_order_sha256']!=actual_log['batch_order_sha256']:
            raise AssertionError('Current sample order mismatch')
        if config['method']=='sgd' and actual_log['final_state_sha256']!=probe_log['final_state_sha256']:
            raise AssertionError('SGD history probe and actual branch diverged')
        if t==0 and fresh_log['final_state_sha256']!=probe_log['final_state_sha256']:
            raise AssertionError('Task-zero matched identity failed')
        rec={'task':t,'pre_state_sha256':pre_hash,'fresh':fresh_log,'probe':probe_log,'actual':actual_log,**extras,
             'pre_task_evaluation':pre_eval,'post_task_evaluation':post_eval,
             'actual_gap_pp':fresh_acc-actual_acc,'history_gap_pp':fresh_acc-probe_acc,
             'objective_cost_pp':probe_acc-actual_acc}
        records.append(rec); atomic_json(output/f'task_{t:03d}.json',rec)
        torch.save(model.state_dict(),output/f'post_task_{t:03d}.pt')
        if memory is not None: torch.save(memory.state_dict(),output/f'post_task_{t:03d}_memory.pt')
        print(json.dumps({'task':t,'actual_gap_pp':rec['actual_gap_pp'],'history_gap_pp':rec['history_gap_pp'],
                          'objective_cost_pp':rec['objective_cost_pp'],'actual_acc':actual_acc}),flush=True)
    result={**meta,'finished_utc':datetime.now(timezone.utc).isoformat(),'tasks':records,'status':'complete'}
    atomic_json(output/'result.json',result)
    write_artifact_manifest(output)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--data-root',default='data'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text())
    try:
        execute(config,args.data_root,args.output)
    except Exception as exc:
        failure=Path(args.output)/'failure.json'
        start=failure.parent/'run_started.json'
        owned=start.exists() and json.loads(start.read_text()).get('pid')==os.getpid()
        if owned and not failure.exists():
            atomic_json(failure,{'error_type':type(exc).__name__,'message':str(exc),'failed_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__': main()
