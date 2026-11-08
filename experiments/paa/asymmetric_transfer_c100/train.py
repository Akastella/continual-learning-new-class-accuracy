"""Final CIFAR-100 task: only restrict incoming-image candidate outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tarfile
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_ROOT=Path(__file__).resolve().parent
ROOT=SCRIPT_ROOT
SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
from experiments.r15_normal_budget import core

EXISTING=Path(r'E:\Projects\iclr 27\results\r15_normal_budget_recovery1')
DATA_ROOT=Path(r'E:\Projects\iclr 27\data\r9_external')


def dump(path,value):core.write_json(path,value)


class IncomingLoss:
    def __init__(self,masked):self.masked=masked;self.calls=0
    def cross_entropy(self,z,y):
        assert z.shape==(len(y),100) and len(y)%2==0
        n=len(y)//2
        if self.calls==0:
            assert bool(((y[:n]>=90)&(y[:n]<100)).all()) and bool((y[n:]<90).all())
        self.calls+=1
        if self.masked:
            remove=torch.zeros_like(z,dtype=torch.bool);remove[:n,:90]=True
            z=z.masked_fill(remove,-1e9)
        return F.cross_entropy(z,y)


def branch(model,dataset,images,labels,seed,epochs,out,deadline,masked=None):
    out.mkdir(parents=True,exist_ok=False)
    previous=core.F;adapter=None
    if masked is not None:
        adapter=IncomingLoss(masked)
        core.F=SimpleNamespace(pad=F.pad,cross_entropy=adapter.cross_entropy)
    try:
        record=core.train_branch(model,dataset,images,labels,seed,9,'er','incoming_mask' if masked else 'unmasked_check',
                                 epochs,out,deadline,first_task=False)
    finally:core.F=previous
    if adapter is not None:assert adapter.calls==record['optimizer_steps']
    return record


def model_from(path,seed):
    model=core.make_model(seed+10000,'cuda').to(memory_format=torch.channels_last)
    saved=core.restore(path,model)
    return model,saved


def source_inputs():
    manifest=core.validate_manifest(SCRIPT_ROOT/'original_manifest.json')
    assert core.read_json(EXISTING/'batch.json')['status']=='complete'
    rows=[]
    for seed in range(3101,3106):
        old=EXISTING/'jobs'/f'seed{seed}_er'
        complete=core.read_json(old/'training_complete.json')
        results=core.read_json(old/'result.json')
        assert results['status']=='complete'
        actual=next(r for r in complete['branches'] if r['task']==9 and r['branch']=='actual')
        common=next(r for r in complete['branches'] if r['task']==9 and r['branch']=='probe')
        pre=torch.load(old/'checkpoints'/'pre_task_09.pt',map_location='cpu',weights_only=True)
        assert core.state_digest(pre['model'])==pre['state_sha256']==actual['start_state_sha256']==common['start_state_sha256']
        for label,record in [('actual',actual),('probe',common)]:
            checkpoint=old/'checkpoints'/record['checkpoint']['path']
            assert core.data.file_digest(checkpoint)==record['checkpoint']['file_sha256']
            endpoint=next(e for e in results['endpoints'] if e['task']==9 and e['branch']==label)
            assert core.data.file_digest(old/'evaluation'/endpoint['path'])==endpoint['sha256']
        rows.append({'seed':seed,'directory':str(old),'pre_state_sha256':pre['state_sha256'],
                     'pre_file_sha256':core.data.file_digest(old/'checkpoints'/'pre_task_09.pt'),
                     'actual':actual,'common':common,'baseline_result':results})
    dump(ROOT/'inputs.json',{'source_manifest':str(EXISTING/'manifest.json'),
         'source_manifest_sha256':core.data.file_digest(EXISTING/'manifest.json'),'seeds':rows})
    return rows


def train_matrix(inputs,deadline):
    dataset=core.data.load_train(DATA_ROOT,download=False)
    assert dataset.provenance['archive_sha256']==core.ARCHIVE_SHA256
    images=torch.from_numpy(dataset.images.copy()).to('cuda')
    labels=None;records=[]
    for item in inputs:
        seed=item['seed'];_,inverse=core.data.class_mapping(seed)
        labels=torch.as_tensor(inverse[dataset.labels],dtype=torch.long,device='cuda')
        pre=Path(item['directory'])/'checkpoints'/'pre_task_09.pt'
        if seed==3101:
            a,_=model_from(pre,seed)
            ref=branch(a,dataset,images,labels,seed,1,ROOT/'checks'/'reference',deadline)
            ref_state={k:v.detach().cpu().clone() for k,v in a.state_dict().items()};del a
            b,_=model_from(pre,seed)
            candidate=branch(b,dataset,images,labels,seed,1,ROOT/'checks'/'interface_identity',deadline,masked=False)
            assert all(torch.equal(ref_state[k],b.state_dict()[k].cpu()) for k in ref_state)
            for key in ('current_order_sha256','memory_indices_sha256','replay_order_sha256','augmentation_keys_sha256','train_ce','train_correct'):
                assert ref['epoch_records'][0][key]==candidate['epoch_records'][0][key]
            del b,ref_state
            timing_model,_=model_from(pre,seed)
            timing=branch(timing_model,dataset,images,labels,seed,1,ROOT/'checks'/'masked_timing',deadline,masked=True)
            del timing_model
            projected=timing['wall_seconds']*350*1.5+120
            dump(ROOT/'operator_check.json',{'status':'passed','unmasked_parameters_exact':True,
                 'one_epoch_reference_seconds':ref['wall_seconds'],'one_epoch_masked_seconds':timing['wall_seconds'],
                 'conservative_projected_matrix_seconds':projected,'test_evaluation_performed':False})
            print(json.dumps({'operator_check':'passed','masked_epoch_seconds':timing['wall_seconds'],
                              'conservative_projected_seconds':projected}),flush=True)
            if projected>50*60:raise RuntimeError('Measured conservative estimate exceeds 50 minutes; report before changing protocol')
        model,saved=model_from(pre,seed)
        out=ROOT/'runs'/f'seed{seed}'
        record=branch(model,dataset,images,labels,seed,70,out,deadline,masked=True)
        assert record['start_state_sha256']==item['actual']['start_state_sha256']
        assert record['optimizer_steps']==5530 and record['current_presentations']==record['replay_presentations']==350000
        reference=item['actual']['epoch_records']
        for observed,expected in zip(record['epoch_records'],reference):
            for key in ('epoch','lr','next_lr','optimizer_steps','current_presentations','replay_presentations',
                        'current_order_sha256','memory_indices_sha256','replay_order_sha256','augmentation_keys_sha256'):
                assert observed[key]==expected[key],(seed,key,observed['epoch'])
        record['checkpoint']=core.save_checkpoint(out/'masked_task09.pt',model,
                            {'seed':seed,'task':9,'epochs':70,'branch':'incoming_mask'})
        record['same_pre_state_and_training_events_as_original']=True
        dump(out/'training.json',record);records.append({'seed':seed,**record});del model
        print(json.dumps({'seed':seed,'training':'complete','seconds':record['wall_seconds']}),flush=True)
    dump(ROOT/'training_complete.json',{'status':'complete','records':records})
    del images,labels


def load_test():
    with tarfile.open(DATA_ROOT/core.data.ARCHIVE_NAME,'r:gz') as tar:
        payload=tar.extractfile('cifar-100-python/test').read()
    assert hashlib.sha256(payload).hexdigest()==core.TEST_SHA256
    raw=pickle.loads(payload,encoding='bytes')
    images=np.asarray(raw[b'data']).reshape(-1,3,32,32).transpose(0,2,3,1)
    return images,np.asarray(raw[b'fine_labels'],dtype=np.int64)


@torch.no_grad()
def infer(model,images,deadline):
    model.eval();blocks=[];before=core.state_digest(model.state_dict())
    for offset in range(0,10000,256):
        if time.monotonic()>deadline:raise TimeoutError('55 minute job limit reached during evaluation')
        x=torch.from_numpy(images[offset:offset+256].copy()).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255
        x=(x-torch.tensor(core.data.MEAN,device='cuda').view(1,3,1,1))/torch.tensor(core.data.STD,device='cuda').view(1,3,1,1)
        blocks.append(model(x).cpu().numpy())
    assert core.state_digest(model.state_dict())==before
    z=np.concatenate(blocks).astype(np.float32);assert np.isfinite(z).all();return z


def evaluate_matrix(inputs,deadline):
    images,original_labels=load_test();reader=ROOT/'reader_data';reader.mkdir(exist_ok=True)
    for item in inputs:
        seed=item['seed'];old=Path(item['directory']);order,inverse=core.data.class_mapping(seed)
        labels=inverse[original_labels]
        for model_name,source in [('usual_er','actual'),('common_sgd','probe')]:
            with np.load(old/'evaluation'/f'{source}_task_09.npz',allow_pickle=False) as d:
                z=d['logits'].copy();assert np.array_equal(d['mapped_labels'],labels)
            if seed==3101 and model_name=='usual_er':
                model,_=model_from(old/'checkpoints'/'actual_task_09.pt',seed)
                replay=infer(model,images,deadline);del model
                assert np.array_equal(replay,z)
                dump(ROOT/'baseline_inference_check.json',{'status':'passed','seed':seed,'all_logits_exact':True})
            save_predictions(reader,seed,model_name,z,labels,original_labels,order)
        model,_=model_from(ROOT/'runs'/f'seed{seed}'/'masked_task09.pt',seed)
        z=infer(model,images,deadline);del model
        np.savez_compressed(ROOT/'runs'/f'seed{seed}'/'logits.npz',logits=z)
        save_predictions(reader,seed,'current_mask',z,labels,original_labels,order)
        print(json.dumps({'seed':seed,'evaluation':'complete'}),flush=True)


def save_predictions(reader,seed,name,z,labels,original_labels,order):
    np.savez_compressed(reader/f's{seed}_{name}.npz',predictions_all100=z.argmax(1).astype(np.int64),
        predictions_current10=(90+z[:,90:100].argmax(1)).astype(np.int64),mapped_labels=labels,
        original_labels=original_labels,official_test_indices=np.arange(10000,dtype=np.int64),
        class_order=order,seed=np.int64(seed),model_name=np.asarray(name),task=np.int64(9))


def main():
    global ROOT, EXISTING, DATA_ROOT
    ap=argparse.ArgumentParser()
    ap.add_argument('--existing-results',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args()
    EXISTING=args.existing_results.resolve();DATA_ROOT=args.data_root.resolve()
    ROOT=args.output_root.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'run_started.json').exists():raise FileExistsError('Existing attempt; preserve and inspect')
    began=time.monotonic();deadline=began+55*60
    meta={'started_utc':core.utc(),'pid':os.getpid(),'max_minutes':55,
          'run_sha256':core.data.file_digest(__file__)}
    dump(ROOT/'run_started.json',meta)
    try:
        dump(ROOT/'environment.json',core.environment())
        inputs=source_inputs();print(json.dumps({'status':'inputs_verified','pid':os.getpid()}),flush=True)
        train_matrix(inputs,deadline)
        evaluate_matrix(inputs,deadline)
        dump(ROOT/'execution.json',{**meta,'status':'complete','finished_utc':core.utc(),'seconds':time.monotonic()-began})
        print(json.dumps({'status':'complete','seconds':time.monotonic()-began}),flush=True)
    except BaseException as exc:
        dump(ROOT/'failure.json',{'utc':core.utc(),'type':type(exc).__name__,'message':str(exc),'seconds':time.monotonic()-began})
        raise


if __name__=='__main__':main()
