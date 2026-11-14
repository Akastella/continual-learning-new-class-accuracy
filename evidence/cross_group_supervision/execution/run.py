"""Prespecified three-condition continuation; unchanged archived core engine."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
import reference_run as ref

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'asymmetric_transfer_c100'
core = ref.core
ORIGINAL_PLAN = core.data.epoch_plan
CONDITIONS = ('mixed_full', 'mixed_restricted', 'old_blend05')


def dump(path, value): core.write_json(path, value)


def mixed_plan(labels, seed, task, epoch, per_class, allocation):
    rows, plan = ORIGINAL_PLAN(labels, seed, task, epoch, per_class, allocation)
    changed = []
    for row in rows:
        current, old = row['current_indices'], row['replay_indices']
        assert len(current) == len(old) and len(old) % 2 == 0
        half = len(old)//2
        replay = np.concatenate((old[:half], current[:half]))
        changed.append({**row, 'replay_indices': replay})
    replay = np.concatenate([r['replay_indices'] for r in changed])
    return changed, {**plan, 'original_replay_order_sha256': plan['replay_order_sha256'],
                     'replay_indices': replay.tolist(), 'replay_order_sha256': core.data.array_digest(replay)}


class Loss:
    def __init__(self, condition): self.condition = condition; self.calls = 0

    def cross_entropy(self, z, y):
        assert z.shape == (len(y), 100) and len(y) % 2 == 0
        n = len(y)//2
        if self.calls == 0:
            assert bool((y[:n] >= 90).all())
            if self.condition.startswith('mixed'):
                assert bool((y[n:n+n//2] < 90).all())
                assert bool((y[n+n//2:] >= 90).all())
            else:
                assert bool((y[n:] < 90).all())
        self.calls += 1
        if self.condition == 'mixed_full':
            return F.cross_entropy(z, y)
        if self.condition == 'mixed_restricted':
            remove = torch.zeros_like(z, dtype=torch.bool); remove[:n, :90] = True
            return F.cross_entropy(z.masked_fill(remove, -1e9), y)
        assert self.condition == 'old_blend05'
        # Half the original concatenated mean is current; half is replay.
        return .5 * (.95*F.cross_entropy(z[:n, 90:], y[:n]-90)
                      + .05*F.cross_entropy(z[:n], y[:n])
                      + F.cross_entropy(z[n:], y[n:]))


def train_one(model, dataset, images, labels, seed, condition, epochs, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    adapter = Loss(condition)
    previous_f, previous_plan = core.F, core.data.epoch_plan
    core.F = SimpleNamespace(pad=F.pad, cross_entropy=adapter.cross_entropy)
    core.data.epoch_plan = mixed_plan if condition.startswith('mixed') else ORIGINAL_PLAN
    try:
        record = core.train_branch(model, dataset, images, labels, seed, 9, 'er', condition,
                                   epochs, output, deadline, first_task=False)
    finally:
        core.F, core.data.epoch_plan = previous_f, previous_plan
    assert adapter.calls == record['optimizer_steps']
    mixed = condition.startswith('mixed')
    record.update(condition=condition, seed=seed,
                  scheduled_current_presentations=epochs*5000,
                  old_replay_presentations=epochs*(2500 if mixed else 5000),
                  revisited_current_presentations=epochs*(2500 if mixed else 0),
                  current_full_class_component=(1 if condition=='mixed_full' else .05 if condition=='old_blend05' else 0),
                  test_evaluation_performed=False)
    return record


def loss_and_schedule_checks(dataset):
    # Deterministic synthetic logits test the operator, without test examples.
    generator = torch.Generator().manual_seed(80301)
    z = torch.randn((128,100), generator=generator, dtype=torch.float64, requires_grad=True)
    y_mixed = torch.cat((90+torch.arange(64)%10, torch.arange(32)%90, 90+torch.arange(32)%10))
    y_old = torch.cat((90+torch.arange(64)%10, torch.arange(64)%90))
    errors = {}
    for name, y in [('mixed_full', y_mixed), ('mixed_restricted', y_mixed), ('old_blend05', y_old)]:
        observed = Loss(name).cross_entropy(z, y)
        if name=='mixed_full': expected=F.cross_entropy(z,y)
        elif name=='mixed_restricted': expected=.5*(F.cross_entropy(z[:64,90:],y[:64]-90)+F.cross_entropy(z[64:],y[64:]))
        else: expected=.5*(.95*F.cross_entropy(z[:64,90:],y[:64]-90)+.05*F.cross_entropy(z[:64],y[:64])+F.cross_entropy(z[64:],y[64:]))
        ga, = torch.autograd.grad(observed,z,retain_graph=True)
        gb, = torch.autograd.grad(expected,z,retain_graph=True)
        assert torch.allclose(observed,expected,rtol=0,atol=1e-12)
        assert torch.allclose(ga,gb,rtol=0,atol=1e-12)
        errors[name]={'loss_abs_error':float((observed-expected).detach()),'gradient_max_error':float((ga-gb).abs().max())}
    checks=[]
    for seed in range(3101,3106):
        _, inverse=core.data.class_mapping(seed)
        for epoch in range(70):
            original, op=ORIGINAL_PLAN(dataset.labels,seed,9,epoch,20,'fixed')
            changed, cp=mixed_plan(dataset.labels,seed,9,epoch,20,'fixed')
            old_n=new_n=0
            for a,b in zip(original,changed):
                n=len(a['current_indices']);half=n//2
                assert np.array_equal(a['current_indices'],b['current_indices'])
                assert np.array_equal(a['replay_indices'][:half],b['replay_indices'][:half])
                assert np.array_equal(a['current_indices'][:half],b['replay_indices'][half:])
                mapped=inverse[dataset.labels[b['replay_indices']]]
                assert (mapped[:half]<90).all() and (mapped[half:]>=90).all()
                old_n+=half;new_n+=half
            assert len(changed)==79 and len(changed[-1]['current_indices'])==8
            assert len(changed[-1]['replay_indices'])==8 and old_n==new_n==2500
            checks.append({'seed':seed,'epoch':epoch+1,'current_order_sha256':cp['current_order_sha256'],
                'original_replay_order_sha256':op['replay_order_sha256'],'mixed_replay_order_sha256':cp['replay_order_sha256'],
                'old_presentations':old_n,'revisited_current_presentations':new_n})
    dump(ROOT/'sampling_check.json',{'status':'passed','test_evaluation_performed':False,
         'full_batches':'64 scheduled current / 32 retained old / 32 revisited current',
         'last_batch':'8 scheduled current / 4 retained old / 4 revisited current','loss_checks':errors,'epochs':checks})


def train_job(args):
    job=args.job
    marker=ROOT/f'{job}_started.json'
    if marker.exists(): raise FileExistsError('Existing attempt; preserve and inspect: '+str(marker))
    started=time.monotonic(); deadline=started+55*60
    meta={'pid':os.getpid(),'started_utc':core.utc(),'job':job,'max_seconds':3300,
          'run_sha256':core.data.file_digest(__file__),'plan_sha256':core.data.file_digest(ROOT/'PLAN.md'),
          'test_evaluation_performed':False}
    dump(marker,meta)
    try:
        dump(ROOT/f'{job}_environment.json',core.environment())
        inputs=ref.source_inputs() if job=='mixed' else core.read_json(ROOT/'inputs.json')['seeds']
        dataset=core.data.load_train(args.data_root,download=False)
        assert dataset.provenance['archive_sha256']==core.ARCHIVE_SHA256
        dump(ROOT/f'{job}_train_provenance.json',dataset.provenance)
        images=torch.from_numpy(dataset.images.copy()).to('cuda')
        if job=='mixed':
            loss_and_schedule_checks(dataset)
            timings={}
            seed=3101;_,inverse=core.data.class_mapping(seed)
            labels=torch.as_tensor(inverse[dataset.labels],dtype=torch.long,device='cuda')
            for condition in CONDITIONS:
                model,_=ref.model_from(Path(inputs[0]['directory'])/'checkpoints/pre_task_09.pt',seed)
                record=train_one(model,dataset,images,labels,seed,condition,1,ROOT/'checks'/condition,deadline)
                dump(ROOT/'checks'/condition/'training.json',record)
                timings[condition]=record['wall_seconds'];del model
            projected_mixed=350*(timings['mixed_full']+timings['mixed_restricted'])
            projected_blend=350*timings['old_blend05']
            check={'status':'passed','test_evaluation_performed':False,'one_epoch_seconds':timings,
                   'expected_mixed_seconds':projected_mixed,'expected_blend_seconds':projected_blend,
                   'conservative_mixed_seconds':1.5*projected_mixed+120,
                   'conservative_blend_seconds':1.5*projected_blend+120,
                   'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
            dump(ROOT/'operator_check.json',check);print(json.dumps({'operator_check':check}),flush=True)
            if check['conservative_mixed_seconds']>3300 or check['conservative_blend_seconds']>3300:
                raise RuntimeError('Measured conservative estimate exceeds prespecified 55-minute cap')
        conditions=CONDITIONS[:2] if job=='mixed' else CONDITIONS[2:]
        records=[]
        for item in inputs:
            seed=item['seed'];_,inverse=core.data.class_mapping(seed)
            labels=torch.as_tensor(inverse[dataset.labels],dtype=torch.long,device='cuda')
            for condition in conditions:
                print(json.dumps({'utc':core.utc(),'seed':seed,'condition':condition,'training':'started'}),flush=True)
                out=ROOT/'runs'/condition/f'seed{seed}'
                model,_=ref.model_from(Path(item['directory'])/'checkpoints/pre_task_09.pt',seed)
                record=train_one(model,dataset,images,labels,seed,condition,70,out,deadline)
                assert record['start_state_sha256']==item['pre_state_sha256']
                assert record['optimizer_steps']==5530 and record['current_presentations']==record['replay_presentations']==350000
                for observed,expected in zip(record['epoch_records'],item['actual']['epoch_records']):
                    for key in ('epoch','lr','next_lr','optimizer_steps','current_presentations','replay_presentations',
                                'current_order_sha256','memory_indices_sha256','augmentation_keys_sha256'):
                        assert observed[key]==expected[key],(condition,seed,key,observed['epoch'])
                    if condition=='old_blend05': assert observed['replay_order_sha256']==expected['replay_order_sha256']
                if condition=='mixed_restricted':
                    pair=core.read_json(ROOT/'runs/mixed_full'/f'seed{seed}'/'training.json')
                    assert all(a['replay_order_sha256']==b['replay_order_sha256'] for a,b in zip(record['epoch_records'],pair['epoch_records']))
                record['checkpoint']=core.save_checkpoint(out/'endpoint.pt',model,{'seed':seed,'task':9,'epochs':70,'branch':condition})
                dump(out/'training.json',record);records.append({'seed':seed,**record});del model
                print(json.dumps({'utc':core.utc(),'seed':seed,'condition':condition,'training':'complete',
                                  'seconds':record['wall_seconds'],'job_seconds':time.monotonic()-started}),flush=True)
                dump(ROOT/f'{job}_progress.json',{'utc':core.utc(),'completed':len(records),'records':records,'test_evaluation_performed':False})
        dump(ROOT/f'{job}_complete.json',{**meta,'status':'complete','finished_utc':core.utc(),
             'seconds':time.monotonic()-started,'records':records,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})
        print(json.dumps({'job':job,'status':'complete','seconds':time.monotonic()-started}),flush=True)
    except BaseException as exc:
        dump(ROOT/f'{job}_failure.json',{'utc':core.utc(),'type':type(exc).__name__,'message':str(exc),'seconds':time.monotonic()-started})
        raise


def evaluate(args):
    started=time.monotonic();deadline=started+55*60
    assert not (ROOT/'evaluation_started.json').exists()
    markers={name:core.read_json(ROOT/f'{name}_complete.json') for name in ('mixed','blend')}
    records=[r for v in markers.values() for r in v['records']]
    assert len(records)==15 and {(r['seed'],r['condition']) for r in records}=={(s,c) for s in range(3101,3106) for c in CONDITIONS}
    dump(ROOT/'training_complete.json',{'status':'complete','utc':core.utc(),'records':records,
         'markers_sha256':{n:core.data.file_digest(ROOT/f'{n}_complete.json') for n in markers}})
    dump(ROOT/'evaluation_started.json',{'utc':core.utc(),'pid':os.getpid(),
         'training_complete_sha256':core.data.file_digest(ROOT/'training_complete.json'),'new_models_complete':15})
    core.environment()
    images,original_labels=ref.load_test();reader=ROOT/'reader_data';reader.mkdir(exist_ok=False)
    inputs=core.read_json(ROOT/'inputs.json')['seeds'];files=[]
    for item in inputs:
        seed=item['seed'];order,inverse=core.data.class_mapping(seed);labels=inverse[original_labels]
        for name in ('usual_er','current_mask','common_sgd'):
            source=PRIOR/'reader_data'/f's{seed}_{name}.npz';target=reader/source.name
            shutil.copy2(source,target);assert core.data.file_digest(source)==core.data.file_digest(target)
            files.append({'path':str(target.relative_to(ROOT)),'sha256':core.data.file_digest(target),'reused_source':str(source)})
        for condition in CONDITIONS:
            out=ROOT/'runs'/condition/f'seed{seed}';record=core.read_json(out/'training.json')
            assert core.data.file_digest(out/'endpoint.pt')==record['checkpoint']['file_sha256']
            model,saved=ref.model_from(out/'endpoint.pt',seed)
            assert saved['state_sha256']==record['end_state_sha256']
            z=ref.infer(model,images,deadline);del model
            np.savez_compressed(out/'logits.npz',logits=z)
            ref.save_predictions(reader,seed,condition,z,labels,original_labels,order)
            path=reader/f's{seed}_{condition}.npz'
            files.append({'path':str(path.relative_to(ROOT)),'sha256':core.data.file_digest(path),
                          'checkpoint_sha256':record['checkpoint']['file_sha256'],'logits_sha256':core.data.file_digest(out/'logits.npz')})
        print(json.dumps({'seed':seed,'evaluation':'complete'}),flush=True)
    dump(ROOT/'evaluation_complete.json',{'status':'complete','utc':core.utc(),'files':files,'seconds':time.monotonic()-started})


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',choices=('mixed','blend','evaluate'),required=True)
    parser.add_argument('--input-root',type=Path,default=ref.EXISTING)
    parser.add_argument('--data-root',type=Path,default=ref.DATA_ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args();ROOT=args.output_dir.resolve();ref.ROOT=ROOT
    ref.EXISTING=args.input_root;ref.DATA_ROOT=args.data_root
    if args.job=='evaluate':evaluate(args)
    else:train_job(args)


if __name__=='__main__': main()
