"""Portable train-only entry: HISTORY_ROOT DATA_ROOT OUTPUT_DIR --job mixed|blend."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import portable_math as op

SCRIPT_ROOT=Path(__file__).resolve().parent
core=op.core
CONDITIONS={'mixed':('mixed_full','mixed_restricted'),'blend':('old_blend05',)}


def source_inputs(history_root, output):
    # Sources and their manifest belong to the distributed code, not output.
    core.validate_manifest(SCRIPT_ROOT/'original_manifest.json')
    assert core.read_json(history_root/'batch.json')['status']=='complete'
    rows=[]
    for seed in range(3101,3106):
        old=history_root/'jobs'/f'seed{seed}_er'
        complete=core.read_json(old/'training_complete.json')
        actual=next(r for r in complete['branches'] if r['task']==9 and r['branch']=='actual')
        common=next(r for r in complete['branches'] if r['task']==9 and r['branch']=='probe')
        path=old/'checkpoints/pre_task_09.pt'
        pre=torch.load(path,map_location='cpu',weights_only=True)
        assert core.state_digest(pre['model'])==pre['state_sha256']==actual['start_state_sha256']==common['start_state_sha256']
        rows.append({'seed':seed,'directory':str(old),'pre_state_sha256':pre['state_sha256'],
                     'pre_file_sha256':core.data.file_digest(path),'actual':actual,'common':common})
    core.write_json(output/'inputs.json',{'source_manifest_sha256':core.data.file_digest(SCRIPT_ROOT/'original_manifest.json'),
                    'history_manifest_sha256':core.data.file_digest(history_root/'manifest.json'),'seeds':rows})
    return rows


def model_from(path,seed):
    model=core.make_model(seed+10000,'cuda').to(memory_format=torch.channels_last)
    saved=core.restore(path,model)
    return model,saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history_root',type=Path,help='Completed original R15 results, containing jobs/seed3101_er etc.')
    parser.add_argument('data_root',type=Path,help='Official CIFAR-100 archive plus extracted train/meta files')
    parser.add_argument('output_dir',type=Path,help='A writable fresh output directory; needs no preplaced files')
    parser.add_argument('--job',choices=('mixed','blend'),required=True)
    parser.add_argument('--check-inputs-only',action='store_true',help='CPU source/pre-state identity check without training or test loading')
    args=parser.parse_args();args.history_root=args.history_root.resolve();args.data_root=args.data_root.resolve()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    marker=output/f'{args.job}_started.json'
    if marker.exists(): raise FileExistsError('Existing attempt; retain outputs and inspect '+str(marker))
    inputs=source_inputs(args.history_root,output)
    if args.check_inputs_only:
        check={'status':'passed','pre_states_verified':5,'manifest_resolved_from_code_directory':True,
               'output_directory_requires_preplaced_files':False,'plan_document_required':False,
               'training_performed':False,'test_payload_loaded':False}
        core.write_json(output/'inputs_check.json',check);print(json.dumps(check));return
    started=time.monotonic();deadline=started+3300
    meta={'utc':core.utc(),'pid':os.getpid(),'job':args.job,'max_seconds':3300,
          'entry_sha256':core.data.file_digest(__file__),
          'operator_sha256':core.data.file_digest(SCRIPT_ROOT/'portable_math.py'),'test_payload_loaded':False}
    core.write_json(marker,meta)
    try:
        core.write_json(output/f'{args.job}_environment.json',core.environment())
        # The loader never extracts or deserializes the official test payload.
        dataset=core.data.load_train(args.data_root,download=False)
        assert dataset.provenance['archive_sha256']==core.ARCHIVE_SHA256
        core.write_json(output/f'{args.job}_train_provenance.json',dataset.provenance)
        images=torch.from_numpy(dataset.images.copy()).to('cuda');records=[]
        for item in inputs:
            seed=item['seed'];_,inverse=core.data.class_mapping(seed)
            labels=torch.as_tensor(inverse[dataset.labels],dtype=torch.long,device='cuda')
            for condition in CONDITIONS[args.job]:
                path=Path(item['directory'])/'checkpoints/pre_task_09.pt'
                model,_=model_from(path,seed)
                run_dir=output/'runs'/condition/f'seed{seed}'
                record=op.train_one(model,dataset,images,labels,seed,condition,70,run_dir,deadline)
                assert record['start_state_sha256']==item['pre_state_sha256']
                assert record['optimizer_steps']==5530 and record['current_presentations']==record['replay_presentations']==350000
                for observed,expected in zip(record['epoch_records'],item['actual']['epoch_records']):
                    for key in ('epoch','lr','next_lr','optimizer_steps','current_presentations','replay_presentations',
                                'current_order_sha256','memory_indices_sha256','augmentation_keys_sha256'):
                        assert observed[key]==expected[key]
                    if condition=='old_blend05':assert observed['replay_order_sha256']==expected['replay_order_sha256']
                record['checkpoint']=core.save_checkpoint(run_dir/'endpoint.pt',model,
                         {'seed':seed,'task':9,'epochs':70,'branch':condition})
                core.write_json(run_dir/'training.json',record);records.append(record);del model
                print(json.dumps({'seed':seed,'condition':condition,'status':'complete','seconds':record['wall_seconds']}),flush=True)
        core.write_json(output/f'{args.job}_complete.json',{**meta,'status':'complete','finished_utc':core.utc(),
                            'seconds':time.monotonic()-started,'records':records})
    except BaseException as exc:
        core.write_json(output/f'{args.job}_failure.json',{'utc':core.utc(),'type':type(exc).__name__,
                        'message':str(exc),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
