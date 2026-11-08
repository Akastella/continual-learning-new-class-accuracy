"""Unfrozen current-only trajectory machinery; no test-data loading here."""
import copy
import hashlib
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiments.r9_followup.engine import local_model, state_hash
from experiments.r9_followup.data import digest_tensor


def epoch_order(n, config, task_id, epoch):
    g = torch.Generator().manual_seed(config.get('order_seed', config['seed'])*100000 + task_id*100 + epoch)
    return torch.randperm(n, generator=g)


def candidate_steps(n, config):
    per_epoch = math.ceil(n/config['batch_size'])
    total = per_epoch*config['epochs']
    return sorted(set([0, *[s for s in [1,2,4,8,16,32,64,128] if s <= total],
                       *[per_epoch*e for e in range(1,config['epochs']+1)]]))


def cifar_images_from_flat(flat):
    """Match torchvision's HWC view and torch clone without changing strides."""
    return torch.from_numpy(flat.reshape(-1,3,32,32).transpose(0,2,3,1)).clone()


def load_train_only(data_root, config, task_id=4):
    """Decode only the five official training batches, preserving torchvision order."""
    from torchvision.datasets import CIFAR10
    folder = Path(data_root)/'r9_official'/'cifar-10-batches-py'
    data, labels, hashes = [], [], {}
    for name, expected_md5 in CIFAR10.train_list:
        path = folder/name
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != expected_md5:
            raise ValueError(f'Train batch checksum mismatch: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
        row = pickle.loads(raw, encoding='latin1')
        data.append(row['data']); labels.extend(row['labels'])
    # torchvision keeps the transposed HWC view's NCHW backing strides. A HWC
    # contiguous copy has equal values but changes CUDA convolution arithmetic.
    images = cifar_images_from_flat(np.vstack(data))
    labels = torch.tensor(labels, dtype=torch.long)
    if images.shape != (50000,32,32,3):
        raise ValueError('Official train split shape mismatch')
    order = np.random.default_rng(config['class_order_seed']).permutation(10)
    inverse = torch.tensor(np.argsort(order), dtype=torch.long)
    original = order[2*task_id:2*task_id+2]
    ids = torch.where((labels == int(original[0])) | (labels == int(original[1])))[0]
    if config.get('max_train') is not None:
        raise ValueError('This loader requires the complete current task')
    def normalize(rows):
        x = images[rows].permute(0,3,1,2).float()/255
        mu = torch.tensor([.4914,.4822,.4465]).view(1,-1,1,1)
        sd = torch.tensor([.2470,.2435,.2616]).view(1,-1,1,1)
        return (x-mu)/sd
    x, y = normalize(ids), inverse[labels[ids]]
    witness_ids = torch.cat([torch.where(labels == c)[0][:100] for c in range(10)])
    # Repeat each real TRAIN image twice: cost proxy only, never an evaluation set.
    proxy_x = normalize(witness_ids).repeat(2,1,1,1)
    provenance = {'decoded_splits':['train'], 'batch_sha256':hashes,
        'official_train_images_sha256':digest_tensor(images),
        'official_train_labels_sha256':digest_tensor(labels),
        'class_order_original_ids':order.tolist(), 'task_id':task_id,
        'official_indices':ids.tolist(), 'images_sha256':digest_tensor(x),
        'labels_sha256':digest_tensor(y), 'proxy_unique_train_images':1000,
        'proxy_forward_images_per_callback':2000}
    return x, y, proxy_x, provenance


def train_trajectory(model, x, y, config, task_id, output, candidates, callback=None,
                     resume=None, stop_step=None, max_seconds=3300):
    """Exact frozen current-only SGD with saved cursors and read-only callbacks.

    Callback signature is (step, model, checkpoint_path). Evaluation code is owned
    by the caller. RNG/mode are restored, parameter/buffer mutation is rejected.
    Resume uses a checkpoint containing the same config/data/order fingerprints.
    """
    if config['benchmark'] != 'split_cifar10_classil' or config.get('momentum',0) != 0:
        raise ValueError('Only class-IL, zero-momentum current-only SGD is supported')
    if x.dtype != torch.float32 or y.dtype != torch.int64 or len(x) != len(y):
        raise ValueError('Expected aligned FP32 input and int64 labels')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    per_epoch = math.ceil(len(y)/config['batch_size'])
    total = per_epoch*config['epochs']
    candidates = list(candidates)
    if candidates != sorted(set(candidates)) or not candidates or candidates[0] != 0 or candidates[-1] != total:
        raise ValueError('Candidate steps must be sorted unique and include 0 and endpoint')
    if any(type(s) is not int or s < 0 or s > total for s in candidates):
        raise ValueError('Invalid candidate step')
    stop = total if stop_step is None else stop_step
    if type(stop) is not int or not 0 <= stop <= total:
        raise ValueError('Invalid stop step')
    device = next(model.parameters()).device
    opt = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0., weight_decay=0.)
    orders = [epoch_order(len(y),config,task_id,e) for e in range(config['epochs'])]
    identity = {'config':config, 'task_id':task_id, 'n':len(y), 'x_sha256':digest_tensor(x),
                'y_sha256':digest_tensor(y), 'order_sha256':[digest_tensor(o) for o in orders],
                'candidates':candidates}
    start_step = 0
    if resume is not None:
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        if saved['identity'] != identity: raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        start_step = saved['step']
        if start_step > stop: raise ValueError('Resume after stop')
    started = time.perf_counter(); emitted = []; callback_seconds = 0.
    def emit(step):
        nonlocal callback_seconds
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        sha = state_hash(model)
        path = output/f'step_{step:06d}.pt'
        if path.exists(): raise FileExistsError(path)
        torch.save({'step':step, 'identity':identity, 'model':state,
                    'optimizer':copy.deepcopy(opt.state_dict()), 'state_sha256':sha},path)
        if callback is not None:
            before = time.perf_counter(); mode = model.training
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            try:
                with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                    model.eval(); callback(step,model,path)
                if state_hash(model) != sha: raise RuntimeError('Callback mutated model parameters or buffers')
            finally:
                model.load_state_dict(state); model.train(mode)
            callback_seconds += time.perf_counter()-before
        emitted.append({'step':step, 'path':path.name, 'state_sha256':sha})
    if resume is None: emit(0)
    for step in range(start_step,stop):
        if time.perf_counter()-started > max_seconds: raise TimeoutError('Trajectory time budget exceeded')
        epoch, batch = divmod(step,per_epoch)
        ix = orders[epoch][batch*config['batch_size']:(batch+1)*config['batch_size']]
        model.train(); bx,by = x[ix].to(device),y[ix].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(bx),by)*1.
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite current-only loss')
        loss.backward(); opt.step()
        if step+1 in candidates or step+1 == stop: emit(step+1)
    if device.type == 'cuda': torch.cuda.synchronize()
    return {'start_step':start_step, 'stop_step':stop, 'total_steps':total,
            'optimizer_steps_executed':stop-start_step, 'batch_order_sha256':identity['order_sha256'],
            'final_state_sha256':state_hash(model), 'seconds':time.perf_counter()-started,
            'callback_seconds':callback_seconds, 'checkpoints':emitted}
