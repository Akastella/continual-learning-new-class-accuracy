"""Frozen follow-up operators with explicit epoch offset and observation only.

Parent: experiments/r9_followup/engine.py SHA256
24f4ef74f231b5e81096bcf8e84911d3e1722fcdcd8ff9368750289fa1133cf2.
Differences: no release controls; epoch offset; per-step draw/teacher hashes;
loss summaries; callback after each epoch. Arithmetic order is unchanged.
"""
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.r9_followup import engine
from experiments.r9_followup.data import digest_tensor

PARENT_SHA256 = '24f4ef74f231b5e81096bcf8e84911d3e1722fcdcd8ff9368750289fa1133cf2'


class Reservoir(engine.Reservoir):
    """Original draw, recording slot identities without additional RNG draws."""
    def sample(self, n, device):
        if not self.rows:
            return None
        ids = self.sample_rng.choice(len(self.rows), size=min(n, len(self.rows)), replace=False)
        self.draw_hashes.append(digest_tensor(torch.as_tensor(ids)))
        rows = [self.rows[i] for i in ids]
        return (torch.stack([r[0] for r in rows]).to(device),
                torch.stack([r[1] for r in rows]).to(device),
                torch.tensor([r[2] for r in rows], device=device),
                torch.stack([r[3] for r in rows]).to(device))

    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.draw_hashes = []


def restore_memory(state):
    if state is None:
        return None
    m = Reservoir(state['capacity'], 0)
    m.seen = state['seen']
    m.rows = copy.deepcopy(state['rows'])
    m.storage_rng.bit_generator.state = copy.deepcopy(state['storage_rng'])
    m.sample_rng.bit_generator.state = copy.deepcopy(state['sample_rng'])
    if len(m.rows) != min(m.capacity, m.seen):
        raise ValueError('Invalid reservoir size')
    return m


def epoch_order(n, config, task_id, epoch):
    g = torch.Generator().manual_seed(config.get('order_seed', config['seed'])*100000 + task_id*100 + epoch)
    return torch.randperm(n, generator=g)


def train_epochs(model, task, config, device, method='sgd', memory=None,
                 start_epoch=4, end_epoch=16, callback=None, deadline=None):
    if method not in {'sgd', 'er', 'derpp'} or (method == 'sgd') != (memory is None):
        raise ValueError('Operator/memory mismatch')
    if config.get('momentum', 0) != 0 or config.get('weight_decay', 0) != 0:
        raise ValueError('Only zero momentum and weight decay')
    if any(config.get(k, False) for k in ('_er_current_weight_only', '_er_current_extra', '_er_restore_current_weight')):
        raise ValueError('Release controls are not budget-extension operators')
    if not 0 <= start_epoch < end_epoch:
        raise ValueError('Invalid epoch range')
    x, y = task['train']; tid = task['id']; bench = config['benchmark']
    opt = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0., weight_decay=0.)
    records = []
    for epoch in range(start_epoch, end_epoch):
        model.train(); started = time.perf_counter()
        ids = epoch_order(len(y), config, tid, epoch)
        counts = dict(optimizer_steps=0, current_examples=0, replay_examples=0, forward_examples=0)
        sums = dict(current_ce=0., objective=0., replay_ce=0., logit_mse=0.)
        steps = []
        for start in range(0, len(y), config['batch_size']):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Whole campaign deadline')
            ix = ids[start:start+config['batch_size']]
            bx, by = x[ix].to(device), y[ix].to(device)
            opt.zero_grad(set_to_none=True)
            z = engine.selected_logits(model, bx, tid, bench)
            arrival_logits = z.detach().clone()
            loss = F.cross_entropy(z, engine.target_labels(by, bench))
            current_ce = float(loss.detach()); replay_ce = 0.; logit_mse = 0.
            loss = loss*1.
            counts['forward_examples'] += len(by)
            draw_start = len(memory.draw_hashes) if memory is not None else 0
            if memory is not None:
                r = memory.sample(config['replay_batch'], device)
                if r is not None:
                    rx, ry, rt, rz = r
                    out = engine.selected_logits(model, rx, rt, bench)
                    if method == 'er':
                        replay_loss = F.cross_entropy(out, engine.target_labels(ry, bench))
                        loss = (len(by)*loss+len(ry)*replay_loss)/(len(by)+len(ry))
                        replay_ce = float(replay_loss.detach())
                    else:
                        mse = F.mse_loss(out, rz)
                        loss = loss+config.get('der_alpha', .1)*mse
                        rx2, ry2, rt2, _ = memory.sample(config['replay_batch'], device)
                        ce = F.cross_entropy(engine.selected_logits(model, rx2, rt2, bench), engine.target_labels(ry2, bench))
                        loss = loss+config.get('der_beta', .5)*ce
                        logit_mse = float(mse.detach()); replay_ce = float(ce.detach())
                        counts['replay_examples'] += len(ry2); counts['forward_examples'] += len(ry2)
                    counts['replay_examples'] += len(ry); counts['forward_examples'] += len(ry)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite objective {tid=} {epoch=}')
            sums['current_ce'] += current_ce; sums['objective'] += float(loss.detach())
            sums['replay_ce'] += replay_ce; sums['logit_mse'] += logit_mse
            loss.backward(); opt.step()
            counts['optimizer_steps'] += 1; counts['current_examples'] += len(by)
            if memory is not None:
                memory.add_batch(bx, by, tid, arrival_logits)
            steps.append({'current_indices_sha256': digest_tensor(ix),
                          'arrival_logits_sha256': digest_tensor(arrival_logits),
                          'replay_slot_sha256': memory.draw_hashes[draw_start:] if memory is not None else []})
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise FloatingPointError('Nonfinite model state')
        record = {'epoch': epoch+1, 'batch_order_sha256': digest_tensor(ids), 'counts': counts,
                  'losses': {'aggregation': 'unweighted_mean_over_optimizer_steps',
                             **{k:v/counts['optimizer_steps'] for k,v in sums.items()}},
                  'steps': steps, 'training_seconds': time.perf_counter()-started,
                  'state_sha256': engine.state_hash(model)}
        if callback is not None:
            before = engine.state_hash(model)
            callback(record, model, memory)
            if engine.state_hash(model) != before:
                raise ValueError('Callback changed model')
        records.append(record)
    return records


def scores(logits, labels, task_ids):
    z = torch.as_tensor(logits); y = torch.as_tensor(labels); tids = torch.as_tensor(task_ids)
    if z.shape != (len(y), 10) or not torch.isfinite(z).all() or set(tids.tolist()) != set(range(5)):
        raise ValueError('Invalid full-five-task evaluation')
    def count(pred, target):
        c = int((pred == target).sum()); n = len(target)
        return {'correct': c, 'n': n, 'accuracy_pct': 100*c/n,
                'prediction_sha256': digest_tensor(pred)}
    pred = z.argmax(1)
    current = tids == 4; old = tids < 4
    return {'all10': {'tasks': [{'task_id': t, **count(pred[tids == t], y[tids == t])} for t in range(5)],
                      'current': count(pred[current], y[current]), 'old': count(pred[old], y[old]),
                      'all': count(pred, y)},
            'current2': count(z[current, 8:10].argmax(1)+8, y[current])}


@torch.no_grad()
def inference(model, tasks, device):
    model.eval(); logits = []; labels = []; indices = []; tids = []
    for task in tasks:
        x, y = task['test']; parts = []
        for start in range(0, len(y), 512):
            parts.append(model(x[start:start+512].to(device)).cpu())
        logits.append(torch.cat(parts).numpy()); labels.append(y.numpy())
        indices.extend(task['splits']['test']['official_indices']); tids.extend([task['id']]*len(y))
    arrays = {'logits': np.concatenate(logits), 'labels': np.concatenate(labels),
              'official_indices': np.asarray(indices, dtype=np.int64), 'task_ids': np.asarray(tids, dtype=np.int64)}
    return arrays, scores(arrays['logits'], arrays['labels'], arrays['task_ids'])
