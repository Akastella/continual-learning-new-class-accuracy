"""Exact copies of the three executed operators; no output or protocol paths."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from experiments.r15_normal_budget import core
ORIGINAL_PLAN=core.data.epoch_plan


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
