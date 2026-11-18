"""Task-7 operators copied exactly from the executed candidate."""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F
shared = Path(__file__).resolve().parents[3] / "experiments/paa/cross_group_supervision"
sys.path.insert(0, str(shared))
import portable_math as previous
core=previous.core
TASK=7
OLD=70
SEEN=80
CONDITIONS=('old_restricted','old_blend05','mixed_full','mixed_restricted')
class Task7Loss:
    def __init__(self, condition):
        assert condition in CONDITIONS
        self.condition = condition
        self.calls = 0

    def cross_entropy(self, logits, labels):
        assert logits.shape == (len(labels), SEEN) and len(labels) % 2 == 0
        n = len(labels) // 2
        if self.calls == 0:
            assert bool(((labels[:n] >= OLD) & (labels[:n] < SEEN)).all())
            if self.condition.startswith('mixed'):
                assert bool((labels[n:n+n//2] < OLD).all())
                assert bool(((labels[n+n//2:] >= OLD) & (labels[n+n//2:] < SEEN)).all())
            else:
                assert bool((labels[n:] < OLD).all())
        self.calls += 1
        if self.condition == 'mixed_full':
            return F.cross_entropy(logits, labels)
        if self.condition.endswith('restricted'):
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove[:n, :OLD] = True
            return F.cross_entropy(logits.masked_fill(remove, -1e9), labels)
        return .5 * (.95 * F.cross_entropy(logits[:n, OLD:SEEN], labels[:n]-OLD)
                      + .05 * F.cross_entropy(logits[:n], labels[:n])
                      + F.cross_entropy(logits[n:], labels[n:]))

def train_one(model, dataset, images, labels, seed, condition, epochs, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    adapter = Task7Loss(condition)
    previous_f, previous_plan = core.F, core.data.epoch_plan
    core.F = SimpleNamespace(pad=F.pad, cross_entropy=adapter.cross_entropy)
    core.data.epoch_plan = previous.mixed_plan if condition.startswith('mixed') else previous.ORIGINAL_PLAN
    try:
        record = core.train_branch(model, dataset, images, labels, seed, TASK, 'er',
                                   condition, epochs, output, deadline, first_task=False)
    finally:
        core.F, core.data.epoch_plan = previous_f, previous_plan
    assert adapter.calls == record['optimizer_steps']
    mixed = condition.startswith('mixed')
    record.update(seed=seed, task=TASK, condition=condition,
                  scheduled_current_presentations=epochs*5000,
                  old_replay_presentations=epochs*(2500 if mixed else 5000),
                  revisited_current_presentations=epochs*(2500 if mixed else 0),
                  test_payload_loaded=False)
    return record
