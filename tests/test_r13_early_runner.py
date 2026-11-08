"""The dense observation schedule must preserve the original SGD trajectory."""
import copy

import numpy as np
import torch

from experiments.r9_retention_trajectory.core import train_trajectory
from experiments.r13_early_retention.core import DENSE, SPARSE
from tools.run_r13_early_retention import normalize


def test_dense_early_observation_preserves_all_six_old_nodes(tmp_path):
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(712)
    x = torch.randn(10000, 3, generator=g)
    y = torch.randint(8, 10, (10000,), generator=g)
    torch.manual_seed(22)
    sparse = torch.nn.Linear(3, 10)
    dense = copy.deepcopy(sparse)
    cfg = dict(benchmark='split_cifar10_classil', seed=201, order_seed=30201,
               batch_size=64, epochs=4, lr=.01)
    old = train_trajectory(sparse, x, y, cfg, 4, tmp_path/'old', list(SPARSE), stop_step=16)
    calls = []

    def observe(step, model, path):
        calls.append(step)
        # Evaluation consumes RNG but is not allowed to perturb training.
        _ = model(torch.randn(9, 3))

    new = train_trajectory(dense, x, y, cfg, 4, tmp_path/'new', list(DENSE),
                           callback=observe, stop_step=16)
    assert calls == list(range(17))
    assert new['optimizer_steps_executed'] == old['optimizer_steps_executed'] == 16
    assert new['batch_order_sha256'] == old['batch_order_sha256']
    for step in (0, 1, 2, 4, 8, 16):
        left = torch.load(tmp_path/'old'/f'step_{step:06d}.pt', weights_only=True)
        right = torch.load(tmp_path/'new'/f'step_{step:06d}.pt', weights_only=True)
        assert left['state_sha256'] == right['state_sha256']
        assert all(torch.equal(left['model'][k], v) for k, v in right['model'].items())


def test_new_data_normalization_preserves_frozen_evaluation_arithmetic_and_layout():
    pixels = np.arange(8*32*32*3, dtype=np.uint8).reshape(8, 32, 32, 3)
    ids = np.array([1, 3, 6], dtype=np.int64)
    old = torch.from_numpy(pixels).permute(0, 3, 1, 2).float()/255
    old = (old-torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1))/torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1)
    new = normalize(pixels)
    assert torch.equal(new, old) and new.stride() == old.stride()
    assert torch.equal(new[ids], old[ids]) and new[ids].stride() == old[ids].stride()
