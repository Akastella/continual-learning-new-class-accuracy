"""Independent functional CNN: new validation and selected test, fixed seed 201."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def forward(x, state):
    for i in [0, 3, 6]:
        x = F.max_pool2d(F.relu(F.conv2d(x, state[f'backbone.{i}.weight'],
            state[f'backbone.{i}.bias'], padding=1)), 2)
    x = F.relu(F.linear(x.flatten(1), state['backbone.10.weight'], state['backbone.10.bias']))
    return F.linear(x, state['head.weight'], state['head.bias'])


def normalized(images):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    return ((x - torch.tensor([.4914, .4822, .4465]).reshape(1, 3, 1, 1)) /
            torch.tensor([.2470, .2435, .2616]).reshape(1, 3, 1, 1))


def replay(output, root=ROOT):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = root / 'results/r13_early_retention_v1'
    manifest = read(raw / 'frozen_manifest.json')
    s = manifest['settings']
    assert sha(__file__) == manifest['source_sha256']['analysis/replay_r13_early_retention.py']
    for name in [s['validation_data'], s['validation_labels'], s['validation_split'], s['test_npz']]:
        assert sha(root / name) == manifest['input_sha256'][name]
    assert sha(raw / 'artifacts.json') == read(raw / 'complete.json')['artifacts_sha256']
    val_ids = np.array(read(root / s['validation_split'])['validation_indices'], dtype=np.int64)
    vx = normalized(np.load(root / s['validation_data'], allow_pickle=False))[val_ids]
    with np.load(root / s['test_npz'], allow_pickle=False) as a:
        tx = normalized(a[s['test_images_key']])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for method in ['sgd', 'er']:
        run_id = f'split_cifar10_classil_{method}_s201'
        for phase, x in [('validation', vx), ('test', tx)]:
            d = read(raw / run_id / (phase + '.json'))
            path = raw / run_id / (phase + '.npz')
            assert sha(path) == d['npz_sha256']
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['indices'], val_ids if phase == 'validation' else np.arange(len(tx)))
                for rec in d['records']:
                    if phase == 'validation' and rec['origin'] != 'r13_new':
                        continue
                    checkpoint = root / rec['checkpoint']
                    assert sha(checkpoint) == rec['checkpoint_sha256']
                    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    if rec['kind'] != 'actual':
                        state = state['model']
                    state = {name: v.cuda() for name, v in state.items()}
                    with torch.no_grad():
                        logits = np.concatenate([forward(x[i:i+512].cuda(), state).cpu().numpy()
                                                 for i in range(0, len(x), 512)])
                    target = z['logits_' + rec['key']]
                    if not np.array_equal(logits, target):
                        raise ValueError((run_id, phase, rec['key'], float(np.abs(logits-target).max())))
                    rows.append(dict(run=run_id, phase=phase, key=rec['key'],
                        checkpoint_sha256=sha(checkpoint), image_evaluations=len(x),
                        logit_elements=int(logits.size), max_absolute_error=0))
    result = dict(status='passed_independent_functional_forward',
        manifest_sha256=sha(raw / 'frozen_manifest.json'), source_sha256=sha(__file__),
        scope='All new validation and all actually scored test checkpoints for fixed seed201 SGD/ER; no training replay',
        rows=rows, logit_elements=sum(r['logit_elements'] for r in rows),
        elapsed_seconds=time.monotonic()-started,
        environment=dict(torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                         threads=2, tf32=False, deterministic=True, batch_eval=512))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'environment']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    replay(a.output, a.root)
