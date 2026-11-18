"""Train one fixed task-7 condition and seed from a supplied pre-task checkpoint."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import task7_math as experiment

core = experiment.core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pre_checkpoint', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--seed', type=int, choices=range(3101, 3106), required=True)
    parser.add_argument('--condition', choices=experiment.CONDITIONS, required=True)
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args()
    saved = torch.load(args.pre_checkpoint, map_location='cpu', weights_only=True)
    assert saved['seed'] == args.seed and saved['task'] == 7 and saved['new_rows_reset']
    assert core.state_digest(saved['model']) == saved['state_sha256']
    assert saved['model']['head.weight'].shape[0] == 100
    if args.check_inputs_only:
        print(json.dumps({'status': 'passed', 'seed': args.seed, 'condition': args.condition,
            'pre_state_sha256': saved['state_sha256'], 'task': 7, 'old_classes': 70,
            'seen_classes': 80, 'training_performed': False}))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    core.write_json(output / 'environment.json', core.environment())
    dataset = core.data.load_train(args.data_root, download=False)
    assert dataset.provenance['archive_sha256'] == core.ARCHIVE_SHA256
    images = torch.from_numpy(dataset.images.copy()).to('cuda')
    _, inverse = core.data.class_mapping(args.seed)
    labels = torch.as_tensor(inverse[dataset.labels], dtype=torch.long, device='cuda')
    model = core.make_model(args.seed+10000, 'cuda').to(memory_format=torch.channels_last)
    model.load_state_dict(saved['model'])
    record = experiment.train_one(model, dataset, images, labels, args.seed,
                                   args.condition, 70, output / 'run', began+3300)
    assert record['start_state_sha256'] == saved['state_sha256']
    assert record['optimizer_steps'] == 5530
    assert record['current_presentations'] == record['replay_presentations'] == 350000
    record['checkpoint'] = core.save_checkpoint(output / 'endpoint.pt', model,
        {'seed': args.seed, 'task': 7, 'epochs': 70, 'branch': args.condition})
    core.write_json(output / 'training.json', record)
    print(json.dumps({'status': 'complete', 'seed': args.seed, 'condition': args.condition,
                      'seconds': time.monotonic()-began}), flush=True)


if __name__ == '__main__':
    main()
