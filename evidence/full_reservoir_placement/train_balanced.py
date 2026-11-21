"""Rerun the CIFAR-10 placement comparison with class-balanced retrieval."""
import argparse
import json
from pathlib import Path
import time

from train import ARTIFACT, POLICIES, SEEDS, config_for, engine, evaluate, log, train_task
import numpy as np
import torch


class BalancedReservoir(engine.Reservoir):
    def sample(self, n, device):
        if not self.rows:
            return None
        n = min(n, len(self.rows))
        labels = torch.stack([row[1] for row in self.rows]).numpy()
        classes = self.sample_rng.permutation(np.unique(labels))
        quota, extra = divmod(n, len(classes))
        ids = []
        for i, label in enumerate(classes):
            members = np.flatnonzero(labels == label)
            take = min(quota + (i < extra), len(members))
            ids.extend(self.sample_rng.choice(members, size=take, replace=False))
        if len(ids) < n:
            other = np.setdiff1d(np.arange(len(self.rows)), ids, assume_unique=True)
            ids.extend(self.sample_rng.choice(other, n-len(ids), replace=False))
        selected = [self.rows[i] for i in self.sample_rng.permutation(ids)]
        return (torch.stack([row[0] for row in selected]).to(device),
                torch.stack([row[1] for row in selected]).to(device),
                torch.tensor([row[2] for row in selected], device=device),
                torch.stack([row[3] for row in selected]).to(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    base = json.loads((ARTIFACT/"configs/r9_followup/split_cifar10_classil_er_s201.json").read_text())
    scores = args.output/"reader_data"
    scores.mkdir(parents=True)
    records = []
    began = time.monotonic()
    for seed in SEEDS:
        cfg = config_for(base, seed)
        tasks, _ = engine.make_stream(cfg, str(args.data_root))
        exposures = {}
        for policy in POLICIES:
            model = engine.local_model(cfg, cfg["init_seed"], device)
            memory = BalancedReservoir(500, seed+77001)
            seen = []
            for index, task in enumerate(tasks):
                if time.monotonic()-began >= 1800:
                    raise TimeoutError("The training exceeded 30 minutes.")
                seen.extend(task["classes"])
                record = train_task(model, memory, task, cfg, sorted(seen), policy, device)
                if policy == POLICIES[0]:
                    exposures[index] = record
                else:
                    assert record == exposures[index], (seed, index, policy)
                records.append(dict(seed=seed, policy=policy, **record))
                evaluate(model, tasks, index, device, scores/f"s{seed}_{policy}_t{index+1}.npz")
        engine.atomic_json(args.output/"training.json", records)
        log(seed=seed, elapsed_seconds=time.monotonic()-began)
        if seed == SEEDS.start and 1.5*(time.monotonic()-began)*len(SEEDS) >= 1740:
            raise TimeoutError("The measured runtime exceeds the training budget.")
    log(status="complete", seconds=time.monotonic()-began)


if __name__ == "__main__":
    main()
