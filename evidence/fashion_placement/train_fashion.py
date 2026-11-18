"""Compare supervision placement in single-pass Fashion-MNIST."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reservoir_supervision_timing"))
from train_timing import ARTIFACT, config_for, engine, evaluate, log, task_batches
import torch
from torch.nn import functional as F

SEEDS = range(6101, 6111)
POLICIES = ("retrieval", "same_step", "expected_count", "omitted", "first_presentation")


def load_images(root, split):
    images = gzip.decompress((root / f"{split}-images-idx3-ubyte.gz").read_bytes())
    labels = gzip.decompress((root / f"{split}-labels-idx1-ubyte.gz").read_bytes())
    rows = torch.from_numpy(np.frombuffer(images, dtype=np.uint8, offset=16).copy()).reshape(-1,28,28)
    targets = torch.from_numpy(np.frombuffer(labels, dtype=np.uint8, offset=8).astype(np.int64))
    assert rows.shape[0] == len(targets) == (60000 if split == "train" else 10000)
    return ((rows.float().div_(255).sub_(0.2860).div_(0.3530)).unsqueeze(1), targets)


def make_tasks(train, test, seed):
    labels = np.random.default_rng(seed).permutation(10).tolist()
    stream = []
    for begin in range(0, 10, 2):
        current = tuple(labels[begin:begin+2])
        pairs = []
        for images, targets in (train, test):
            keep = (targets == current[0]) | (targets == current[1])
            pairs.append((images[keep], targets[keep]))
        stream.append(dict(id=begin//2, classes=current, train=pairs[0], test=pairs[1]))
    return stream, labels


def scheduled_retrievals(memory, batches, task, n_replay):
    sampler = engine.Reservoir(memory.capacity, 0)
    sampler.seen = memory.seen
    sampler.storage_rng.bit_generator.state = memory.storage_rng.bit_generator.state
    sampler.sample_rng.bit_generator.state = memory.sample_rng.bit_generator.state
    dummy = torch.zeros(())
    sampler.rows = [(dummy, dummy, t, dummy) for _, _, t, _ in memory.rows]
    weights = np.zeros(sum(len(ids) for ids in batches), dtype=np.float64)
    for ids in batches:
        sampled = sampler.sample(n_replay, "cpu")
        now = sampled[1][sampled[2] == task].int().tolist()
        weights[now] += 1/len(sampled[1])
        sampler.add_batch(ids, ids, task, torch.zeros(len(ids)))
    return torch.from_numpy(weights)


def task_update(model, memory, tasks, task_id, cfg, policy, device, output):
    task = tasks[task_id]
    images, labels = task["train"]
    seen = torch.zeros(10, dtype=torch.bool, device=device)
    seen[[label for previous in tasks[:task_id+1] for label in previous["classes"]]] = True
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    batches = task_batches(task, cfg)
    weights = None
    if task_id and policy == "first_presentation":
        weights = scheduled_retrievals(memory, batches, task_id, cfg["replay_batch"])
    past_current = absent = 0
    dose = 0.0
    moments = {}
    counts = []
    for step, ids in enumerate(batches, 1):
        model.train()
        x, y = images[ids].to(device), labels[ids].to(device)
        logits = model(x)
        active = torch.zeros_like(seen)
        active[y] = True
        absent += int(active.sum()) != len(task["classes"])
        loss = F.cross_entropy(logits.masked_fill(~active, -1e9), y)
        sampled = memory.sample(cfg["replay_batch"], device)
        if sampled is not None:
            rx, ry, rt, indices = sampled
            current = rt == task_id
            counts.append(indices[current].tolist())
            rlogits = model(rx)
            if policy == "retrieval":
                allowed = seen
                weight = int(current.sum())/len(ry)
            else:
                allowed = torch.where(current[:, None], active, seen).clone()
                allowed[torch.arange(len(ry), device=device), ry] = True
                weight = 0.0
            loss += F.cross_entropy(rlogits.masked_fill(~allowed, -1e9), ry)
            if task_id and policy in ("same_step", "expected_count", "first_presentation"):
                group_loss = logits[:, seen].logsumexp(1)-logits[:, active].logsumexp(1)
                if policy == "first_presentation":
                    coefficients = weights[ids].to(device=device, dtype=group_loss.dtype)
                    loss += (coefficients*group_loss).sum()
                    weight = float(weights[ids].sum())
                else:
                    weight = (int(current.sum())/len(ry) if policy == "same_step"
                              else past_current/memory.seen)
                    loss += weight*group_loss.mean()
            dose += weight if task_id else 0.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        memory.add_batch(x, y, task_id, ids)
        past_current += len(ids)
        if task_id and step in (16, len(batches)):
            evaluate(model, tasks, task_id, device, output / f"{step}.npz")
            moments[step] = dose
    return moments, counts, absent


def shared_history(args, train, test, base, device):
    records = []
    began = time.monotonic()
    for seed in SEEDS:
        cfg = config_for(base, seed)
        tasks, order = make_tasks(train, test, cfg["class_order_seed"])
        reference = engine.local_model(cfg, cfg["init_seed"], device)
        reservoir = engine.Reservoir(cfg["memory_size"], seed+77001)
        task_update(reference, reservoir, tasks, 0, cfg, "retrieval", device, args.output)
        for task in range(1,len(tasks)):
            if time.monotonic()-began > 2700:
                raise TimeoutError("The shared-history comparison exceeded 45 minutes.")
            draws = {}
            totals = {}
            for policy in ("first_presentation", "retrieval"):
                shared = policy == "retrieval"
                model = reference if shared else copy.deepcopy(reference)
                memory = reservoir if shared else copy.deepcopy(reservoir)
                directory = args.output / f"s{seed}_t{task+1}_{policy}"
                directory.mkdir()
                moments, sampled, absent = task_update(model, memory, tasks, task, cfg, policy, device, directory)
                draws[policy] = sampled
                totals[policy] = moments[max(moments)]
                records.append(dict(seed=seed, task=task+1, policy=policy, dose_by_update=moments,
                                    class_order=order, missing_current_class_batches=absent))
            assert draws["first_presentation"] == draws["retrieval"]
            assert abs(totals["first_presentation"]-totals["retrieval"]) < 1e-10
        engine.atomic_json(args.output / "training.json", records)
        log(seed=seed, status="trained")
    log(status="complete", seconds=time.monotonic()-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with the four original gzip files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-history", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    train, test = [load_images(args.data_root, split) for split in ("train", "t10k")]
    base = json.loads((ARTIFACT/"configs/r9_followup/split_cifar10_classil_er_s201.json").read_text())
    base["benchmark"] = "fashion_mnist"
    args.output.mkdir(parents=True)
    if args.shared_history:
        shared_history(args, train, test, base, device)
        return
    rows = []
    streams = {}
    for policy in POLICIES:
        began = time.monotonic()
        for seed in SEEDS:
            cfg = config_for(base, seed)
            tasks, order = make_tasks(train, test, cfg["class_order_seed"])
            model = engine.local_model(cfg, cfg["init_seed"], device)
            memory = engine.Reservoir(cfg["memory_size"], seed+77001)
            for task in range(len(tasks)):
                if time.monotonic()-began > 2700:
                    raise TimeoutError("The ten-history condition exceeded 45 minutes.")
                directory = args.output / f"s{seed}_t{task+1}_{policy}"
                if task:
                    directory.mkdir()
                started = time.monotonic()
                moments, draws, absent = task_update(model, memory, tasks, task, cfg, policy, device, directory)
                if seed == SEEDS.start and task == 1:
                    log(policy=policy, measured_seconds=time.monotonic()-started)
                if policy == "retrieval":
                    streams[seed,task] = draws
                else:
                    assert draws == streams[seed,task], (seed, task, policy)
                if task:
                    rows.append(dict(seed=seed, task=task+1, policy=policy, dose_by_update=moments,
                                     class_order=order, missing_current_class_batches=absent))
            log(policy=policy, seed=seed, status="trained")
        engine.atomic_json(args.output / "training.json", rows)
        log(policy=policy, status="condition_complete", seconds=time.monotonic()-began)
    log(status="complete", histories=len(SEEDS)*len(POLICIES))


if __name__ == "__main__":
    main()
