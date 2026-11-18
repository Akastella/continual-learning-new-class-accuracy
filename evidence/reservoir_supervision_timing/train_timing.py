"""Compare first-presentation and retrieval timing at the same total per-image weight."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_reservoir_placement"))
from train import ARTIFACT, config_for, engine, evaluate, log
import torch
from torch.nn import functional as F

SEEDS = range(5101, 5111)
POLICIES = ("retrieval", "first_presentation", "expected_count")
EARLY_STEPS = (4, 16, 40)


def task_batches(task, cfg):
    generator = torch.Generator().manual_seed(cfg["order_seed"]*100000 + task["id"]*100)
    return torch.randperm(len(task["train"][1]), generator=generator).split(cfg["batch_size"])


def planned_weights(task, cfg, memory, batches):
    # Only identities and sampling decisions are needed for this calculation.
    sampler = engine.Reservoir(memory.capacity, 0)
    sampler.seen = memory.seen
    sampler.storage_rng.bit_generator.state = memory.storage_rng.bit_generator.state
    sampler.sample_rng.bit_generator.state = memory.sample_rng.bit_generator.state
    unused = torch.zeros(())
    for _, _, old_task, _ in memory.rows:
        sampler.rows.append((unused, unused, old_task, unused))
    weights = np.zeros(len(task["train"][1]), dtype=np.float64)
    retrieval = []
    for ids in batches:
        sampled = sampler.sample(cfg["replay_batch"], "cpu")
        now = [] if sampled is None else sampled[1][sampled[2] == task["id"]].int().tolist()
        denominator = len(ids) + (0 if sampled is None else len(sampled[1]))
        weights[now] += 1/denominator
        retrieval.append(now)
        sampler.add_batch(ids, ids, task["id"], torch.zeros(len(ids)))
    return torch.from_numpy(weights), retrieval


def run_task(model, memory, tasks, task_id, cfg, policy, device, output, weights=None, indices=None):
    task = tasks[task_id]
    images, labels = task["train"]
    current = torch.zeros(10, dtype=torch.bool, device=device)
    seen = current.clone()
    current[list(task["classes"])] = True
    seen[[label for prior in tasks[:task_id+1] for label in prior["classes"]]] = True
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    batches = task_batches(task, cfg)
    n_current, dose = 0, 0.0
    measurements = {}
    for step, ids in enumerate(batches, 1):
        model.train()
        x, y = images[ids].to(device), labels[ids].to(device)
        logits = model(x)
        total = F.cross_entropy(logits.masked_fill(~current, -1e9), y, reduction="sum")
        replay = memory.sample(cfg["replay_batch"], device)
        weight = 0.0
        if replay is None:
            loss = total/len(ids)
        else:
            rx, ry, rt, replay_ids = replay
            now = rt == task_id
            if indices is not None:
                assert replay_ids[now].tolist() == indices[step-1]
            replay_logits = model(rx)
            allowed = seen if policy == "retrieval" else seen & (~now[:, None] | current)
            total += F.cross_entropy(replay_logits.masked_fill(~allowed, -1e9), ry, reduction="sum")
            size = len(ids) + len(ry)
            loss = total/size
            if policy == "retrieval":
                weight = int(now.sum())/size
            elif task_id:
                group = logits[:, seen].logsumexp(1)-logits[:, current].logsumexp(1)
                if policy == "first_presentation":
                    coefficients = weights[ids].to(device=device, dtype=group.dtype)
                    loss += (coefficients*group).sum()
                    weight = float(weights[ids].sum())
                elif policy == "expected_count":
                    weight = len(ry)*n_current/(memory.seen*size)
                    loss += weight*group.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # The sampler simply carries image indices. The stored field is unused by the loss.
        memory.add_batch(x, y, task_id, ids)
        n_current += len(ids)
        dose += weight if task_id else 0.
        if output is not None and (step in EARLY_STEPS or step == len(batches)):
            evaluate(model, tasks, task_id, device, output / f"{step}.npz")
            measurements[step] = dose
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carry-history", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    base = json.loads((ARTIFACT/"configs/r9_followup/split_cifar10_classil_er_s201.json").read_text())
    args.output.mkdir(parents=True)
    records = []
    began = time.monotonic()
    log(status="started")
    for seed in SEEDS:
        cfg = config_for(base, seed)
        tasks, stream = engine.make_stream(cfg, str(args.data_root))
        reference = engine.local_model(cfg, cfg["init_seed"], device)
        memory = engine.Reservoir(cfg["memory_size"], seed+77001)
        run_task(reference, memory, tasks, 0, cfg, "retrieval", device, None)
        continued = copy.deepcopy(reference) if args.carry_history else None
        policies = ("first_presentation", "retrieval") if args.carry_history else (
            "first_presentation", "expected_count", "retrieval")
        for task_id in range(1, len(tasks)):
            if time.monotonic()-began > 3000:
                raise TimeoutError("The ten-history comparison exceeded 50 minutes.")
            weights, ids = planned_weights(tasks[task_id], cfg, memory, task_batches(tasks[task_id], cfg))
            doses = {}
            for policy in policies:
                model = reference
                branch_memory = memory
                if policy != "retrieval":
                    model = continued if args.carry_history else copy.deepcopy(reference)
                    branch_memory = copy.deepcopy(memory)
                directory = args.output / f"s{seed}_t{task_id+1}_{policy}"
                directory.mkdir()
                started = time.monotonic()
                result = run_task(model, branch_memory, tasks, task_id, cfg, policy, device,
                                  directory, weights=weights, indices=ids)
                seconds = time.monotonic()-started
                if seed == SEEDS.start and task_id == 1:
                    estimate = 1.5*seconds*40
                    log(policy=policy, measured_seconds=seconds, estimated_policy_seconds=estimate)
                    if estimate >= 900:
                        raise TimeoutError("Estimated runtime exceeds the measured-task budget.")
                doses[policy] = result
                records.append(dict(seed=seed, task=task_id+1, policy=policy, dose_by_update=result,
                                    class_order=stream["class_order_original_ids"]))
            for step in doses["retrieval"]:
                assert doses["first_presentation"][step] >= doses["retrieval"][step]-1e-10
            end = max(doses["retrieval"])
            assert abs(doses["first_presentation"][end]-doses["retrieval"][end]) < 1e-10
        engine.atomic_json(args.output/"training.json", records)
        log(seed=seed, status="trained")
    log(status="complete", seconds=time.monotonic()-began)


if __name__ == "__main__":
    main()
