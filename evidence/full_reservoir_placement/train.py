"""Compare equal-weight supervision placements through a full single-pass stream."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parents[1]
sys.path.insert(0, str(ARTIFACT / "experiments/paa/asymmetric_loss/source"))
from r9_followup import engine

POLICIES = ("replay", "incoming", "omit")
SEEDS = range(4101, 4111)


def log(**record):
    print(json.dumps({"time": time.time(), **record}), flush=True)


def train_task(model, memory, task, cfg, seen_classes, policy, device):
    model.train()
    task_id = task["id"]
    images, labels = task["train"]
    seen = torch.zeros(10, dtype=torch.bool, device=device)
    current = torch.zeros_like(seen)
    seen[seen_classes] = True
    current[list(task["classes"])] = True
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.))
    rng = torch.Generator().manual_seed(cfg["order_seed"]*100000 + task_id*100)
    order = torch.randperm(len(labels), generator=rng)
    exposure = [0, 0]
    for ids in order.split(cfg["batch_size"]):
        x, y = images[ids].to(device), labels[ids].to(device)
        logits = model(x)
        total = F.cross_entropy(logits.masked_fill(~current, -1e9), y, reduction="sum")
        replay = memory.sample(cfg["replay_batch"], device)
        n = len(y)
        if replay is not None:
            rx, ry, rt, _ = replay
            now = rt == task_id
            count = int(now.sum())
            allowed = seen if policy == "replay" else seen & (~now[:, None] | current)
            replay_logits = model(rx)
            total += F.cross_entropy(replay_logits.masked_fill(~allowed, -1e9), ry, reduction="sum")
            if policy == "incoming" and count:
                group_loss = logits[:, seen].logsumexp(1)-logits[:, current].logsumexp(1)
                total += count*group_loss.mean()
            n += len(ry)
            exposure[0] += len(ry)
            exposure[1] += count
        opt.zero_grad(set_to_none=True)
        (total/n).backward()
        opt.step()
        memory.add_batch(x, y, task_id, logits.detach())
    content = hashlib.sha256()
    for x,y,t,_ in memory.rows:
        content.update(x.numpy().tobytes())
        content.update(bytes((int(y),t)))
    return dict(task=task_id+1, scheduled=len(labels), replay=exposure[0], replay_current=exposure[1],
                current_order=engine.digest_tensor(order), memory_content=content.hexdigest(),
                sampler_state=memory.sample_rng.bit_generator.state)


def evaluate(model, tasks, last, device, output):
    model.eval()
    current = np.sort(tasks[last]["classes"])
    seen = np.sort([label for task in tasks[:last+1] for label in task["classes"]])
    prior = np.setdiff1d(seen, current)
    logits, labels = [], []
    with torch.inference_mode():
        for task in tasks[:last+1]:
            x, y = task["test"]
            logits.extend(model(batch.to(device)).cpu().numpy() for batch in x.split(512))
            labels.extend(y.tolist())
    z, y = np.concatenate(logits), np.asarray(labels)
    prediction = seen[z[:, seen].argmax(1)]
    new_prediction = current[z[:, current].argmax(1)]
    data = dict(labels=y, predicted=prediction, new_prediction=new_prediction, new_classes=current)
    if len(prior):
        data.update(old_prediction=prior[z[:,prior].argmax(1)],
                    old_score=z[:,prior].max(1), new_score=z[:,current].max(1))
    np.savez_compressed(output, **data)


def config_for(base, seed):
    return {**base, "seed": seed, "init_seed": 10000+seed, "order_seed": 30000+seed,
            "class_order_seed": 40000+seed, "epochs": 1}


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
    args.output.mkdir(parents=True)
    scores = args.output/"reader_data"
    scores.mkdir()
    models = args.output/"final_models"
    models.mkdir()
    log(status="started", pid=os.getpid())

    # Use a representative full pass with a populated reservoir for resource estimation.
    trial_config = config_for(base, 4101)
    trial_tasks, _ = engine.make_stream(trial_config, str(args.data_root))
    for policy in POLICIES:
        trial = engine.local_model(trial_config, trial_config["init_seed"], device)
        memory = engine.Reservoir(500, 4101+77001)
        old_x, old_y = trial_tasks[3]["train"]
        memory.add_batch(old_x[:500], old_y[:500], 3, torch.zeros(500,10))
        began = time.monotonic()
        train_task(trial, memory, trial_tasks[4], trial_config, list(range(10)), policy, device)
        seconds = time.monotonic()-began
        cap = 1.5*seconds*50 + 120
        log(policy=policy, trial_seconds=seconds, estimated_cap_seconds=cap)
        if cap >= 3300:
            raise TimeoutError("The estimated ten-history batch exceeds 55 minutes.")
    del trial_tasks

    records = []
    observed = {}
    for policy in POLICIES:
        began = time.monotonic()
        for seed in SEEDS:
            if time.monotonic()-began > 3300:
                raise TimeoutError("The policy's ten-history batch exceeds 55 minutes.")
            cfg = config_for(base, seed)
            tasks, _ = engine.make_stream(cfg, str(args.data_root))
            model = engine.local_model(cfg, cfg["init_seed"], device)
            memory = engine.Reservoir(500, seed+77001)
            seen = []
            for i,task in enumerate(tasks):
                seen.extend(task["classes"])
                record = train_task(model, memory, task, cfg, sorted(seen), policy, device)
                if policy == POLICIES[0]:
                    observed[seed,i] = record
                else:
                    assert record == observed[seed,i], (seed, i, policy)
                records.append(dict(seed=seed, policy=policy, **record))
                evaluate(model, tasks, i, device, scores/f"s{seed}_{policy}_t{i+1}.npz")
            torch.save(model.state_dict(), models/f"s{seed}_{policy}.pt")
            engine.atomic_json(args.output/"training.json", records)
            log(policy=policy, seed=seed, status="trained")
        log(policy=policy, status="batch_complete", seconds=time.monotonic()-began)
    log(status="complete", histories=30)


if __name__ == "__main__":
    main()
