"""Reviewer-requested fixed-buffer and normalized-compensation controls.

Both controls branch from frozen R9 checkpoints. They do not replace or extend
the continuing histories. Output includes checkpoints, per-example logits,
input hashes, and seed-level summaries.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)


def configure():
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen-checkpoint controls")
    return torch.device("cuda")


def config(seed: int) -> dict:
    path = ROOT / "configs" / "r9_followup" / f"split_cifar10_classil_er_s{seed}.json"
    cfg = read(path)
    expected = {"method": "er", "epochs": 4, "batch_size": 64, "lr": 0.01,
                "memory_size": 500, "replay_batch": 32, "tasks": 5}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected source config for seed {seed}")
    return cfg


def source_run(seed: int, method: str) -> Path:
    return ROOT / "results" / "r9_followup" / f"split_cifar10_classil_{method}_s{seed}"


def restore_memory(path: Path):
    import torch
    from experiments.r9_budget_extension.core import restore_memory as restore
    return restore(torch.load(path, map_location="cpu", weights_only=False))


def normalized_compensation(seed: int, cfg: dict, tasks: list, device, output: Path) -> dict:
    import torch
    from torch.nn import functional as F
    from experiments.r9_followup import engine

    run = source_run(seed, "er")
    sgd_result = read(source_run(seed, "sgd") / "result.json")
    er_result = read(run / "result.json")
    task_rows = []
    for task_id in range(1, 5):
        state_path = run / f"pre_task_{task_id:03d}.pt"
        memory_path = run / f"pre_task_{task_id:03d}_memory.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model = engine.local_model(cfg, cfg["init_seed"], device, state)
        memory = restore_memory(memory_path)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.0, weight_decay=0.0)
        task = tasks[task_id]
        x, y = task["train"]
        orders = []
        coefficient_sums = []
        counts = {"optimizer_steps": 0, "current_examples": 0, "replay_examples": 0}
        for epoch in range(4):
            generator = torch.Generator().manual_seed(cfg["order_seed"] * 100000 + task_id * 100 + epoch)
            indices = torch.randperm(len(y), generator=generator)
            orders.append(engine.digest_tensor(indices))
            for start in range(0, len(y), cfg["batch_size"]):
                ix = indices[start:start + cfg["batch_size"]]
                bx, by = x[ix].to(device), y[ix].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx)
                arrival_logits = logits.detach().clone()
                current_loss = F.cross_entropy(logits, by)
                rx, ry, rt, _ = memory.sample(cfg["replay_batch"], device)
                replay_loss = F.cross_entropy(model(rx), ry)
                wc = len(by) / (len(by) + len(ry))
                coefficient_sum = 2.0 - wc
                loss = (current_loss + (1.0 - wc) * replay_loss) / coefficient_sum
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite normalized-compensation loss")
                loss.backward()
                optimizer.step()
                memory.add_batch(bx, by, task_id, arrival_logits)
                coefficient_sums.append(coefficient_sum)
                counts["optimizer_steps"] += 1
                counts["current_examples"] += len(by)
                counts["replay_examples"] += len(ry)
        expected_orders = er_result["tasks"][task_id]["actual"]["batch_order_sha256"]
        if orders != expected_orders:
            raise ValueError("Current minibatch order drift")
        model.eval()
        test_x, test_y = task["test"]
        logits_parts = []
        with torch.no_grad():
            for start in range(0, len(test_y), 512):
                logits_parts.append(model(test_x[start:start + 512].to(device)).cpu())
        logits = torch.cat(logits_parts)
        prediction = logits.argmax(1)
        correct = int((prediction == test_y).sum())
        task_dir = output / "normalized_compensation" / f"s{seed}_t{task_id}"
        task_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = task_dir / "endpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        npz = task_dir / "endpoint.npz"
        np.savez_compressed(npz, logits=logits.numpy(), labels=test_y.numpy(),
                            prediction=prediction.numpy())
        unnormalized = er_result["tasks"][task_id]["compensated_er"]["trace"][-1]["evaluation"]["accuracy_pct"]
        sgd_actual = sgd_result["tasks"][task_id]["actual"]["trace"][-1]["evaluation"]["accuracy_pct"]
        task_rows.append({
            "task": task_id,
            "accuracy_pct": 100.0 * correct / len(test_y),
            "correct": correct,
            "n": len(test_y),
            "sgd_actual_accuracy_pct": sgd_actual,
            "unnormalized_compensated_accuracy_pct": unnormalized,
            "gap_vs_sgd_pp": sgd_actual - 100.0 * correct / len(test_y),
            "unnormalized_gap_vs_sgd_pp": sgd_actual - unnormalized,
            "coefficient_sum_range_before_normalization": [min(coefficient_sums), max(coefficient_sums)],
            "counts": counts,
            "checkpoint_sha256": sha(checkpoint),
            "npz_sha256": sha(npz),
            "state_sha256": engine.state_hash(model),
            "source_sha256": {state_path.relative_to(ROOT).as_posix(): sha(state_path),
                              memory_path.relative_to(ROOT).as_posix(): sha(memory_path)},
        })
    return {
        "seed": seed,
        "tasks": task_rows,
        "seed_mean_gap_vs_sgd_pp": statistics.mean(r["gap_vs_sgd_pp"] for r in task_rows),
        "seed_mean_unnormalized_gap_vs_sgd_pp": statistics.mean(
            r["unnormalized_gap_vs_sgd_pp"] for r in task_rows),
    }


class FrozenReservoir:
    def __init__(self, base):
        self._base = base
        self.initial_seen = base.seen
        self.initial_rows = copy.deepcopy(base.rows)

    @property
    def rows(self):
        return self._base.rows

    @property
    def draw_hashes(self):
        return self._base.draw_hashes

    def sample(self, n, device):
        return self._base.sample(n, device)

    def add_batch(self, *_args, **_kwargs):
        return None

    def state_dict(self):
        return self._base.state_dict()


def fixed_buffer(seed: int, cfg: dict, tasks: list, device, output: Path) -> dict:
    import torch
    from experiments.r9_followup import engine
    from experiments.r9_budget_extension import core

    run = source_run(seed, "er")
    state_path = run / "post_task_004.pt"
    memory_path = run / "post_task_004_memory.pt"
    model = engine.local_model(cfg, cfg["init_seed"], device,
                               torch.load(state_path, map_location="cpu", weights_only=True))
    memory = FrozenReservoir(restore_memory(memory_path))
    endpoint_rows = []

    def callback(record, active_model, _memory):
        if record["epoch"] not in (8, 16):
            return
        arrays, scores = core.inference(active_model, tasks, device)
        endpoint_dir = output / "fixed_buffer" / f"s{seed}_e{record['epoch']}"
        endpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = endpoint_dir / "endpoint.pt"
        torch.save(active_model.state_dict(), checkpoint)
        npz = endpoint_dir / "endpoint.npz"
        np.savez_compressed(npz, **arrays)
        endpoint_rows.append({"epoch": record["epoch"], "scores": scores,
                              "checkpoint_sha256": sha(checkpoint), "npz_sha256": sha(npz),
                              "state_sha256": engine.state_hash(active_model)})

    records = core.train_epochs(model, tasks[4], cfg, device, "er", memory,
                                start_epoch=4, end_epoch=16, callback=callback)
    if len(endpoint_rows) != 2:
        raise ValueError("Missing fixed-buffer endpoints")
    if memory._base.seen != memory.initial_seen or any(
            not torch.equal(a[0], b[0]) or int(a[1]) != int(b[1]) or int(a[2]) != int(b[2])
            for a, b in zip(memory._base.rows, memory.initial_rows)):
        raise ValueError("Frozen reservoir contents changed")
    evolving_path = ROOT / "results" / "r9_budget_extension_v1" / "trajectories" / f"s{seed}_er_actual" / "trajectory.json"
    evolving = read(evolving_path)
    evolving_endpoints = {int(r["epoch"]): r["scores"] for r in evolving["endpoints"]}
    return {
        "seed": seed,
        "source_sha256": {state_path.relative_to(ROOT).as_posix(): sha(state_path),
                          memory_path.relative_to(ROOT).as_posix(): sha(memory_path),
                          evolving_path.relative_to(ROOT).as_posix(): sha(evolving_path)},
        "buffer_rows": len(memory.rows),
        "buffer_seen_constant": memory._base.seen,
        "training_records": records,
        "fixed_endpoints": endpoint_rows,
        "evolving_endpoints": {str(k): evolving_endpoints[k] for k in (8, 16)},
    }


def summarize(rows: list[dict]) -> dict:
    normalized = [r["seed_mean_gap_vs_sgd_pp"] for r in rows]
    unnormalized = [r["seed_mean_unnormalized_gap_vs_sgd_pp"] for r in rows]
    return {
        "normalized_gap_vs_sgd_pp": {"mean": statistics.mean(normalized), "sd": statistics.stdev(normalized), "n": len(normalized)},
        "unnormalized_gap_vs_sgd_pp": {"mean": statistics.mean(unnormalized), "sd": statistics.stdev(unnormalized), "n": len(unnormalized)},
    }


def summarize_fixed(rows: list[dict]) -> dict:
    out = {}
    for epoch in (8, 16):
        for mode in ("fixed", "evolving"):
            values = {key: [] for key in ("current", "old", "all", "current2")}
            for row in rows:
                if mode == "fixed":
                    score = next(x["scores"] for x in row["fixed_endpoints"] if x["epoch"] == epoch)
                else:
                    score = row["evolving_endpoints"][str(epoch)]
                values["current"].append(score["all10"]["current"]["accuracy_pct"])
                values["old"].append(score["all10"]["old"]["accuracy_pct"])
                values["all"].append(score["all10"]["all"]["accuracy_pct"])
                values["current2"].append(score["current2"]["accuracy_pct"])
            out[f"epoch{epoch}_{mode}"] = {
                k: {"mean": statistics.mean(v), "sd": statistics.stdev(v), "n": len(v)}
                for k, v in values.items()
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/r17_review_controls_v1")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = configure()
    write_new(output / "started.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
        "python": platform.python_version(), "device": str(device),
    })
    normalized_rows, fixed_rows = [], []
    from experiments.r9_followup import engine
    for seed in range(201, 211):
        cfg = config(seed)
        tasks, _ = engine.make_stream(cfg, ROOT / "data")
        normalized_rows.append(normalized_compensation(seed, cfg, tasks, device, output))
        fixed_rows.append(fixed_buffer(seed, cfg, tasks, device, output))
        write_new(output / f"seed_{seed}.json", {"normalized_compensation": normalized_rows[-1],
                                                  "fixed_buffer": fixed_rows[-1]})
        print(json.dumps({"completed_seed": seed, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    report = {
        "schema": "official_review_controls_v1",
        "status": "complete",
        "review_triggered_posthoc_descriptive": True,
        "normalized_compensation": {"units": normalized_rows, "summary": summarize(normalized_rows)},
        "fixed_buffer": {"units": fixed_rows, "summary": summarize_fixed(fixed_rows)},
        "elapsed_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_new(output / "summary.json", report)
    print(json.dumps({"status": "complete", "summary": report["normalized_compensation"]["summary"],
                      "fixed": report["fixed_buffer"]["summary"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
