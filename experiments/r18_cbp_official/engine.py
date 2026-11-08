"""Run official-semantics CBP in the frozen primary CIFAR-10 protocol."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from experiments.r18_cbp_official.cbp import OfficialConvGnT
from experiments.r9_followup import engine as primary


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class Network(primary.Network):
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layers = self.backbone
        x1 = layers[2](layers[1](layers[0](x)))
        x2 = layers[5](layers[4](layers[3](x1)))
        x3 = layers[8](layers[7](layers[6](x2)))
        flat = layers[9](x3)
        x4 = layers[11](layers[10](flat))
        return self.head(x4), [x1, x2, flat, x4]

    def weight_layers(self) -> list[torch.nn.Module]:
        return [self.backbone[0], self.backbone[3], self.backbone[6], self.backbone[10], self.head]


def local_model(config: dict, device: torch.device, state: dict | None = None) -> Network:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["init_seed"])
        model = Network(config["benchmark"], config["width"])
    if state is not None:
        model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def infer(model: Network, task: dict, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    x, y = task["test"]
    pieces = []
    for start in range(0, len(y), 512):
        pieces.append(model(x[start:start + 512].to(device)).cpu())
    logits = torch.cat(pieces)
    predictions = logits.argmax(1)
    correct = int((predictions == y).sum())
    score = {
        "n": len(y),
        "correct": correct,
        "accuracy_pct": 100.0 * correct / len(y),
        "cross_entropy": float(F.cross_entropy(logits, y)),
        "prediction_sha256": primary.digest_tensor(predictions),
        "logits_sha256": primary.digest_tensor(logits),
    }
    return score, logits.numpy(), y.numpy()


def train_plain(model: Network, task: dict, config: dict, device: torch.device) -> dict:
    x, y = task["train"]
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.0, weight_decay=0.0)
    orders = []
    counts = {"optimizer_steps": 0, "current_examples": 0}
    for epoch in range(config["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["order_seed"] * 100000 + task["id"] * 100 + epoch)
        indices = torch.randperm(len(y), generator=generator)
        orders.append(primary.digest_tensor(indices))
        for start in range(0, len(y), config["batch_size"]):
            ix = indices[start:start + config["batch_size"]]
            bx, by = x[ix].to(device), y[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
            counts["optimizer_steps"] += 1
            counts["current_examples"] += len(by)
    return {"counts": counts, "batch_order_sha256": orders, "final_state_sha256": primary.state_hash(model)}


def train_cbp(
    model: Network,
    task: dict,
    config: dict,
    device: torch.device,
    cbp: OfficialConvGnT,
) -> dict:
    x, y = task["train"]
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.0, weight_decay=0.0)
    orders = []
    counts = {"optimizer_steps": 0, "current_examples": 0}
    before = list(cbp.replacements)
    for epoch in range(config["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["order_seed"] * 100000 + task["id"] * 100 + epoch)
        indices = torch.randperm(len(y), generator=generator)
        orders.append(primary.digest_tensor(indices))
        for start in range(0, len(y), config["batch_size"]):
            ix = indices[start:start + config["batch_size"]]
            bx, by = x[ix].to(device), y[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model.forward_with_features(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            cbp.step(features)
            counts["optimizer_steps"] += 1
            counts["current_examples"] += len(by)
    return {
        "counts": counts,
        "batch_order_sha256": orders,
        "final_state_sha256": primary.state_hash(model),
        "task_replacements_by_layer": [x - y for x, y in zip(cbp.replacements, before)],
        "cumulative_cbp": cbp.summary(),
    }


def validate(config: dict) -> None:
    expected = {
        "benchmark": "split_cifar10_classil", "method": "cbp", "tasks": 5,
        "epochs": 4, "batch_size": 64, "lr": 0.01, "width": 256,
        "momentum": 0.0, "weight_decay": 0.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Primary protocol drift: {key}={config.get(key)!r}, expected {value!r}")
    if config.get("profile") not in {"nature_cifar", "official_conv_default"}:
        raise ValueError("Unknown official profile")


def execute(config: dict, data_root: Path, output: Path) -> dict:
    validate(config)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(config.get("threads", 2))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for folder in (
        ROOT / "experiments/r18_cbp_official", ROOT / "third_party/r18_loss_of_plasticity_official"
    ) for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    started = time.perf_counter()
    atomic_json(output / "run_started.json", {
        "config": config,
        "source_sha256": sources,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        },
    })
    tasks, provenance = primary.make_stream(config, data_root)
    atomic_json(output / "data_manifest.json", {
        "provenance": provenance,
        "tasks": [{k: v for k, v in t.items() if k not in ("train", "test")} for t in tasks],
    })
    model = local_model(config, device)
    initial = copy.deepcopy(model.state_dict())
    torch.save(initial, output / "initial.pt")
    cbp = OfficialConvGnT(
        model.weight_layers(), replacement_rate=config["replacement_rate"],
        decay_rate=config["decay_rate"], maturity_threshold=config["maturity_threshold"],
        num_last_filter_outputs=16, rng_seed=config["cbp_rng_seed"], device=device,
    )
    records = []
    for task_id, task in enumerate(tasks):
        pre = copy.deepcopy(model.state_dict())
        torch.save(pre, output / f"pre_task_{task_id:03d}.pt")
        fresh = local_model(config, device, initial)
        probe = local_model(config, device, pre)
        fresh_log = train_plain(fresh, task, config, device)
        probe_log = train_plain(probe, task, config, device)
        actual_log = train_cbp(model, task, config, device, cbp)
        if not (fresh_log["batch_order_sha256"] == probe_log["batch_order_sha256"] == actual_log["batch_order_sha256"]):
            raise AssertionError("Current minibatch order mismatch")
        scores, arrays = {}, {}
        for name, branch in (("fresh", fresh), ("probe", probe), ("actual", model)):
            score, logits, labels = infer(branch, task, device)
            scores[name] = score
            arrays[f"{name}_current_logits"] = logits
            arrays[f"{name}_current_labels"] = labels
            torch.save(branch.state_dict(), output / f"{name}_task_{task_id:03d}.pt")
        post_scores = []
        for other in tasks:
            score, logits, labels = infer(model, other, device)
            post_scores.append(score)
            arrays[f"actual_task{other['id']}_logits"] = logits
            arrays[f"actual_task{other['id']}_labels"] = labels
        npz = output / f"task_{task_id:03d}_logits.npz"
        np.savez_compressed(npz, **arrays)
        record = {
            "task": task_id,
            "pre_state_sha256": primary.state_hash(local_model(config, device, pre)),
            "fresh": {**fresh_log, "evaluation": scores["fresh"]},
            "probe": {**probe_log, "evaluation": scores["probe"]},
            "actual": {**actual_log, "evaluation": scores["actual"]},
            "post_task_evaluation": post_scores,
            "actual_gap_pp": scores["fresh"]["accuracy_pct"] - scores["actual"]["accuracy_pct"],
            "history_gap_pp": scores["fresh"]["accuracy_pct"] - scores["probe"]["accuracy_pct"],
            "objective_cost_pp": scores["probe"]["accuracy_pct"] - scores["actual"]["accuracy_pct"],
            "logits_npz_sha256": sha(npz),
        }
        records.append(record)
        atomic_json(output / f"task_{task_id:03d}.json", record)
        print(json.dumps({
            "task": task_id, "profile": config["profile"],
            "actual": scores["actual"]["accuracy_pct"], "probe": scores["probe"]["accuracy_pct"],
            "replacements": actual_log["task_replacements_by_layer"],
        }), flush=True)
    result = {
        "schema": "r18_official_cbp_port_v1", "status": "complete", "config": config,
        "tasks": records, "cbp_final": cbp.summary(),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "result.json", result)
    artifacts = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    atomic_json(output / "artifact_manifest.json", {"sha256": artifacts})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    execute(config, ROOT / args.data_root, ROOT / args.output)


if __name__ == "__main__":
    main()
