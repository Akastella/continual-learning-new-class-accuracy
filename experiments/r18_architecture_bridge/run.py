"""Run one paired CIFAR-10 ResNet-32 bridge seed with raw endpoint evidence."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

from .data import make_tasks, tensor_sha256
from .model import make_model, file_sha256, UPSTREAM_COMMIT, UPSTREAM_PATH

ROOT = Path(__file__).resolve().parents[2]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


class Reservoir:
    """Uniform reservoir over the multi-epoch presentation stream."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.storage_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.sample_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.seen = 0
        self.rows: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor, task_id: int, source_ids: torch.Tensor) -> None:
        for index in range(len(labels)):
            self.seen += 1
            slot = len(self.rows) if len(self.rows) < self.capacity else int(self.storage_rng.integers(self.seen))
            if slot < self.capacity:
                row = (
                    images[index].detach().cpu().clone(),
                    labels[index].detach().cpu().clone(),
                    task_id,
                    int(source_ids[index]),
                )
                if slot == len(self.rows):
                    self.rows.append(row)
                else:
                    self.rows[slot] = row

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.rows:
            return None
        ids = self.sample_rng.choice(len(self.rows), size=min(n, len(self.rows)), replace=False)
        rows = [self.rows[int(index)] for index in ids]
        images = torch.stack([row[0] for row in rows]).to(device, memory_format=torch.channels_last)
        labels = torch.stack([row[1] for row in rows]).to(device)
        return images, labels

    def state_dict(self) -> dict[str, object]:
        source_ids = [row[3] for row in self.rows]
        return {
            "capacity": self.capacity,
            "seen": self.seen,
            "rows": self.rows,
            "unique_source_examples": len(set(source_ids)),
            "duplicate_slots": len(source_ids) - len(set(source_ids)),
            "storage_rng": self.storage_rng.bit_generator.state,
            "sample_rng": self.sample_rng.bit_generator.state,
        }


@torch.no_grad()
def infer(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, object]:
    model.eval()
    logits_blocks: list[np.ndarray] = []
    for start in range(0, len(labels), 512):
        batch = images[start : start + 512].to(device, memory_format=torch.channels_last)
        logits_blocks.append(model(batch).cpu().numpy().astype(np.float32, copy=False))
    logits = np.concatenate(logits_blocks)
    labels_np = labels.numpy()
    predictions = logits.argmax(axis=1)
    correct = int(np.equal(predictions, labels_np).sum())
    losses = F.cross_entropy(torch.from_numpy(logits), labels, reduction="mean")
    return {
        "logits": logits,
        "n": len(labels),
        "correct": correct,
        "accuracy_pct": 100.0 * correct / len(labels),
        "cross_entropy": float(losses),
        "predictions_sha256": hashlib.sha256(predictions.astype("<i8").tobytes()).hexdigest(),
    }


def train_branch(
    model: torch.nn.Module,
    task: dict[str, object],
    config: dict[str, object],
    device: torch.device,
    deadline: float,
    method: str = "sgd",
    memory: Reservoir | None = None,
) -> dict[str, object]:
    images, labels, source_ids = task["train"]
    test_images, test_labels, _ = task["test"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    trace = [{"epoch": 0, **{k: v for k, v in infer(model, test_images, test_labels, device).items() if k != "logits"}}]
    counts = {"optimizer_steps": 0, "current_examples": 0, "replay_examples": 0, "forward_examples": 0}
    order_hashes: list[str] = []
    started = time.monotonic()
    for epoch in range(int(config["epochs"])):
        if time.monotonic() > deadline:
            raise TimeoutError("Per-seed deadline exceeded")
        model.train()
        generator = torch.Generator().manual_seed(
            int(config["order_seed"]) * 100000 + int(task["id"]) * 100 + epoch
        )
        order = torch.randperm(len(labels), generator=generator)
        order_hashes.append(tensor_sha256(order))
        for start in range(0, len(labels), int(config["batch_size"])):
            ids = order[start : start + int(config["batch_size"])]
            batch_images = images[ids].to(device, memory_format=torch.channels_last)
            batch_labels = labels[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            current_loss = F.cross_entropy(logits, batch_labels)
            loss = current_loss
            counts["forward_examples"] += len(batch_labels)
            if memory is not None:
                replay = memory.sample(int(config["replay_batch"]), device)
                if replay is not None:
                    replay_images, replay_labels = replay
                    replay_logits = model(replay_images)
                    replay_loss = F.cross_entropy(replay_logits, replay_labels)
                    loss = (
                        len(batch_labels) * current_loss + len(replay_labels) * replay_loss
                    ) / (len(batch_labels) + len(replay_labels))
                    counts["replay_examples"] += len(replay_labels)
                    counts["forward_examples"] += len(replay_labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            counts["optimizer_steps"] += 1
            counts["current_examples"] += len(batch_labels)
            if memory is not None:
                memory.add_batch(batch_images, batch_labels, int(task["id"]), source_ids[ids])
        endpoint = infer(model, test_images, test_labels, device)
        trace.append({"epoch": epoch + 1, **{k: v for k, v in endpoint.items() if k != "logits"}})
    torch.cuda.synchronize()
    return {
        "method": method,
        "trace": trace,
        "counts": counts,
        "batch_order_sha256": order_hashes,
        "wall_seconds": time.monotonic() - started,
        "final_state_sha256": state_sha256(model),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, role: str, task: int) -> dict[str, object]:
    torch.save(
        {"state_dict": model.state_dict(), "state_sha256": state_sha256(model), "role": role, "task": task},
        path,
    )
    return {"path": path.name, "file_sha256": file_sha256(path), "state_sha256": state_sha256(model)}


def save_current_logits(
    path: Path,
    task: dict[str, object],
    branches: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, object]:
    images, labels, official_indices = task["test"]
    arrays: dict[str, np.ndarray] = {
        "mapped_labels": labels.numpy().astype(np.int64),
        "official_indices": official_indices.numpy().astype(np.int64),
        "task": np.asarray(int(task["id"]), dtype=np.int64),
    }
    metrics: dict[str, object] = {}
    for name, model in branches.items():
        result = infer(model, images, labels, device)
        arrays[name + "_logits"] = result.pop("logits")
        metrics[name] = result
    np.savez(path, **arrays)
    return {"path": path.name, "file_sha256": file_sha256(path), "metrics": metrics}


def save_actual_full_test_logits(
    path: Path,
    all_test: tuple[torch.Tensor, torch.Tensor],
    class_inverse: np.ndarray,
    sgd: torch.nn.Module,
    er: torch.nn.Module,
    device: torch.device,
) -> dict[str, object]:
    images, original_labels = all_test
    mapped_labels = torch.from_numpy(class_inverse).long()[original_labels]
    sgd_result = infer(sgd, images, mapped_labels, device)
    er_result = infer(er, images, mapped_labels, device)
    np.savez(
        path,
        mapped_labels=mapped_labels.numpy().astype(np.int64),
        original_labels=original_labels.numpy().astype(np.int64),
        official_indices=np.arange(len(original_labels), dtype=np.int64),
        sgd_actual_logits=sgd_result.pop("logits"),
        er_actual_logits=er_result.pop("logits"),
    )
    return {
        "path": path.name,
        "file_sha256": file_sha256(path),
        "metrics": {"sgd_actual": sgd_result, "er_actual": er_result},
    }


def build_artifact_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.json":
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(output / "artifact_sha256.json", {"schema": 1, "files": files})


def execute(seed: int, config_path: Path, data_root: Path, output: Path) -> dict[str, object]:
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if seed not in base_config["seeds"]:
        raise ValueError("Seed is not in the frozen R18 matrix")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    config = {
        **base_config,
        "seed": seed,
        "init_seed": seed + int(base_config["initialization_seed_offset"]),
        "order_seed": seed + int(base_config["minibatch_order_seed_offset"]),
        "class_order_seed": seed + int(base_config["class_order_seed_offset"]),
        "reservoir_seed": seed + int(base_config["reservoir_seed_offset"]),
    }
    source_dir = Path(__file__).parent
    source_hashes = {path.name: file_sha256(path) for path in source_dir.glob("*.py")}
    meta = {
        "status": "running",
        "started_utc": utc(),
        "config": config,
        "config_file_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "upstream": {"commit": UPSTREAM_COMMIT, "path": UPSTREAM_PATH.relative_to(ROOT).as_posix(), "sha256": file_sha256(UPSTREAM_PATH)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": config["device"],
            "device_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "pid": os.getpid(),
    }
    write_json(output / "run_started.json", meta)
    (output / "source").mkdir()
    for path in source_dir.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    shutil.copy2(config_path, output / "source" / "config.json")

    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(str(config["device"]))
    deadline = time.monotonic() + float(config["deadline_seconds_per_seed"])
    tasks, provenance = make_tasks(data_root, int(config["class_order_seed"]))
    write_json(
        output / "data_manifest.json",
        {
            "provenance": provenance,
            "tasks": [{key: value for key, value in task.items() if key not in ("train", "test")} for task in tasks],
        },
    )
    from .data import load_official

    official, _ = load_official(data_root)
    all_test = official["test"]
    class_inverse = np.asarray(provenance["original_class_to_global_output"], dtype=np.int64)

    initial_model = make_model(int(config["init_seed"]), device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    save_checkpoint(output / "initial.pt", initial_model, "initial", -1)
    sgd_model = make_model(int(config["init_seed"]), device)
    er_model = make_model(int(config["init_seed"]), device)
    if state_sha256(sgd_model) != state_sha256(er_model) or state_sha256(sgd_model) != state_sha256(initial_model):
        raise AssertionError("Paired initialization mismatch")
    del initial_model
    reservoir = Reservoir(int(config["memory_size"]), int(config["reservoir_seed"]))
    records: list[dict[str, object]] = []
    started = time.monotonic()
    for task_id, task in enumerate(tasks):
        pre_sgd = copy.deepcopy(sgd_model.state_dict())
        pre_er = copy.deepcopy(er_model.state_dict())
        checkpoints: dict[str, object] = {
            "sgd_pre": save_checkpoint(output / f"task_{task_id:02d}_sgd_pre.pt", sgd_model, "sgd_pre", task_id),
            "er_pre": save_checkpoint(output / f"task_{task_id:02d}_er_pre.pt", er_model, "er_pre", task_id),
        }
        torch.save(reservoir.state_dict(), output / f"task_{task_id:02d}_er_memory_pre.pt")

        fresh_model = make_model(int(config["init_seed"]), device)
        fresh_model.load_state_dict(initial_state)
        fresh_log = train_branch(fresh_model, task, config, device, deadline, "fresh")
        checkpoints["fresh"] = save_checkpoint(output / f"task_{task_id:02d}_fresh.pt", fresh_model, "fresh", task_id)

        sgd_model.load_state_dict(pre_sgd)
        sgd_log = train_branch(sgd_model, task, config, device, deadline, "sgd_actual_probe")
        checkpoints["sgd_actual_probe"] = save_checkpoint(output / f"task_{task_id:02d}_sgd_actual_probe.pt", sgd_model, "sgd_actual_probe", task_id)

        er_probe = make_model(int(config["init_seed"]), device)
        er_probe.load_state_dict(pre_er)
        er_probe_log = train_branch(er_probe, task, config, device, deadline, "er_probe")
        checkpoints["er_probe"] = save_checkpoint(output / f"task_{task_id:02d}_er_probe.pt", er_probe, "er_probe", task_id)

        er_model.load_state_dict(pre_er)
        er_actual_log = train_branch(er_model, task, config, device, deadline, "er_actual", reservoir)
        checkpoints["er_actual"] = save_checkpoint(output / f"task_{task_id:02d}_er_actual.pt", er_model, "er_actual", task_id)
        torch.save(reservoir.state_dict(), output / f"task_{task_id:02d}_er_memory_post.pt")

        if fresh_log["batch_order_sha256"] != sgd_log["batch_order_sha256"] or fresh_log["batch_order_sha256"] != er_probe_log["batch_order_sha256"] or fresh_log["batch_order_sha256"] != er_actual_log["batch_order_sha256"]:
            raise AssertionError("Current minibatch orders are not paired")
        if task_id == 0 and fresh_log["final_state_sha256"] != sgd_log["final_state_sha256"]:
            raise AssertionError("Task-zero fresh/SGD identity failed")
        if task_id == 0 and fresh_log["final_state_sha256"] != er_probe_log["final_state_sha256"]:
            raise AssertionError("Task-zero fresh/ER-probe identity failed")

        current_logits = save_current_logits(
            output / f"task_{task_id:02d}_current_logits.npz",
            task,
            {"fresh": fresh_model, "sgd_actual_probe": sgd_model, "er_probe": er_probe, "er_actual": er_model},
            device,
        )
        full_logits = save_actual_full_test_logits(
            output / f"task_{task_id:02d}_actual_full_test_logits.npz",
            all_test,
            class_inverse,
            sgd_model,
            er_model,
            device,
        )
        fresh_accuracy = float(current_logits["metrics"]["fresh"]["accuracy_pct"])
        sgd_accuracy = float(current_logits["metrics"]["sgd_actual_probe"]["accuracy_pct"])
        er_probe_accuracy = float(current_logits["metrics"]["er_probe"]["accuracy_pct"])
        er_actual_accuracy = float(current_logits["metrics"]["er_actual"]["accuracy_pct"])
        record = {
            "task": task_id,
            "checkpoints": checkpoints,
            "fresh": fresh_log,
            "sgd_actual_probe": sgd_log,
            "er_probe": er_probe_log,
            "er_actual": er_actual_log,
            "current_logits": current_logits,
            "actual_full_test_logits": full_logits,
            "metrics": {
                "fresh_accuracy_pct": fresh_accuracy,
                "sgd_actual_probe_accuracy_pct": sgd_accuracy,
                "er_probe_accuracy_pct": er_probe_accuracy,
                "er_actual_accuracy_pct": er_actual_accuracy,
                "sgd_actual_gap_pp": fresh_accuracy - sgd_accuracy,
                "sgd_history_gap_pp": fresh_accuracy - sgd_accuracy,
                "er_actual_gap_pp": fresh_accuracy - er_actual_accuracy,
                "er_history_gap_pp": fresh_accuracy - er_probe_accuracy,
                "delta_A_er_minus_sgd_pp": sgd_accuracy - er_actual_accuracy,
                "delta_H_er_minus_sgd_pp": sgd_accuracy - er_probe_accuracy,
            },
            "reservoir": {
                "seen": reservoir.seen,
                "slots": len(reservoir.rows),
                "unique_source_examples": reservoir.state_dict()["unique_source_examples"],
                "duplicate_slots": reservoir.state_dict()["duplicate_slots"],
            },
        }
        records.append(record)
        write_json(output / f"task_{task_id:02d}.json", record)
        print(json.dumps({"seed": seed, "task": task_id, **record["metrics"]}), flush=True)
        del fresh_model, er_probe

    result = {
        **meta,
        "status": "complete",
        "finished_utc": utc(),
        "wall_seconds": time.monotonic() - started,
        "tasks": records,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(output / "result.json", result)
    build_artifact_manifest(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default="experiments/r18_architecture_bridge/config.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        execute(args.seed, Path(args.config), Path(args.data_root), output)
    except BaseException as error:
        if output.exists() and not (output / "failure.json").exists():
            write_json(output / "failure.json", {"utc": utc(), "error": type(error).__name__ + ": " + str(error)})
        raise


if __name__ == "__main__":
    main()

