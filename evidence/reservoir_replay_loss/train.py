"""Compare losses on naturally retrieved current images using the original ER engine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parents[1]
sys.path.insert(0, str(ARTIFACT / "experiments/paa/asymmetric_loss/source"))
from r9_followup import engine

SEEDS = range(201, 211)
CONDITIONS = ("all_class_replay", "current_two_replay")


class Reservoir(engine.Reservoir):
    def __init__(self, path, seed):
        state = torch.load(path, map_location="cpu", weights_only=True)
        super().__init__(state["capacity"], seed)
        self.rows, self.seen = state["rows"], state["seen"]
        self.storage_rng.bit_generator.state = state["storage_rng"]
        self.sample_rng.bit_generator.state = state["sample_rng"]
        self.replayed_current = 0
        self.draw_order = hashlib.sha256()

    def sample(self, n, device):
        draw = super().sample(n, device)
        _, labels, tasks, _ = draw
        self.replayed_current += int((tasks == 4).sum())
        self.draw_order.update(torch.stack((labels, tasks)).cpu().numpy().tobytes())
        return draw

    def memory_content(self):
        content = hashlib.sha256()
        for image, label, task, _ in self.rows:
            content.update(image.numpy().tobytes())
            content.update(bytes((int(label), task)))
        return content.hexdigest()


def load_inputs(seed, history, data_root):
    name = f"split_cifar10_classil_er_s{seed}"
    cfg = json.loads((ARTIFACT / "configs/r9_followup" / f"{name}.json").read_text())
    tasks, _ = engine.make_stream(cfg, str(data_root))
    return cfg, tasks, history / name


def train(cfg, task, history, condition, device):
    seed = cfg["seed"]
    state = torch.load(history / "pre_task_004.pt", map_location="cpu", weights_only=True)
    model = engine.local_model(cfg, cfg["init_seed"], device, state)
    memory = Reservoir(history / "pre_task_004_memory.pt", seed + 77001)
    is_current = torch.zeros(10, dtype=torch.bool, device=device)
    is_current[list(task["classes"])] = True
    calls = 0

    def training_loss(logits, labels):
        nonlocal calls
        incoming = calls % 2 == 0
        calls += 1
        if incoming or condition == "current_two_replay":
            excluded = is_current[labels, None] & ~is_current[None, :]
            logits = logits.masked_fill(excluded, -1e9)
        return F.cross_entropy(logits, labels)

    prior_f, prior_evaluation = engine.F, engine.evaluate
    engine.F = SimpleNamespace(cross_entropy=training_loss)
    engine.evaluate = lambda *args: {}
    try:
        record = engine.train_branch(model, task, cfg, device, "er", memory)
    finally:
        engine.F, engine.evaluate = prior_f, prior_evaluation
    assert calls == 2 * record["counts"]["optimizer_steps"]
    record.update(replayed_current=memory.replayed_current,
                  replay_label_order=memory.draw_order.hexdigest(),
                  final_buffer_images=memory.memory_content())
    return model, record


def evaluate(cfg, tasks, state, path, device, condition):
    model = engine.local_model(cfg, cfg["init_seed"], device, state).eval()
    labels, scores = [], []
    with torch.inference_mode():
        for task in tasks:
            images, truth = task["test"]
            for batch in images.split(512):
                scores.append(model(batch.to(device)).cpu().numpy())
            labels.extend(truth.tolist())
    labels, scores = np.asarray(labels), np.concatenate(scores)
    new = np.sort(tasks[-1]["classes"])
    old = np.setdiff1d(np.arange(10), new)
    old_index, new_index = scores[:, old].argmax(1), scores[:, new].argmax(1)
    row = np.arange(len(scores))
    np.savez_compressed(path, labels=labels, new_classes=new,
        predicted=scores.argmax(1), old_prediction=old[old_index], new_prediction=new[new_index],
        old_score=scores[row, old[old_index]], new_score=scores[row, new[new_index]],
        seed=cfg["seed"], condition=condition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, tasks, history = load_inputs(201, args.history, args.data_root)
    x, y = tasks[-1]["train"]
    trial = {**tasks[-1], "train": (x[:1280], y[:1280])}
    _, measured = train({**cfg, "epochs": 1}, trial, history, CONDITIONS[0], device)
    estimate = 1.5 * measured["seconds"] * 157 / 20 * 80 + 120
    print(json.dumps({"device": args.device, "timed_updates": 20,
        "seconds": measured["seconds"], "estimated_cap_seconds": estimate}), flush=True)
    if args.timing_only or estimate >= 3300:
        return

    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "runs"
    root.mkdir()
    deadline = time.monotonic() + 3300
    records = []
    for seed in SEEDS:
        cfg, tasks, history = load_inputs(seed, args.history, args.data_root)
        pair = []
        for condition in CONDITIONS:
            if time.monotonic() > deadline:
                raise TimeoutError("The training budget expired.")
            model, record = train(cfg, tasks[-1], history, condition, device)
            torch.save(model.state_dict(), root / f"s{seed}_{condition}.pt")
            pair.append(record)
            records.append({"seed": seed, "condition": condition, **record})
        for key in ("counts", "batch_order_sha256", "replayed_current",
                    "replay_label_order", "final_buffer_images"):
            assert pair[0][key] == pair[1][key], (seed, key)
        engine.atomic_json(args.output / "training.json", records)
        print(json.dumps({"seed": seed, "status": "trained",
                          "current_replay_presentations": pair[0]["replayed_current"]}), flush=True)
    out = args.output / "reader_data"
    out.mkdir()
    for seed in SEEDS:
        cfg, tasks, _ = load_inputs(seed, args.history, args.data_root)
        for condition in CONDITIONS:
            state = torch.load(root / f"s{seed}_{condition}.pt", map_location="cpu", weights_only=True)
            evaluate(cfg, tasks, state, out / f"s{seed}_{condition}.npz", device, condition)
    print(json.dumps({"status": "complete", "endpoints": 20}), flush=True)


if __name__ == "__main__":
    main()
