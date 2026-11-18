"""Train one mixed-replay model with a specified loss on current images."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

SHARED = Path(__file__).resolve().parents[2] / "experiments/paa/cross_group_supervision"
sys.path.insert(0, str(SHARED))
import portable_math as mixed

core = mixed.core


def train_task(model, dataset, images, labels, seed, task, condition, output, deadline):
    old_classes = task * 10
    seen_classes = old_classes + 10
    old_columns = torch.arange(seen_classes, device=images.device) < old_classes

    def cross_entropy(logits, targets):
        if condition == "all_seen_on_nonrepeated":
            half = len(targets) // 4
            row = torch.arange(len(targets), device=targets.device)
            restricted = (row < half) | (row >= 3 * half)
        else:
            restricted = targets >= old_classes
        excluded = restricted[:, None] & old_columns
        return F.cross_entropy(logits.masked_fill(excluded, -1e9), targets)

    output.mkdir()
    original_loss, original_plan = core.F, core.data.epoch_plan
    core.F = SimpleNamespace(pad=F.pad, cross_entropy=cross_entropy)
    core.data.epoch_plan = mixed.mixed_plan
    try:
        return core.train_branch(
            model, dataset, images, labels, seed, task, "er",
            condition, 70, output, deadline, first_task=False,
        )
    finally:
        core.F, core.data.epoch_plan = original_loss, original_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pre_checkpoint", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-index", type=int, choices=(7, 9), required=True)
    parser.add_argument("--seed", type=int, choices=range(3101, 3106), required=True)
    parser.add_argument("--condition", choices=("revisit_current10", "all_seen_on_nonrepeated"),
                        default="revisit_current10")
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args()
    seed, task = args.seed, args.task_index
    state = torch.load(args.pre_checkpoint, map_location="cpu", weights_only=True)
    if (state["seed"], state["task"]) != (seed, task) or not state["new_rows_reset"]:
        raise ValueError("Supply the original ER pre-task checkpoint for this seed and task.")
    if core.state_digest(state["model"]) != state["state_sha256"]:
        raise ValueError("The checkpoint does not match its recorded model state.")
    if args.check_inputs_only:
        print(json.dumps({"seed": seed, "task_index": task, "seen_classes": (task + 1) * 10}))
        return

    started = time.monotonic()
    args.output.mkdir(parents=True)
    dataset = core.data.load_train(args.data_root, download=False)
    images = torch.from_numpy(dataset.images.copy()).to("cuda")
    _, inverse = core.data.class_mapping(seed)
    labels = torch.as_tensor(inverse[dataset.labels], dtype=torch.long, device="cuda")
    model = core.make_model(seed + 10000, "cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    record = train_task(model, dataset, images, labels, seed, task, args.condition,
                        args.output / "run", started + 3300)
    record["checkpoint"] = core.save_checkpoint(args.output / "endpoint.pt", model,
        {"seed": seed, "task": task, "epochs": 70, "branch": args.condition})
    core.write_json(args.output / "training.json", record)
    print(json.dumps({"seed": seed, "task_index": task, "seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
