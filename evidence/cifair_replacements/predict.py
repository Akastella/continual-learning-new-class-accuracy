"""Evaluate one existing CIFAR-100 endpoint on ciFAIR replacement images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "experiments/paa/cross_group_supervision"
sys.path.insert(0, str(SHARED))
import portable_math

core = portable_math.core


def replacement_data(path):
    source = json.loads((HERE / "source.json").read_text(encoding="utf-8"))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source["test_sha256"]:
        raise ValueError("Use the ciFAIR-100 v1.0 test file linked in the README.")
    data = pickle.loads(payload, encoding="bytes")
    images = np.asarray(data["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(data["fine_labels"], dtype=np.int16)
    with (HERE / "replacement_rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = np.array([int(row["test_row"]) for row in csv.DictReader(stream)])
    return images, labels, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-index", type=int, choices=(7, 9), required=True)
    parser.add_argument("--seed", type=int, choices=range(3101, 3106), required=True)
    parser.add_argument("--condition", choices=("full", "restricted"), required=True)
    args = parser.parse_args()
    task, seed = args.task_index, args.seed
    old, seen = 10 * task, 10 * (task + 1)
    models = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))["models"]
    bias = next(row["offset"] for row in models
                if (row["task"], row["seed"], row["condition"]) == (task, seed, args.condition))
    images, labels, rows = replacement_data(args.test_file)
    order, inverse = core.data.class_mapping(seed)
    rows = rows[inverse[labels[rows]] < seen]
    mapped = inverse[labels[rows]]

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = core.make_model(seed + 10000, "cpu").to(memory_format=torch.channels_last)
    state = core.restore(args.checkpoint, model)
    if (state["seed"], state["task"]) != (seed, task):
        raise ValueError("The checkpoint's seed or task does not match the requested evaluation.")
    model.eval()
    mean = torch.tensor(core.data.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(core.data.STD).view(1, 3, 1, 1)
    logits = []
    with torch.inference_mode():
        for batch in np.array_split(images[rows], range(128, len(rows), 128)):
            tensor = torch.from_numpy(batch).to(dtype=torch.float32) / 255
            logits.append(model((tensor - mean) / std).numpy())
    scores = np.concatenate(logits)
    old_pred, new_pred = scores[:, :old].argmax(1), old + scores[:, old:seen].argmax(1)
    indices = np.arange(len(scores))
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, test_rows=rows.astype(np.int16), original_labels=labels[rows],
            mapped_labels=mapped.astype(np.int16), class_order=order.astype(np.int16),
            old_score=scores[indices, old_pred], current_score=scores[indices, new_pred],
            old_prediction=old_pred.astype(np.int16), current_prediction=new_pred.astype(np.int16),
            offset=np.float64(bias), task=np.int64(task), seed=np.int64(seed),
            condition=np.asarray(args.condition))
    print(f"Saved scores for {len(rows)} replacement images to {args.output}.")


if __name__ == "__main__":
    main()
