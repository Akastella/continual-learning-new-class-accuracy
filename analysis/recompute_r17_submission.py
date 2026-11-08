"""Recompute the R17 reviewer controls from bundled per-example evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, atol: float = 1e-10) -> None:
    if abs(actual - expected) > atol:
        raise AssertionError(f"numeric mismatch: {actual} != {expected}")


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "n": len(values)}


def score(prediction: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    correct = int((prediction[mask] == labels[mask]).sum())
    return {"correct": correct, "n": n, "accuracy_pct": 100.0 * correct / n}


def check_score(actual: dict, expected: dict) -> None:
    for key in ("correct", "n"):
        if actual[key] != expected[key]:
            raise AssertionError(f"count mismatch for {key}: {actual[key]} != {expected[key]}")
    close(actual["accuracy_pct"], expected["accuracy_pct"])


def recorded_score(scores: dict, key: str) -> dict:
    return scores["current2"] if key == "current2" else scores["all10"][key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "results/r17_review_controls_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.raw.resolve()
    units = []

    for seed in range(201, 211):
        record_path = raw / f"seed_{seed}.json"
        record = read(record_path)
        normalized_accuracies = []
        for expected in record["normalized_compensation"]["tasks"]:
            task = expected["task"]
            path = raw / "normalized_compensation" / f"s{seed}_t{task}" / "endpoint.npz"
            if sha(path) != expected["npz_sha256"]:
                raise AssertionError(f"hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                logits = data["logits"]
                labels = data["labels"].astype(np.int64)
                saved = data["prediction"].astype(np.int64)
            if not np.isfinite(logits).all():
                raise AssertionError(f"nonfinite logits: {path}")
            prediction = logits.argmax(axis=1)
            if not np.array_equal(prediction, saved):
                raise AssertionError(f"saved prediction mismatch: {path}")
            actual = score(prediction, labels, np.ones(len(labels), dtype=bool))
            check_score(actual, expected)
            normalized_accuracies.append(actual["accuracy_pct"])
        normalized_gap = statistics.mean(
            expected["sgd_actual_accuracy_pct"] - accuracy
            for expected, accuracy in zip(record["normalized_compensation"]["tasks"], normalized_accuracies)
        )
        close(normalized_gap, record["normalized_compensation"]["seed_mean_gap_vs_sgd_pp"])

        fixed = {}
        for expected in record["fixed_buffer"]["fixed_endpoints"]:
            epoch = expected["epoch"]
            path = raw / "fixed_buffer" / f"s{seed}_e{epoch}" / "endpoint.npz"
            if sha(path) != expected["npz_sha256"]:
                raise AssertionError(f"hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                logits = data["logits"]
                labels = data["labels"].astype(np.int64)
                task_ids = data["task_ids"].astype(np.int64)
                indices = data["official_indices"].astype(np.int64)
            if not np.isfinite(logits).all() or len(labels) != 10_000:
                raise AssertionError(f"invalid endpoint: {path}")
            if not np.array_equal(np.sort(indices), np.arange(10_000)):
                raise AssertionError(f"incomplete official index set: {path}")
            prediction = logits.argmax(axis=1)
            scores = {
                "current": score(prediction, labels, task_ids == 4),
                "old": score(prediction, labels, task_ids < 4),
                "all": score(prediction, labels, np.ones(len(labels), dtype=bool)),
                "current2": score(logits[:, 8:10].argmax(axis=1) + 8, labels, task_ids == 4),
            }
            for name, actual in scores.items():
                check_score(actual, recorded_score(expected["scores"], name))
            fixed[str(epoch)] = scores
        units.append({
            "seed": seed,
            "seed_json_sha256": sha(record_path),
            "normalized_gap_vs_sgd_pp": normalized_gap,
            "fixed": fixed,
            "evolving": record["fixed_buffer"]["evolving_endpoints"],
        })

    summary = {"normalized_gap_vs_sgd_pp": stats([u["normalized_gap_vs_sgd_pp"] for u in units])}
    for epoch in (8, 16):
        for mode in ("fixed", "evolving"):
            summary[f"epoch{epoch}_{mode}"] = {
                key: stats([
                    recorded_score(u[mode][str(epoch)], key)["accuracy_pct"]
                    if mode == "evolving" else u[mode][str(epoch)][key]["accuracy_pct"]
                    for u in units
                ])
                for key in ("current", "old", "all", "current2")
            }
        summary[f"epoch{epoch}_fixed_minus_evolving"] = {
            key: stats([
                u["fixed"][str(epoch)][key]["accuracy_pct"]
                - recorded_score(u["evolving"][str(epoch)], key)["accuracy_pct"]
                for u in units
            ])
            for key in ("current", "old", "all", "current2")
        }

    report = {
        "schema": "r17_submission_recompute_v1",
        "status": "PASS",
        "evaluation_type": "saved_real_cifar10_labels",
        "all_logits_finite": True,
        "units": units,
        "summary": summary,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"status": "PASS", "seeds": len(units), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
