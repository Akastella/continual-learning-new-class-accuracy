"""Standalone recomputation from compact R18 integer decision evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (201, 202, 203, 204, 205)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.shape != labels.shape or predictions.ndim != 1:
        raise ValueError("Prediction/label schema mismatch")
    return float(100.0 * np.equal(predictions, labels).mean())


def mean_sd(values: list[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1)), "n": len(array), "values": array.tolist()}


def verify(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in ("manifest.json", "recomputed.json")
    }
    if set(manifest["files"]) != actual:
        raise ValueError("Compact evidence inventory mismatch")
    for name, record in manifest["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Compact evidence hash mismatch: {name}")
    return manifest


def recompute(root: Path) -> dict[str, object]:
    manifest = verify(root)
    units = []
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        task_rows = []
        for task in range(5):
            with np.load(directory / f"task_{task:02d}_current.npz", allow_pickle=False) as raw:
                labels = raw["mapped_labels"]
                values = {
                    "fresh_accuracy_pct": accuracy(raw["fresh_predictions"], labels),
                    "sgd_actual_probe_accuracy_pct": accuracy(raw["sgd_actual_probe_predictions"], labels),
                    "er_probe_accuracy_pct": accuracy(raw["er_probe_predictions"], labels),
                    "er_actual_accuracy_pct": accuracy(raw["er_actual_predictions"], labels),
                }
            values.update(
                delta_A_er_minus_sgd_pp=values["sgd_actual_probe_accuracy_pct"] - values["er_actual_accuracy_pct"],
                delta_H_er_minus_sgd_pp=values["sgd_actual_probe_accuracy_pct"] - values["er_probe_accuracy_pct"],
            )
            with np.load(directory / f"task_{task:02d}_actual_full_test.npz", allow_pickle=False) as full:
                full_labels = full["mapped_labels"]
                old = full_labels < 2 * task
                seen = full_labels < 2 * (task + 1)
                for method in ("sgd", "er"):
                    predictions = full[method + "_actual_predictions"]
                    values[method + "_actual_seen_accuracy_pct"] = accuracy(predictions[seen], full_labels[seen])
                    values[method + "_actual_old_accuracy_pct"] = accuracy(predictions[old], full_labels[old]) if task else None
                if task == 4:
                    values["sgd_actual_all_accuracy_pct"] = accuracy(full["sgd_actual_predictions"], full_labels)
                    values["er_actual_all_accuracy_pct"] = accuracy(full["er_actual_predictions"], full_labels)
            task_rows.append(values)
        noninitial = task_rows[1:]
        average_keys = (
            "fresh_accuracy_pct",
            "sgd_actual_probe_accuracy_pct",
            "er_probe_accuracy_pct",
            "er_actual_accuracy_pct",
            "delta_A_er_minus_sgd_pp",
            "delta_H_er_minus_sgd_pp",
            "sgd_actual_seen_accuracy_pct",
            "sgd_actual_old_accuracy_pct",
            "er_actual_seen_accuracy_pct",
            "er_actual_old_accuracy_pct",
        )
        means = {key: float(np.mean([row[key] for row in noninitial])) for key in average_keys}
        means.update(
            final_all_test_sgd_actual_accuracy_pct=task_rows[4]["sgd_actual_all_accuracy_pct"],
            final_all_test_er_actual_accuracy_pct=task_rows[4]["er_actual_all_accuracy_pct"],
            final_all_test_er_minus_sgd_pp=task_rows[4]["er_actual_all_accuracy_pct"] - task_rows[4]["sgd_actual_all_accuracy_pct"],
            task_count_delta_A_positive=int(sum(row["delta_A_er_minus_sgd_pp"] > 0 for row in noninitial)),
            task_count_delta_H_negative=int(sum(row["delta_H_er_minus_sgd_pp"] < 0 for row in noninitial)),
            task_count_joint_ordering=int(sum(row["delta_A_er_minus_sgd_pp"] > 0 and row["delta_H_er_minus_sgd_pp"] < 0 for row in noninitial)),
        )
        units.append({"seed": seed, "task_metrics": task_rows, "noninitial_task_means": means})
    summary_keys = tuple(units[0]["noninitial_task_means"])
    summary = {key: mean_sd([unit["noninitial_task_means"][key] for unit in units]) for key in summary_keys}
    return {
        "schema": "r18_architecture_bridge_compact_recomputation_v1",
        "status": "passed",
        "manifest_sha256": sha256(root / "manifest.json"),
        "analysis_unit": "five paired seeds; each seed averages noninitial tasks 1-4 equally",
        "units": units,
        "summary": summary,
        "sign_counts_across_seed_means": {
            "delta_A_positive": int(sum(unit["noninitial_task_means"]["delta_A_er_minus_sgd_pp"] > 0 for unit in units)),
            "delta_H_negative": int(sum(unit["noninitial_task_means"]["delta_H_er_minus_sgd_pp"] < 0 for unit in units)),
            "joint": int(sum(unit["noninitial_task_means"]["delta_A_er_minus_sgd_pp"] > 0 and unit["noninitial_task_means"]["delta_H_er_minus_sgd_pp"] < 0 for unit in units)),
            "n": len(units),
        },
        "sign_counts_across_20_seed_tasks": {
            "delta_A_positive": int(sum(unit["noninitial_task_means"]["task_count_delta_A_positive"] for unit in units)),
            "delta_H_negative": int(sum(unit["noninitial_task_means"]["task_count_delta_H_negative"] for unit in units)),
            "joint": int(sum(unit["noninitial_task_means"]["task_count_joint_ordering"] for unit in units)),
            "n": 20,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/r18_architecture_bridge_compact_v1")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    result = recompute(root)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Preserve existing recomputation: {output}")
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["summary"], "sign_counts": result["sign_counts_across_seed_means"]}))


if __name__ == "__main__":
    main()

