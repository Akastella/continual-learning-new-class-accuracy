"""Recompute the R18 architecture bridge from saved per-example logits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (201, 202, 203, 204, 205)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "n": len(array),
        "values": array.tolist(),
    }


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.shape != (len(labels), 10) or logits.dtype != np.float32:
        raise ValueError("Unexpected logit schema")
    return float(100.0 * np.equal(logits.argmax(1), labels).mean())


def verify_inventory(directory: Path) -> dict[str, object]:
    manifest = json.loads((directory / "artifact_sha256.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported artifact manifest")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError(f"Artifact inventory mismatch in {directory}")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {path}")
    return manifest


def read_r18_seed(directory: Path, seed: int) -> dict[str, object]:
    verify_inventory(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result["config"]["seed"] != seed:
        raise ValueError(f"Incomplete or wrong seed result: {directory}")
    task_rows: list[dict[str, float]] = []
    for task_id, record in enumerate(result["tasks"]):
        path = directory / record["current_logits"]["path"]
        if file_sha256(path) != record["current_logits"]["file_sha256"]:
            raise ValueError("Current logit hash mismatch")
        with np.load(path, allow_pickle=False) as raw:
            labels = raw["mapped_labels"]
            values = {
                "fresh_accuracy_pct": accuracy(raw["fresh_logits"], labels),
                "sgd_actual_probe_accuracy_pct": accuracy(raw["sgd_actual_probe_logits"], labels),
                "er_probe_accuracy_pct": accuracy(raw["er_probe_logits"], labels),
                "er_actual_accuracy_pct": accuracy(raw["er_actual_logits"], labels),
            }
        values.update(
            sgd_actual_gap_pp=values["fresh_accuracy_pct"] - values["sgd_actual_probe_accuracy_pct"],
            sgd_history_gap_pp=values["fresh_accuracy_pct"] - values["sgd_actual_probe_accuracy_pct"],
            er_actual_gap_pp=values["fresh_accuracy_pct"] - values["er_actual_accuracy_pct"],
            er_history_gap_pp=values["fresh_accuracy_pct"] - values["er_probe_accuracy_pct"],
            delta_A_er_minus_sgd_pp=values["sgd_actual_probe_accuracy_pct"] - values["er_actual_accuracy_pct"],
            delta_H_er_minus_sgd_pp=values["sgd_actual_probe_accuracy_pct"] - values["er_probe_accuracy_pct"],
        )
        for key, value in values.items():
            if not np.isclose(value, record["metrics"][key], atol=1e-12, rtol=0):
                raise ValueError(f"JSON metric differs from raw logits: seed={seed}, task={task_id}, key={key}")
        full_path = directory / record["actual_full_test_logits"]["path"]
        if file_sha256(full_path) != record["actual_full_test_logits"]["file_sha256"]:
            raise ValueError("Full-test logit hash mismatch")
        with np.load(full_path, allow_pickle=False) as full:
            full_labels = full["mapped_labels"]
            old_mask = full_labels < 2 * task_id
            seen_mask = full_labels < 2 * (task_id + 1)
            for method in ("sgd", "er"):
                logits = full[method + "_actual_logits"]
                values[method + "_actual_seen_accuracy_pct"] = accuracy(logits[seen_mask], full_labels[seen_mask])
                values[method + "_actual_old_accuracy_pct"] = (
                    accuracy(logits[old_mask], full_labels[old_mask]) if task_id > 0 else None
                )
        task_rows.append(values)
    noninitial = task_rows[1:]
    keys = tuple(noninitial[0])
    seed_means = {key: float(np.mean([row[key] for row in noninitial])) for key in keys}
    final_path = directory / result["tasks"][-1]["actual_full_test_logits"]["path"]
    with np.load(final_path, allow_pickle=False) as raw:
        final_labels = raw["mapped_labels"]
        seed_means["final_all_test_sgd_actual_accuracy_pct"] = accuracy(raw["sgd_actual_logits"], final_labels)
        seed_means["final_all_test_er_actual_accuracy_pct"] = accuracy(raw["er_actual_logits"], final_labels)
    seed_means["final_all_test_er_minus_sgd_pp"] = (
        seed_means["final_all_test_er_actual_accuracy_pct"]
        - seed_means["final_all_test_sgd_actual_accuracy_pct"]
    )
    seed_means["task_count_delta_A_positive"] = int(sum(row["delta_A_er_minus_sgd_pp"] > 0 for row in noninitial))
    seed_means["task_count_delta_H_negative"] = int(sum(row["delta_H_er_minus_sgd_pp"] < 0 for row in noninitial))
    seed_means["task_count_joint_ordering"] = int(
        sum(row["delta_A_er_minus_sgd_pp"] > 0 and row["delta_H_er_minus_sgd_pp"] < 0 for row in noninitial)
    )
    return {"seed": seed, "task_metrics": task_rows, "noninitial_task_means": seed_means, "wall_seconds": result["wall_seconds"]}


def read_compact_seed(seed: int) -> dict[str, float]:
    rows = {}
    for method in ("sgd", "er"):
        path = ROOT / "results" / "r9_followup" / f"split_cifar10_classil_{method}_s{seed}" / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        current = result["tasks"][1:]
        rows[method] = {
            "fresh": float(np.mean([task["fresh"]["trace"][-1]["evaluation"]["accuracy_pct"] for task in current])),
            "actual": float(np.mean([task["actual"]["trace"][-1]["evaluation"]["accuracy_pct"] for task in current])),
            "probe": float(np.mean([task["probe"]["trace"][-1]["evaluation"]["accuracy_pct"] for task in current])),
            "final_all": float(np.mean([row["accuracy_pct"] for row in result["tasks"][-1]["post_task_evaluation"]])),
        }
    return {
        "fresh_accuracy_pct": rows["sgd"]["fresh"],
        "sgd_actual_probe_accuracy_pct": rows["sgd"]["actual"],
        "er_actual_accuracy_pct": rows["er"]["actual"],
        "er_probe_accuracy_pct": rows["er"]["probe"],
        "delta_A_er_minus_sgd_pp": rows["sgd"]["actual"] - rows["er"]["actual"],
        "delta_H_er_minus_sgd_pp": rows["sgd"]["probe"] - rows["er"]["probe"],
        "final_all_test_sgd_actual_accuracy_pct": rows["sgd"]["final_all"],
        "final_all_test_er_actual_accuracy_pct": rows["er"]["final_all"],
        "final_all_test_er_minus_sgd_pp": rows["er"]["final_all"] - rows["sgd"]["final_all"],
    }


def paired_bootstrap(values: np.ndarray, rng_seed: int = 2026091901) -> dict[str, object]:
    if values.shape != (5, 2):
        raise ValueError("The frozen bridge expects five paired seeds and two endpoints")
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    indices = rng.integers(0, len(values), size=(100000, len(values)))
    draws = values[indices].mean(axis=1)
    interval = np.quantile(draws, [0.0125, 0.9875], axis=0, method="linear")
    return {
        "method": "paired-seed percentile bootstrap with 97.5% marginal intervals",
        "role": "descriptive architecture bridge; not the paper's prespecified primary inference",
        "rng_seed": rng_seed,
        "n_resamples": len(indices),
        "resample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "delta_A_interval_pp": interval[:, 0].tolist(),
        "delta_H_interval_pp": interval[:, 1].tolist(),
    }


def analyze(root: Path) -> dict[str, object]:
    units = [read_r18_seed(root / f"seed{seed}", seed) for seed in SEEDS]
    keys = tuple(units[0]["noninitial_task_means"])
    summary = {
        key: mean_sd([unit["noninitial_task_means"][key] for unit in units])
        for key in keys
    }
    compact = {seed: read_compact_seed(seed) for seed in SEEDS}
    compact_summary = {
        key: mean_sd([compact[seed][key] for seed in SEEDS])
        for key in compact[SEEDS[0]]
    }
    interactions = {}
    for key in (
        "delta_A_er_minus_sgd_pp",
        "delta_H_er_minus_sgd_pp",
        "final_all_test_sgd_actual_accuracy_pct",
        "final_all_test_er_actual_accuracy_pct",
        "final_all_test_er_minus_sgd_pp",
    ):
        interactions[key] = mean_sd(
            [
                unit["noninitial_task_means"][key] - compact[unit["seed"]][key]
                for unit in units
            ]
        )
    endpoint_values = np.asarray(
        [
            [
                unit["noninitial_task_means"]["delta_A_er_minus_sgd_pp"],
                unit["noninitial_task_means"]["delta_H_er_minus_sgd_pp"],
            ]
            for unit in units
        ],
        dtype=np.float64,
    )
    return {
        "schema": "r18_architecture_bridge_analysis_v1",
        "status": "complete",
        "analysis_unit": "five paired seeds; each seed averages tasks 1-4 equally",
        "units": units,
        "resnet32_summary": summary,
        "compact_cnn_same_seed_reference": {str(seed): compact[seed] for seed in SEEDS},
        "compact_cnn_same_seed_summary": compact_summary,
        "resnet32_minus_compact_interaction": interactions,
        "descriptive_bootstrap": paired_bootstrap(endpoint_values),
        "sign_counts_across_seed_means": {
            "delta_A_positive": int(sum(row[0] > 0 for row in endpoint_values)),
            "delta_H_negative": int(sum(row[1] < 0 for row in endpoint_values)),
            "joint_delta_A_positive_delta_H_negative": int(sum(row[0] > 0 and row[1] < 0 for row in endpoint_values)),
            "n_seeds": len(endpoint_values),
        },
        "sign_counts_across_20_seed_tasks": {
            "delta_A_positive": int(sum(unit["noninitial_task_means"]["task_count_delta_A_positive"] for unit in units)),
            "delta_H_negative": int(sum(unit["noninitial_task_means"]["task_count_delta_H_negative"] for unit in units)),
            "joint_delta_A_positive_delta_H_negative": int(sum(unit["noninitial_task_means"]["task_count_joint_ordering"] for unit in units)),
            "n_seed_tasks": 20,
        },
        "interpretation_gate": {
            "delta_A_positive": bool(summary["delta_A_er_minus_sgd_pp"]["mean"] > 0),
            "delta_H_negative": bool(summary["delta_H_er_minus_sgd_pp"]["mean"] < 0),
            "same_mean_ordering_pattern": bool(
                summary["delta_A_er_minus_sgd_pp"]["mean"] > 0
                and summary["delta_H_er_minus_sgd_pp"]["mean"] < 0
            ),
            "scope": "same CIFAR-10 stream and four-epoch optimizer protocol; architecture-only bridge to ResNet-32-GN8",
        },
        "analysis_source_sha256": file_sha256(Path(__file__)),
    }


def _cell(record: dict[str, object], signed: bool = False) -> str:
    mean = float(record["mean"])
    sd = float(record["sample_sd"])
    prefix = "+" if signed and mean > 0 else ""
    return f"${prefix}{mean:.2f}\\pm{sd:.2f}$"


def write_tex(result: dict[str, object], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    resnet = result["resnet32_summary"]
    compact = result["compact_cnn_same_seed_summary"]
    interaction = result["resnet32_minus_compact_interaction"]
    rows = [
        "% Five paired seeds; every seed averages noninitial tasks 1--4.",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Architecture & Fresh $F$ & SGD $a{=}p$ & ER $a$ & ER $p$ & $\Delta A$ & $\Delta H$ & Final SGD & Final ER \\",
        r"\midrule",
        "Compact CNN & " + " & ".join(
            [
                _cell(compact["fresh_accuracy_pct"]),
                _cell(compact["sgd_actual_probe_accuracy_pct"]),
                _cell(compact["er_actual_accuracy_pct"]),
                _cell(compact["er_probe_accuracy_pct"]),
                _cell(compact["delta_A_er_minus_sgd_pp"], True),
                _cell(compact["delta_H_er_minus_sgd_pp"], True),
                _cell(compact["final_all_test_sgd_actual_accuracy_pct"]),
                _cell(compact["final_all_test_er_actual_accuracy_pct"]),
            ]
        ) + r" \\",
        "ResNet-32--GN8 & " + " & ".join(
            [
                _cell(resnet["fresh_accuracy_pct"]),
                _cell(resnet["sgd_actual_probe_accuracy_pct"]),
                _cell(resnet["er_actual_accuracy_pct"]),
                _cell(resnet["er_probe_accuracy_pct"]),
                _cell(resnet["delta_A_er_minus_sgd_pp"], True),
                _cell(resnet["delta_H_er_minus_sgd_pp"], True),
                _cell(resnet["final_all_test_sgd_actual_accuracy_pct"]),
                _cell(resnet["final_all_test_er_actual_accuracy_pct"]),
            ]
        ) + r" \\",
        r"\midrule",
        "ResNet $-$ CNN & -- & -- & -- & -- & "
        + _cell(interaction["delta_A_er_minus_sgd_pp"], True)
        + " & "
        + _cell(interaction["delta_H_er_minus_sgd_pp"], True)
        + " & "
        + _cell(interaction["final_all_test_sgd_actual_accuracy_pct"], True)
        + " & "
        + _cell(interaction["final_all_test_er_actual_accuracy_pct"], True)
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (directory / "architecture_bridge_table.tex").write_text("\n".join(rows), encoding="utf-8")
    names = {
        "BridgeResFresh": resnet["fresh_accuracy_pct"],
        "BridgeResSGDActual": resnet["sgd_actual_probe_accuracy_pct"],
        "BridgeResERActual": resnet["er_actual_accuracy_pct"],
        "BridgeResERProbe": resnet["er_probe_accuracy_pct"],
        "BridgeResDeltaA": resnet["delta_A_er_minus_sgd_pp"],
        "BridgeResDeltaH": resnet["delta_H_er_minus_sgd_pp"],
        "BridgeResFinalSGD": resnet["final_all_test_sgd_actual_accuracy_pct"],
        "BridgeResFinalER": resnet["final_all_test_er_actual_accuracy_pct"],
        "BridgeCompactDeltaA": compact["delta_A_er_minus_sgd_pp"],
        "BridgeCompactDeltaH": compact["delta_H_er_minus_sgd_pp"],
        "BridgeInteractionDeltaA": interaction["delta_A_er_minus_sgd_pp"],
        "BridgeInteractionDeltaH": interaction["delta_H_er_minus_sgd_pp"],
    }
    macro_lines = []
    for name, record in names.items():
        macro_lines.append(f"\\newcommand{{\\{name}}}{{{float(record['mean']):.2f}}}")
        macro_lines.append(f"\\newcommand{{\\{name}SD}}{{{float(record['sample_sd']):.2f}}}")
    signs = result["sign_counts_across_seed_means"]
    macro_lines.extend(
        [
            f"\\newcommand{{\\BridgeSeedCount}}{{{signs['n_seeds']}}}",
            f"\\newcommand{{\\BridgeDeltaAPositiveSeeds}}{{{signs['delta_A_positive']}}}",
            f"\\newcommand{{\\BridgeDeltaHNegativeSeeds}}{{{signs['delta_H_negative']}}}",
            f"\\newcommand{{\\BridgeJointSeeds}}{{{signs['joint_delta_A_positive_delta_H_negative']}}}",
        ]
    )
    (directory / "architecture_bridge_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/r18_architecture_bridge_v1")
    parser.add_argument("--output", default="results/r18_architecture_bridge_v1/analysis.json")
    parser.add_argument("--tex-dir", default="paper_r9/generated")
    args = parser.parse_args()
    result = analyze(ROOT / args.root)
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Preserve existing analysis: {output}")
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_tex(result, ROOT / args.tex_dir)
    print(json.dumps({"status": result["status"], "summary": result["resnet32_summary"], "gate": result["interpretation_gate"]}))


if __name__ == "__main__":
    main()
