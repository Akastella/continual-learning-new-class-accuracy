"""Validate and summarize the five-seed official CBP port."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(201, 206))
PROFILES = ("nature_cifar", "official_conv_default")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values), "sd": statistics.stdev(values),
        "n": len(values), "values": values,
    }


def latex_pm(item: dict, digits: int = 2) -> str:
    return f"${item['mean']:.{digits}f}\\pm{item['sd']:.{digits}f}$"


def write_paper_outputs(summary: dict) -> None:
    generated = ROOT / "paper_r9/generated"
    generated.mkdir(parents=True, exist_ok=True)
    labels = {
        "nature_cifar": "CBP: Nature CIFAR",
        "official_conv_default": "CBP: Conv default (sens.)",
    }
    rows = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Profile & $a$ (\%) & $p$ (\%) & $C$ (pp) & Final all (\%) & Replaced \\",
        r"\midrule",
    ]
    for profile in PROFILES:
        item = summary[profile]
        rows.append(
            f"{labels[profile]} & {latex_pm(item['a'])} & {latex_pm(item['p'])} & "
            f"{latex_pm(item['C'], 3)} & {latex_pm(item['final_all'])} & "
            f"{latex_pm(item['total_replacements'], 1)} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "cbp_official_table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    nature, sensitivity = summary["nature_cifar"], summary["official_conv_default"]
    commands = {
        "CBPNatureActual": nature["a"]["mean"],
        "CBPNatureActualSD": nature["a"]["sd"],
        "CBPNatureProbe": nature["p"]["mean"],
        "CBPNatureProbeSD": nature["p"]["sd"],
        "CBPNatureCost": nature["C"]["mean"],
        "CBPNatureCostSD": nature["C"]["sd"],
        "CBPNatureFinalAll": nature["final_all"]["mean"],
        "CBPNatureFinalAllSD": nature["final_all"]["sd"],
        "CBPNatureReplacements": nature["total_replacements"]["mean"],
        "CBPNatureVsSGDActual": nature["cbp_minus_sgd_actual_pp"]["mean"],
        "CBPNatureVsERActual": nature["cbp_minus_er_actual_pp"]["mean"],
        "CBPNatureVsERProbe": nature["cbp_minus_er_probe_pp"]["mean"],
        "CBPNatureVsERFinalAll": nature["cbp_minus_er_final_all_pp"]["mean"],
        "CBPDefaultActual": sensitivity["a"]["mean"],
        "CBPDefaultActualSD": sensitivity["a"]["sd"],
        "CBPDefaultProbe": sensitivity["p"]["mean"],
        "CBPDefaultProbeSD": sensitivity["p"]["sd"],
        "CBPDefaultCost": sensitivity["C"]["mean"],
        "CBPDefaultCostSD": sensitivity["C"]["sd"],
        "CBPDefaultFinalAll": sensitivity["final_all"]["mean"],
        "CBPDefaultFinalAllSD": sensitivity["final_all"]["sd"],
        "CBPDefaultReplacements": sensitivity["total_replacements"]["mean"],
        "CBPDefaultVsSGDActual": sensitivity["cbp_minus_sgd_actual_pp"]["mean"],
    }
    macro_lines = [
        f"\\newcommand{{\\{name}}}{{{value:.4f}}}"
        if name == "CBPNatureVsSGDActual"
        else f"\\newcommand{{\\{name}}}{{{value:.3f}}}"
        for name, value in commands.items()
    ]
    (generated / "cbp_official_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")


def endpoint(row: dict) -> dict:
    tasks = row["tasks"][1:]
    final = row["tasks"][-1]["post_task_evaluation"]
    return {
        "F": statistics.mean(t["fresh"]["evaluation"]["accuracy_pct"] for t in tasks),
        "p": statistics.mean(t["probe"]["evaluation"]["accuracy_pct"] for t in tasks),
        "a": statistics.mean(t["actual"]["evaluation"]["accuracy_pct"] for t in tasks),
        "A": statistics.mean(t["actual_gap_pp"] for t in tasks),
        "H": statistics.mean(t["history_gap_pp"] for t in tasks),
        "C": statistics.mean(t["objective_cost_pp"] for t in tasks),
        "final_all": sum(x["correct"] for x in final) / sum(x["n"] for x in final) * 100.0,
        "replacements": row["cbp_final"]["replacements_by_layer"],
    }


def main() -> None:
    input_hashes, units = {}, []
    exact_checks = []
    run_source_hashes = []
    artifact_files_verified = 0
    for profile in PROFILES:
        for seed in SEEDS:
            result_path = ROOT / "results/r18_cbp_official" / f"{profile}_s{seed}" / "result.json"
            sgd_path = ROOT / "results/r9_followup" / f"split_cifar10_classil_sgd_s{seed}" / "result.json"
            er_path = ROOT / "results/r9_followup" / f"split_cifar10_classil_er_s{seed}" / "result.json"
            row, sgd, er = read(result_path), read(sgd_path), read(er_path)
            input_hashes[result_path.relative_to(ROOT).as_posix()] = sha(result_path)
            input_hashes[sgd_path.relative_to(ROOT).as_posix()] = sha(sgd_path)
            input_hashes[er_path.relative_to(ROOT).as_posix()] = sha(er_path)
            if row.get("status") != "complete" or row["config"]["profile"] != profile:
                raise ValueError(f"Incomplete or mislabeled result: {result_path}")
            run_root = result_path.parent
            started = read(run_root / "run_started.json")
            run_source_hashes.append(started["source_sha256"])
            artifact_manifest = read(run_root / "artifact_manifest.json")["sha256"]
            for relative, expected_hash in artifact_manifest.items():
                artifact = run_root / relative
                if not artifact.is_file() or sha(artifact) != expected_hash:
                    raise ValueError(f"Artifact hash mismatch: {artifact}")
                artifact_files_verified += 1
            for task_id, (cbp_task, sgd_task) in enumerate(zip(row["tasks"], sgd["tasks"])):
                frozen_fresh = sgd_task["fresh"]["trace"][-1]["evaluation"]
                exact_keys = ("n", "correct", "accuracy_pct", "prediction_sha256")
                fresh_matches = all(cbp_task["fresh"]["evaluation"][key] == frozen_fresh[key]
                                    for key in exact_keys)
                fresh_matches &= cbp_task["fresh"]["final_state_sha256"] == sgd_task["fresh"]["final_state_sha256"]
                if not fresh_matches:
                    raise ValueError(f"Fresh mismatch {profile=} {seed=} {task_id=}")
                if cbp_task["actual"]["batch_order_sha256"] != sgd_task["actual"]["batch_order_sha256"]:
                    raise ValueError("Batch-order mismatch")
                if profile == "nature_cifar" and sum(
                    cbp_task["actual"]["cumulative_cbp"]["replacements_by_layer"]) == 0:
                    same = cbp_task["actual"]["final_state_sha256"] == sgd_task["actual"]["final_state_sha256"]
                    exact_checks.append(same)
                    if not same:
                        raise ValueError(f"Pre-replacement SGD identity failed {seed=} {task_id=}")
            metrics = endpoint(row)
            sgd_metrics = {
                "p": statistics.mean(t["probe"]["trace"][-1]["evaluation"]["accuracy_pct"] for t in sgd["tasks"][1:]),
                "a": statistics.mean(t["actual"]["trace"][-1]["evaluation"]["accuracy_pct"] for t in sgd["tasks"][1:]),
                "A": statistics.mean(t["actual_gap_pp"] for t in sgd["tasks"][1:]),
                "H": statistics.mean(t["history_gap_pp"] for t in sgd["tasks"][1:]),
            }
            er_metrics = {
                "p": statistics.mean(t["probe"]["trace"][-1]["evaluation"]["accuracy_pct"] for t in er["tasks"][1:]),
                "a": statistics.mean(t["actual"]["trace"][-1]["evaluation"]["accuracy_pct"] for t in er["tasks"][1:]),
                "A": statistics.mean(t["actual_gap_pp"] for t in er["tasks"][1:]),
                "H": statistics.mean(t["history_gap_pp"] for t in er["tasks"][1:]),
                "final_all": sum(x["correct"] for x in er["tasks"][-1]["post_task_evaluation"]) /
                             sum(x["n"] for x in er["tasks"][-1]["post_task_evaluation"]) * 100.0,
            }
            units.append({
                "profile": profile, "seed": seed, **metrics,
                "cbp_minus_sgd_actual_pp": metrics["a"] - sgd_metrics["a"],
                "cbp_minus_sgd_probe_pp": metrics["p"] - sgd_metrics["p"],
                "cbp_minus_sgd_actual_gap_pp": metrics["A"] - sgd_metrics["A"],
                "cbp_minus_sgd_history_gap_pp": metrics["H"] - sgd_metrics["H"],
                "cbp_minus_er_actual_pp": metrics["a"] - er_metrics["a"],
                "cbp_minus_er_probe_pp": metrics["p"] - er_metrics["p"],
                "cbp_minus_er_actual_gap_pp": metrics["A"] - er_metrics["A"],
                "cbp_minus_er_history_gap_pp": metrics["H"] - er_metrics["H"],
                "cbp_minus_er_final_all_pp": metrics["final_all"] - er_metrics["final_all"],
            })
    summary = {}
    numeric = ("F", "p", "a", "A", "H", "C", "final_all", "cbp_minus_sgd_actual_pp",
               "cbp_minus_sgd_probe_pp", "cbp_minus_sgd_actual_gap_pp", "cbp_minus_sgd_history_gap_pp",
               "cbp_minus_er_actual_pp", "cbp_minus_er_probe_pp", "cbp_minus_er_actual_gap_pp",
               "cbp_minus_er_history_gap_pp", "cbp_minus_er_final_all_pp")
    for profile in PROFILES:
        rows = [x for x in units if x["profile"] == profile]
        summary[profile] = {key: describe([x[key] for x in rows]) for key in numeric}
        summary[profile]["replacements_by_layer"] = [sum(x["replacements"][i] for x in rows) / len(rows) for i in range(4)]
        summary[profile]["total_replacements"] = describe([sum(x["replacements"]) for x in rows])
    if any(x != run_source_hashes[0] for x in run_source_hashes[1:]):
        raise ValueError("Run source snapshots differ across the campaign")
    source_record = read(ROOT / "third_party/r18_loss_of_plasticity_official/SOURCE.json")
    for relative, expected_hash in source_record["primary_files"].items():
        if sha(ROOT / "third_party/r18_loss_of_plasticity_official" / relative) != expected_hash:
            raise ValueError(f"Official source snapshot drift: {relative}")
    output = {
        "schema": "r18_official_cbp_evidence_v1", "status": "complete",
        "posthoc_descriptive": True, "seeds": list(SEEDS), "units": units, "summary": summary,
        "integrity_checks": {
            "fresh_matches_frozen_sgd": True,
            "batch_orders_match_frozen_sgd": True,
            "nature_profile_pre_replacement_exact_sgd_states": all(exact_checks),
            "nature_profile_pre_replacement_state_comparisons": len(exact_checks),
            "campaign_source_snapshots_identical": True,
            "artifact_files_hash_verified": artifact_files_verified,
            "official_primary_source_hashes_verified": len(source_record["primary_files"]),
        },
        "input_sha256": input_hashes,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "results/r18_cbp_official/evidence.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_paper_outputs(summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
