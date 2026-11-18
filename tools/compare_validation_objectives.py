"""Compare two validation objectives on exactly the same feasible score choices."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np


SEEDS = tuple(range(201, 211))
THRESHOLDS = (30, 35)
OBJECTIVES = ("new_accuracy", "overall_accuracy")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "values": array.tolist(),
            "mean": float(array.mean()) if len(array) else None,
            "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "positive": int((array > 0).sum()), "negative": int((array < 0).sum()),
            "zero": int((array == 0).sum())}


def group_counts(logits, labels):
    old_prediction = logits[:, :8].argmax(axis=1)
    new_prediction = 8 + logits[:, 8:].argmax(axis=1)
    margins = logits[:, :8].max(axis=1)-logits[:, 8:].max(axis=1)
    old_correct = (labels < 8) & (old_prediction == labels)
    new_correct = (labels >= 8) & (new_prediction == labels)
    return np.sort(margins[old_correct]), np.sort(margins[new_correct])


def correct_counts(old_margins, new_margins, scalars):
    # Equality chooses old. Current winners require a strictly larger scalar.
    old = len(old_margins)-np.searchsorted(old_margins, scalars, side="left")
    new = np.searchsorted(new_margins, scalars, side="left")
    return old, new


def integer_scores(logits, labels, scalar):
    shifted = logits.copy()
    shifted[:, 8:] += scalar
    correct = shifted.argmax(axis=1) == labels
    old = labels < 8
    old_n, new_n = int(old.sum()), int((~old).sum())
    old_correct, new_correct = int(correct[old].sum()), int(correct[~old].sum())
    return {"old_n": old_n, "new_n": new_n, "old_correct": old_correct,
            "new_correct": new_correct, "all_correct": old_correct+new_correct,
            "old_pct": 100*old_correct/old_n, "new_pct": 100*new_correct/new_n,
            "all_pct": 100*(old_correct+new_correct)/(old_n+new_n)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh output directory, normally run1 beside this script")
    args = parser.parse_args()
    root, output = args.artifact.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    module_path = root / "tools/recompute_score_frontier.py"
    spec = importlib.util.spec_from_file_location("original_frontier", module_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    names = original.all_names()
    assert len(names) == len(set(names)) == 30
    inputs = {}

    def record(path):
        relative = path.relative_to(root).as_posix()
        if relative not in inputs:
            inputs[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    record(module_path)
    record(root / "tools/compare_fixed_margin.py")
    validation, feasible, choices, availability = {}, [], [], []
    coverage_checks = full_argmax_checks = 0
    for seed in SEEDS:
        reference = None
        model_points = {}
        for name in names:
            if "_current_only_step_" in name:
                method = name.split("_")[0]
                path = root / f"evidence/retention_selection_original/split_cifar10_classil_{method}_s{seed}/validation.npz"
            else:
                path = root / f"evidence/weight_retention/validation_predictions/s{seed}_{name}.npz"
            record(path)
            logits, labels = original.loadval(root, seed, name)
            with np.load(path, allow_pickle=False) as archive:
                indices = archive["indices"].copy()
            assert logits.shape == (1000, 10) and np.isfinite(logits).all()
            assert len(np.unique(indices)) == 1000
            assert np.array_equal(np.bincount(labels), np.full(10, 100))
            if reference is None:
                reference = labels, indices
            else:
                assert np.array_equal(labels, reference[0])
                assert np.array_equal(indices, reference[1])
            validation[seed, name] = logits, labels, indices
            old_margins, new_margins = group_counts(logits, labels)
            scalars = np.unique(old_margins)
            old_correct, new_correct = correct_counts(old_margins, new_margins, scalars)
            points = []
            for scalar, old_count, new_count in zip(scalars, old_correct, new_correct):
                if old_count < 240:
                    continue
                row = {"seed": seed, "candidate": name, "scalar": float(scalar),
                       "validation_old_correct": int(old_count),
                       "validation_new_correct": int(new_count),
                       "validation_all_correct": int(old_count+new_count),
                       "validation_old_n": 800, "validation_new_n": 200,
                       "feasible30": bool(old_count >= 240), "feasible35": bool(old_count >= 280)}
                direct = integer_scores(logits, labels, scalar)
                assert direct["old_correct"] == old_count and direct["new_correct"] == new_count
                full_argmax_checks += 1
                points.append(row)
                feasible.append(row)
            model_points[name] = points
            # Independently sweep every accuracy-changing event, at equality and just after it.
            # This covers old/new correct-count states, including the all-old state.
            events = np.unique(np.concatenate((old_margins, new_margins)))
            probes = np.r_[np.nextafter(events[0], -np.inf), events, np.nextafter(events, np.inf)]
            old_probe, new_probe = correct_counts(old_margins, new_margins, probes)
            for threshold in THRESHOLDS:
                required = threshold*8
                eligible = [point for point in points if point[f"feasible{threshold}"]]
                state_mask = old_probe >= required
                assert bool(eligible) == bool(state_mask.any())
                if eligible:
                    assert max(point["validation_new_correct"] for point in eligible) == int(new_probe[state_mask].max())
                    assert max(point["validation_all_correct"] for point in eligible) == int((old_probe+new_probe)[state_mask].max())
                coverage_checks += 1
        for threshold in THRESHOLDS:
            points = [point for model in names for point in model_points[model] if point[f"feasible{threshold}"]]
            availability.append({"seed": seed, "validation_threshold_pct": threshold,
                                 "eligible_models": len({point["candidate"] for point in points}),
                                 "shared_feasible_operating_points": len(points),
                                 "both_objectives_available": bool(points)})
            for objective in OBJECTIVES:
                field = "validation_new_correct" if objective == "new_accuracy" else "validation_all_correct"
                winner = min(points, key=lambda point: (-point[field], point["candidate"], -point["scalar"])) if points else None
                choices.append({"seed": seed, "validation_threshold_pct": threshold,
                                "test_old_target_pct": 30, "objective": objective,
                                "available": winner is not None,
                                **({key: winner[key] for key in ("candidate", "scalar", "validation_old_correct", "validation_new_correct", "validation_all_correct")} if winner else {})})

    data = root / "evidence/score_frontier"
    recorded_choices = data / "choices.json"
    recorded_margin = data / "fixed_margin/test_scores.csv"
    record(recorded_choices)
    record(recorded_margin)
    references = json.loads(recorded_choices.read_text(encoding="utf-8"))["rows"]
    margin_rows = list(csv.DictReader(recorded_margin.open(encoding="utf-8", newline="")))
    for choice in choices:
        if choice["objective"] != "new_accuracy":
            continue
        seed, threshold = choice["seed"], choice["validation_threshold_pct"]
        if threshold == 30:
            reference = next(row for row in references if row["seed"] == seed and row["q"] == 30 and row["policy"] == "offset30")
            candidate, scalar = reference["candidate"], reference["b"]
        else:
            reference = next(row for row in margin_rows if int(row["seed"]) == seed and row["dataset"] == "cifar101" and row["policy"] == "offset30")
            candidate, scalar = reference["candidate"], float(reference["scalar"])
        assert choice["candidate"] == candidate and choice["scalar"] == scalar
    write_csv(output / "validation_operating_points.csv", feasible)
    write_csv(output / "availability.csv", availability)
    write_json(output / "validation_choices.json", {"started_utc": started_utc,
        "choices_fixed_before_test_load": True, "tie_rule": "objective count descending, candidate name ascending, scalar descending", "choices": choices})
    choice_hash = sha256(output / "validation_choices.json")
    choices_fixed_utc = datetime.now(timezone.utc).isoformat()

    # Test access starts only after all 40 choices have been persisted and hashed.
    test_rows = []
    test_inputs = 0
    for dataset in ("cifar101", "cifar102"):
        for seed in SEEDS:
            reference = None
            for name in names:
                path = data / f"diagnostics/test_logits/{dataset}_s{seed}_{name}.npz"
                if not path.exists():
                    path = data / f"test_logits/{dataset}_s{seed}_{name}.npz"
                record(path)
                with np.load(path, allow_pickle=False) as archive:
                    logits, labels, indices = archive["logits"].astype(float), archive["labels"], archive["indices"]
                test_inputs += 1
                expected_n = 1000 if dataset == "cifar101" else 2000
                assert logits.shape == (expected_n, 10) and np.isfinite(logits).all()
                assert len(np.unique(indices)) == expected_n
                assert np.array_equal(np.bincount(labels), np.full(10, expected_n//10))
                if reference is None:
                    reference = labels.copy(), indices.copy()
                else:
                    assert np.array_equal(labels, reference[0]) and np.array_equal(indices, reference[1])
                if dataset == "cifar101":
                    assert not np.intersect1d(indices, validation[seed, name][2]).size
                for choice in choices:
                    if choice["seed"] == seed and choice["available"] and choice["candidate"] == name:
                        scores = integer_scores(logits, labels, choice["scalar"])
                        violation = scores["old_correct"]*100 < 30*scores["old_n"]
                        test_rows.append({"dataset": dataset, **choice, **scores,
                                          "below30_old": violation})
    assert test_inputs == 600 and len(test_rows) == 80
    assert sha256(output / "validation_choices.json") == choice_hash
    recorded_test_path = data / "test_scores.csv"
    record(recorded_test_path)
    recorded_test_rows = list(csv.DictReader(recorded_test_path.open(encoding="utf-8", newline="")))
    matched_test_rows = 0
    for row in test_rows:
        if row["objective"] != "new_accuracy":
            continue
        source_rows = recorded_test_rows if row["validation_threshold_pct"] == 30 else margin_rows
        reference = next(item for item in source_rows
                         if item["dataset"] == row["dataset"] and int(item["seed"]) == row["seed"]
                         and item["policy"] == "offset30" and ("q" not in item or int(item["q"]) == 30))
        for metric, legacy in (("old_pct", "old"), ("new_pct", "current"), ("all_pct", "all")):
            assert abs(row[metric]-float(reference[legacy])) < 1e-10
        matched_test_rows += 1
    assert matched_test_rows == 40
    write_csv(output / "test_seed_values.csv", test_rows)
    pairs, summaries = [], []
    for dataset in ("cifar101", "cifar102"):
        for threshold in THRESHOLDS:
            selected = [row for row in test_rows if row["dataset"] == dataset and row["validation_threshold_pct"] == threshold]
            lookup = {(row["seed"], row["objective"]): row for row in selected}
            same_choices = same_models = 0
            violations_a_only = violations_b_only = 0
            for seed in SEEDS:
                a, b = lookup[seed, "new_accuracy"], lookup[seed, "overall_accuracy"]
                same_models += a["candidate"] == b["candidate"]
                same = a["candidate"] == b["candidate"] and a["scalar"] == b["scalar"]
                same_choices += same
                violations_a_only += a["below30_old"] and not b["below30_old"]
                violations_b_only += b["below30_old"] and not a["below30_old"]
                pairs.append({"dataset": dataset, "validation_threshold_pct": threshold, "seed": seed,
                              "same_model_and_scalar": same, "same_model": a["candidate"] == b["candidate"],
                              "new_accuracy_candidate": a["candidate"], "new_accuracy_scalar": a["scalar"],
                              "overall_accuracy_candidate": b["candidate"], "overall_accuracy_scalar": b["scalar"],
                              "new_difference_pp": a["new_pct"]-b["new_pct"],
                              "old_difference_pp": a["old_pct"]-b["old_pct"],
                              "all_difference_pp": a["all_pct"]-b["all_pct"],
                              "new_objective_below30": a["below30_old"],
                              "overall_objective_below30": b["below30_old"]})
            paired = [row for row in pairs if row["dataset"] == dataset and row["validation_threshold_pct"] == threshold]
            summaries.append({"dataset": dataset, "validation_threshold_pct": threshold,
                "common_available_seeds": list(SEEDS), "same_model_and_scalar": same_choices,
                "same_model": same_models, "new_objective_only_violations": violations_a_only,
                "overall_objective_only_violations": violations_b_only,
                "objectives": {objective: {"below30_count": sum(row["below30_old"] for row in selected if row["objective"] == objective),
                    **{metric: describe([row[metric] for row in selected if row["objective"] == objective]) for metric in ("old_pct", "new_pct", "all_pct")}} for objective in OBJECTIVES},
                "paired_new_minus_overall_objective": {metric: describe([row[metric] for row in paired]) for metric in ("new_difference_pp", "old_difference_pp", "all_difference_pp")}})
    write_csv(output / "paired_seed_values.csv", pairs)
    write_json(output / "summary.json", summaries)
    write_json(output / "input_manifest.json", inputs)
    metadata = {"started_utc": started_utc, "choices_fixed_utc": choices_fixed_utc,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter()-start, "python": platform.python_version(),
                "numpy": np.__version__, "validation_models": 300, "test_models": test_inputs,
                "validation_choices": len(choices), "validation_boundary_points": len(feasible),
                "full_argmax_boundary_checks": full_argmax_checks,
                "all_correctness_event_coverage_checks": coverage_checks,
                "legacy_new_objective_choices_exactly_matched": 20,
                "legacy_new_objective_test_rows_matched": matched_test_rows,
                "validation_choices_sha256": choice_hash, "script_sha256": sha256(Path(__file__)),
                "input_manifest_sha256": sha256(output / "input_manifest.json"),
                "statistical_unit": "paired seed", "interpretation": "supplementary descriptive comparison on existing candidate models and image sources",
                "training_performed": False, "new_data_or_annotation": False}
    write_json(output / "execution.json", metadata)
    print(json.dumps({"execution": metadata, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
