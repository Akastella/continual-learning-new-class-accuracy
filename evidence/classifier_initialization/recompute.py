"""Recompute classifier-state results from compact arrays; requires NumPy only."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_file(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    if any(value is None for value in values):
        return None
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}


def pct(correct, n):
    return None if correct is None else 100 * correct / n


def paired_stats(values):
    result = stats(values)
    if result is None:
        return None
    observed = abs(float(np.mean(values)))
    permuted = [abs(float(np.mean(np.asarray(signs) * values))) for signs in itertools.product((-1, 1), repeat=len(values))]
    result["descriptive_exact_sign_permutation_p_two_sided"] = sum(x >= observed - 1e-12 for x in permuted) / len(permuted)
    return result


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(HERE / "predictions.npz", allow_pickle=False)
    rows = []
    for meta in manifest["endpoints"]:
        row = {k: meta[k] for k in ("id", "history_epochs", "method", "seed", "task", "condition", "coverage")}
        tid = meta["task"]
        if meta["coverage"] == "all_seen_predictions":
            prefix = meta["id"]
            prediction, label, index, eval_task = [data[prefix + "_" + k] for k in ("prediction", "label", "index", "eval_task")]
            assert prediction.shape == label.shape == index.shape == eval_task.shape == (2000 * (tid + 1),)
            assert prediction.dtype == label.dtype == eval_task.dtype == np.uint8 and index.dtype == np.uint16
            assert len(np.unique(index)) == len(index) and np.all(prediction < 10) and np.all(label < 10)
            assert np.all(label // 2 == eval_task)
            current, old = eval_task == tid, eval_task < tid
            assert int(current.sum()) == 2000 and int(old.sum()) == 2000 * tid
            current_correct = int(np.count_nonzero(prediction[current] == label[current]))
            old_correct = int(np.count_nonzero(prediction[old] == label[old]))
            assert current_correct == meta["current_correct"] and old_correct == meta["old_correct"]
        else:
            current_correct, old_correct = meta["current_correct"], None
        all_correct = None if old_correct is None else old_correct + current_correct
        for name, correct, n in (("old", old_correct, 2000 * tid), ("current", current_correct, 2000), ("all_seen", all_correct, 2000 * (tid + 1))):
            row[name + "_correct"] = correct
            row[name + "_n"] = n
            row[name + "_accuracy_pct"] = pct(correct, n)
        rows.append(row)
    assert len(rows) == 160
    csv_file(HERE / "recomputed_endpoint_counts.csv", rows)
    seed_rows = []
    for history, condition, method, seed in itertools.product((4, 16), ("original", "common"), ("sgd", "er"), manifest["seed_order"]):
        selected = [r for r in rows if (r["history_epochs"], r["condition"], r["method"], r["seed"]) == (history, condition, method, seed)]
        assert len(selected) == 4
        seed_row = {"history_epochs": history, "condition": condition, "method": method, "seed": seed}
        for metric in ("old", "current", "all_seen"):
            accuracies = [r[metric + "_accuracy_pct"] for r in selected]
            seed_row[metric + "_task_mean_accuracy_pct"] = None if any(x is None for x in accuracies) else float(np.mean(accuracies))
            counts = [r[metric + "_correct"] for r in selected]
            correct = None if any(x is None for x in counts) else sum(counts)
            n = sum(r[metric + "_n"] for r in selected)
            seed_row[metric + "_summed_model_prediction_correct"] = correct
            seed_row[metric + "_summed_model_prediction_n"] = n
            seed_row[metric + "_pooled_model_predictions_accuracy_pct"] = pct(correct, n)
            final = next(r for r in selected if r["task"] == 4)
            seed_row[metric + "_task4_accuracy_pct"] = final[metric + "_accuracy_pct"]
        seed_rows.append(seed_row)
    csv_file(HERE / "recomputed_seed_scores.csv", seed_rows)
    summary = {}
    paired = []
    for history in (4, 16):
        block = {}
        for condition in ("original", "common"):
            methods = {}
            for method in ("sgd", "er"):
                selected = [r for r in seed_rows if (r["history_epochs"], r["condition"], r["method"]) == (history, condition, method)]
                methods[method] = {key: stats([r[key] for r in selected]) for key in selected[0] if key.endswith("accuracy_pct")}
            block[condition] = methods
            method_differences = {}
            for metric in ("old", "current", "all_seen"):
                key = metric + "_task_mean_accuracy_pct"
                differences = []
                for seed in manifest["seed_order"]:
                    a, b = [next(r for r in seed_rows if (r["history_epochs"], r["condition"], r["method"], r["seed"]) == (history, condition, method, seed))[key] for method in ("er", "sgd")]
                    value = None if a is None or b is None else a - b
                    differences.append(value)
                    paired.append({"history_epochs": history, "condition": condition, "metric": metric, "seed": seed, "er_minus_sgd_pp": value})
                method_differences[metric] = paired_stats(differences)
            block[condition]["paired_er_minus_sgd_pp"] = method_differences
        block["difference_in_differences_pp"] = {}
        for metric in ("old", "current", "all_seen"):
            a, b = [block[condition]["paired_er_minus_sgd_pp"][metric] for condition in ("common", "original")]
            block["difference_in_differences_pp"][metric] = None if a is None or b is None else paired_stats([x - y for x, y in zip(a["values"], b["values"])])
        summary[str(history)] = block
    csv_file(HERE / "recomputed_paired_differences.csv", paired)
    save(HERE / "recomputed_summary.json", {"n_seeds": 5, "aggregation": "Mean of four task-level accuracies within seed, followed by mean and sample SD over seeds. Pooled model-prediction counts are separately labelled.", "histories": summary})
    print(json.dumps({"endpoints": len(rows), "array_endpoints": sum(r["coverage"] == "all_seen_predictions" for r in rows), "original16_old_scores_missing": 40,
                      "common_old_accuracy_pct": {h: {m: summary[h]["common"][m]["old_task_mean_accuracy_pct"] for m in ("sgd", "er")} for h in ("4", "16")}}, indent=2))


if __name__ == "__main__":
    main()
