"""Identical original head, inherited remaining network, common four-epoch SGD.

All outputs are confined to this file's directory. Original sources and data are
read-only inputs. Run stages: prepare, pilot, run, summarize.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT
PROJECT = Path("E:/Projects/iclr 27")
OLD = Path("C:/Users/S0604/Documents/CodexResearch/iclr27_ijcnn2027")
DATA = PROJECT / "data"
HISTORY4 = PROJECT / "results/r9_followup"
HISTORY16 = OLD / "new_experiments/full_history16"
SEEDS = tuple(range(201, 206))
METHODS = ("sgd", "er")
TASKS = (1, 2, 3, 4)
HEAD_KEYS = ("head.weight", "head.bias")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(history, method, seed):
    if history == 4:
        return HISTORY4 / f"split_cifar10_classil_{method}_s{seed}"
    return HISTORY16 / f"{method}_s{seed}"


def load_engine():
    sys.path.insert(0, str(SCRIPT_ROOT / "source"))
    from r9_followup import engine, data
    from torchvision.datasets import CIFAR10

    # Retain the original stream construction, normalization, mapping and order;
    # require the existing files instead of allowing any network download.
    def offline_cifar(root, train, download):
        return CIFAR10(root=root, train=train, download=False)

    data.CIFAR10 = offline_cifar
    return engine


def setup():
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_default_dtype() == torch.float32
    assert torch.cuda.is_available(), "Original protocol requires CUDA"
    return torch.device("cuda:0")


def state(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def equal_states(a, b):
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def record(result, task_id):
    return next(r for r in result["tasks"] if r["task"] == task_id)


def at_four(result, task_id):
    return next(r["evaluation"] for r in record(result, task_id)["probe"]["trace"] if r["epoch"] == 4)


def prepare():
    import torch
    copied = {}
    dest = ROOT / "source/r9_followup"
    dest.mkdir(parents=True, exist_ok=True)
    for original in sorted((PROJECT / "experiments/r9_followup").glob("*.py")):
        target = dest / original.name
        if target.exists():
            assert target.read_bytes() == original.read_bytes()
        else:
            target.write_bytes(original.read_bytes())
        copied[original.name] = sha(target)
    engine = load_engine()
    inputs = []
    generated = {}
    stable_fields = ("benchmark", "seed", "init_seed", "order_seed", "class_order_seed", "tasks", "batch_size", "lr", "width", "max_train", "max_test", "threads", "device")
    reference_configs = {}
    reference_manifests = {}
    for seed, history, method in itertools.product(SEEDS, (4, 16), METHODS):
        directory = source(history, method, seed)
        result = read(directory / "result.json")
        cfg = result["config"]
        assert result["status"] == "complete", directory
        assert cfg["epochs"] == history and cfg["method"] == method
        assert cfg["seed"] == seed and cfg["benchmark"] == "split_cifar10_classil"
        assert copied == result["source_sha256"], (directory, "original source differs")
        stable = {k: cfg[k] for k in stable_fields}
        if seed in reference_configs:
            assert reference_configs[seed] == stable
        reference_configs[seed] = stable
        manifest = read(directory / "data_manifest.json")
        if seed in reference_manifests:
            assert reference_manifests[seed] == manifest
        reference_manifests[seed] = manifest
        initial = state(directory / "initial.pt")
        if seed not in generated:
            generated[seed] = engine.local_model(cfg, cfg["init_seed"], torch.device("cpu")).state_dict()
        assert equal_states(initial, generated[seed]), (directory, "initial differs")
        assert tuple(k for k in initial if k.startswith("head.")) == HEAD_KEYS
        assert initial["head.weight"].shape == (10, 256)
        assert all(v.dtype == torch.float32 for v in initial.values())
        details = []
        for tid in TASKS:
            path = directory / f"pre_task_{tid:03d}.pt"
            before = state(path)
            assert before.keys() == initial.keys()
            ref = record(result, tid)
            model = engine.local_model(cfg, cfg["init_seed"], torch.device("cpu"), before)
            assert engine.state_hash(model) == ref["pre_state_sha256"]
            assert at_four(result, tid)["n"] == 2000
            details.append({"task": tid, "pre_task": str(path), "pre_file_sha256": sha(path), "pre_state_sha256": ref["pre_state_sha256"], "baseline_epoch4": at_four(result, tid)})
        inputs.append({"history_epochs": history, "method": method, "seed": seed,
                       "directory": str(directory), "config": cfg, "initial_sha256": sha(directory / "initial.pt"),
                       "result_sha256": sha(directory / "result.json"), "data_manifest_sha256": sha(directory / "data_manifest.json"), "tasks": details})
    write(ROOT / "input_inventory.json", {"created_utc": now(), "source_sha256": copied, "inputs": inputs,
                                         "intervention_branches": 80, "initial_states_exact_original": 20,
                                         "pre_task_states_verified": 80, "offline_dataset": str(DATA / "r9_official/cifar-10-batches-py")})
    print(json.dumps({"stage": "prepare", "inputs": len(inputs), "pre_task_states": 80, "status": "complete"}), flush=True)


def make_tasks(engine, cfg):
    tasks, provenance = engine.make_stream(cfg, DATA)
    manifest = {"provenance": provenance, "tasks": [{k: v for k, v in t.items() if k not in ("train", "test")} for t in tasks]}
    expected = read(source(4, "sgd", cfg["seed"]) / "data_manifest.json")
    assert manifest == expected, "Original input tensors or sample mapping differ"
    return tasks


def save_logits(engine, model, task, cfg, device, output):
    import torch
    model.eval()
    z = []
    x, y = task["test"]
    with torch.no_grad():
        for start in range(0, len(y), 512):
            z.append(engine.selected_logits(model, x[start:start + 512].to(device), task["id"], cfg["benchmark"]).cpu())
    logits = torch.cat(z)
    assert logits.dtype == torch.float32 and logits.shape == (2000, 10)
    correct = int((logits.argmax(1) == y).sum())
    torch.save({"logits": logits, "labels": y, "official_test_indices": torch.tensor(task["splits"]["test"]["official_indices"], dtype=torch.long),
                "seed": cfg["seed"], "task": task["id"], "original_classes": task["original_classes"]}, output)
    return correct


def pilot():
    import torch
    assert (ROOT / "input_inventory.json").exists()
    engine = load_engine()
    device = setup()
    out = ROOT / "pilot"
    out.mkdir(exist_ok=True)
    seed, tid = 201, 1
    tasks = make_tasks(engine, read(source(4, "sgd", seed) / "result.json")["config"])
    checks = []
    for method in METHODS:
        original = source(4, method, seed)
        result = read(original / "result.json")
        cfg = {**result["config"], "epochs": 4}
        model = engine.local_model(cfg, cfg["init_seed"], device, state(original / f"pre_task_{tid:03d}.pt"))
        log = engine.train_branch(model, tasks[tid], cfg, device)
        expected = record(result, tid)["probe"]
        comparisons = {"trace_exact": log["trace"] == expected["trace"],
                       "counts_exact": log["counts"] == expected["counts"],
                       "batch_order_exact": log["batch_order_sha256"] == expected["batch_order_sha256"],
                       "final_state_exact": equal_states(model.state_dict(), state(original / f"probe_task_{tid:03d}.pt")),
                       "final_state_digest_exact": log["final_state_sha256"] == expected["final_state_sha256"]}
        correct = save_logits(engine, model, tasks[tid], cfg, device, out / f"{method}_s{seed}_task{tid}_logits.pt")
        checks.append({"method": method, "seed": seed, "task": tid, "history_epochs": 4,
                       "expected_correct": at_four(result, tid)["correct"], "reproduced_correct": correct, "n": 2000,
                       "seconds": log["seconds"], "comparisons": comparisons, "log": log})
        write(out / "comparison.json", {"created_utc": now(), "checks": checks, "status": "complete" if len(checks) == 2 and all(all(c["comparisons"].values()) for c in checks) else "in_progress"})
        assert all(comparisons.values()), comparisons
        print(json.dumps({"stage": "pilot", "method": method, "correct": correct, "exact": True, "seconds": log["seconds"]}), flush=True)


def run():
    import numpy as np
    import torch
    assert read(ROOT / "pilot/comparison.json")["status"] == "complete"
    inventory = read(ROOT / "input_inventory.json")
    engine = load_engine()
    device = setup()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(ROOT / "run_started.json", {"created_utc": now(), "pid": os.getpid(),
                                      "runner_sha256": sha(__file__), "python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "device": torch.cuda.get_device_name(), "sequential": True})
    rows = []
    for seed in SEEDS:
        cfg0 = read(source(4, "sgd", seed) / "result.json")["config"]
        tasks = make_tasks(engine, cfg0)
        for history, method in itertools.product((4, 16), METHODS):
            original = source(history, method, seed)
            result = read(original / "result.json")
            cfg = {**result["config"], "epochs": 4}
            initial = state(original / "initial.pt")
            for tid in TASKS:
                if time.perf_counter() - start > 55 * 60:
                    raise TimeoutError("55-minute execution bound reached")
                target = ROOT / "branches" / f"h{history}_{method}_s{seed}_t{tid}"
                target.mkdir(parents=True, exist_ok=True)
                if (target / "result.json").exists():
                    saved = read(target / "result.json")
                    rows.append(saved)
                    continue
                before = state(original / f"pre_task_{tid:03d}.pt")
                model = engine.local_model(cfg, cfg["init_seed"], device, before)
                model.head.load_state_dict({k.removeprefix("head."): initial[k] for k in HEAD_KEYS})
                current = model.state_dict()
                assert all(torch.equal(current[k].cpu(), initial[k]) for k in HEAD_KEYS)
                assert all(torch.equal(current[k].cpu(), v) for k, v in before.items() if k not in HEAD_KEYS)
                assert all(p.requires_grad for p in model.parameters())
                before_correct = save_logits(engine, model, tasks[tid], cfg, device, target / "initial_head_logits.pt")
                intervened_start_digest = engine.state_hash(model)
                log = engine.train_branch(model, tasks[tid], cfg, device)
                assert log["batch_order_sha256"] == record(result, tid)["probe"]["batch_order_sha256"][:4]
                assert log["counts"] == {"optimizer_steps": 628, "current_examples": 40000, "replay_examples": 0, "forward_examples": 40000}
                assert log["operator"]["method"] == "sgd" and not log["operator"]["er_current_weight_only"]
                assert log["operator"]["core_current_factor_min"] == log["operator"]["core_current_factor_max"] == 1
                final = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(final, target / "final.pt")
                correct = save_logits(engine, model, tasks[tid], cfg, device, target / "final_logits.pt")
                assert correct == log["trace"][-1]["evaluation"]["correct"]
                saved = {"history_epochs": history, "method": method, "seed": seed, "task": tid,
                         "original_directory": str(original), "baseline": at_four(result, tid),
                         "intervention": log["trace"][-1]["evaluation"], "initial_head_correct": before_correct,
                         "non_head_unchanged_initially": True, "whole_original_head_exact": True, "all_parameters_trainable": True,
                         "initial_state_sha256": intervened_start_digest, "log": log}
                if history == 4:
                    comparison = engine.local_model(cfg, cfg["init_seed"], device, state(original / f"probe_task_{tid:03d}.pt"))
                    original_correct = save_logits(engine, comparison, tasks[tid], cfg, device, target / "original_head_logits.pt")
                    assert original_correct == saved["baseline"]["correct"]
                    del comparison
                write(target / "result.json", saved)
                rows.append(saved)
                print(json.dumps({"stage": "run", "branch": target.name, "completed": len(rows), "correct": correct,
                                  "n": 2000, "seconds": round(log["seconds"], 3)}), flush=True)
                del model
        del tasks
    assert len(rows) == 80
    write(ROOT / "execution.json", {"finished_utc": now(), "branches": 80, "seconds": time.perf_counter() - start,
                                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                                    "status": "complete", "sequential": True})
    summarize()


def describe(values):
    return {"n_seeds": len(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def exact_two_sided(values):
    observed = abs(statistics.mean(values))
    permutations = [abs(sum(s * v for s, v in zip(signs, values)) / len(values)) for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(v >= observed - 1e-12 for v in permutations) / len(permutations)


def summarize():
    raw, lookup = [], {}
    for path in sorted((ROOT / "branches").glob("*/result.json")):
        r = read(path)
        key = r["history_epochs"], r["method"], r["seed"], r["task"]
        assert key not in lookup
        lookup[key] = r
        raw.append({"history_epochs": key[0], "method": key[1], "seed": key[2], "task": key[3],
                    "n": r["baseline"]["n"], "original_correct": r["baseline"]["correct"],
                    "initial_head_correct": r["intervention"]["correct"],
                    "original_accuracy_pct": r["baseline"]["accuracy_pct"],
                    "initial_head_accuracy_pct": r["intervention"]["accuracy_pct"],
                    "intervention_minus_original_pp": r["intervention"]["accuracy_pct"] - r["baseline"]["accuracy_pct"]})
    assert len(raw) == 80
    write_csv(ROOT / "raw_counts.csv", raw)
    seed_rows, paired_tasks = [], []
    for history, seed in itertools.product((4, 16), SEEDS):
        row = {"history_epochs": history, "seed": seed, "n_per_method": 8000}
        for method, condition in itertools.product(METHODS, ("original", "initial_head")):
            name = "baseline" if condition == "original" else "intervention"
            correct = sum(lookup[history, method, seed, tid][name]["correct"] for tid in TASKS)
            row[f"{method}_{condition}_correct"] = correct
            row[f"{method}_{condition}_accuracy_pct"] = 100 * correct / 8000
        row["original_er_minus_sgd_pp"] = row["er_original_accuracy_pct"] - row["sgd_original_accuracy_pct"]
        row["initial_head_er_minus_sgd_pp"] = row["er_initial_head_accuracy_pct"] - row["sgd_initial_head_accuracy_pct"]
        row["difference_in_differences_pp"] = row["initial_head_er_minus_sgd_pp"] - row["original_er_minus_sgd_pp"]
        for method in METHODS:
            row[f"{method}_intervention_minus_original_pp"] = row[f"{method}_initial_head_accuracy_pct"] - row[f"{method}_original_accuracy_pct"]
        seed_rows.append(row)
        for tid in TASKS:
            br = {"history_epochs": history, "seed": seed, "task": tid, "n_per_method": 2000}
            for condition in ("original", "initial_head"):
                name = "baseline" if condition == "original" else "intervention"
                br[f"{condition}_er_minus_sgd_correct"] = lookup[history, "er", seed, tid][name]["correct"] - lookup[history, "sgd", seed, tid][name]["correct"]
                br[f"{condition}_er_minus_sgd_pp"] = br[f"{condition}_er_minus_sgd_correct"] / 20
            br["difference_in_differences_pp"] = br["initial_head_er_minus_sgd_pp"] - br["original_er_minus_sgd_pp"]
            paired_tasks.append(br)
    write_csv(ROOT / "seed_paired.csv", seed_rows)
    write_csv(ROOT / "task_paired.csv", paired_tasks)
    summaries = {}
    for history in (4, 16):
        rs = [r for r in seed_rows if r["history_epochs"] == history]
        stats = {}
        for key in rs[0]:
            if key.endswith("_pp") or key.endswith("_accuracy_pct"):
                values = [r[key] for r in rs]
                stats[key] = describe(values)
                if key in ("initial_head_er_minus_sgd_pp", "difference_in_differences_pp"):
                    stats[key]["descriptive_exact_sign_permutation_p_two_sided"] = exact_two_sided(values)
                if key.endswith("er_minus_sgd_pp"):
                    stats[key]["positive_seeds"] = sum(v > 0 for v in values)
        stats["by_task"] = {str(tid): {key: describe([r[key] for r in paired_tasks if r["history_epochs"] == history and r["task"] == tid])
                                            for key in ("original_er_minus_sgd_pp", "initial_head_er_minus_sgd_pp", "difference_in_differences_pp")}
                            for tid in TASKS}
        summaries[str(history)] = stats
    write(ROOT / "summary.json", {"created_utc": now(), "unit": "percentage points unless accuracy_pct", "independent_unit": "seed", "n_seeds": 5,
                                 "tasks_averaged": list(TASKS), "difference_in_differences_definition": "(ER-SGD with original initial head) minus (ER-SGD with inherited head)",
                                 "histories": summaries, "execution": read(ROOT / "execution.json")})
    print(json.dumps({"stage": "summarize", "histories": {h: {k: v for k, v in stats.items() if k in ("original_er_minus_sgd_pp", "initial_head_er_minus_sgd_pp", "difference_in_differences_pp")} for h, stats in summaries.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "pilot", "run", "summarize"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT,
                        help="Author-supplied original project containing experiments/r9_followup")
    parser.add_argument("--history4-root", type=Path,
                        help="Directory containing split_cifar10_classil_{sgd,er}_s{seed} checkpoints")
    parser.add_argument("--history16-root", type=Path, default=HISTORY16,
                        help="Directory containing {sgd,er}_s{seed} sixteen-epoch checkpoints")
    parser.add_argument("--data-root", type=Path,
                        help="Existing data directory containing r9_official/cifar-10-batches-py")
    parser.add_argument("--output-root",type=Path,required=True)
    arguments = parser.parse_args()
    ROOT = arguments.output_root.resolve()
    ROOT.mkdir(parents=True,exist_ok=True)
    PROJECT = arguments.project_root.resolve()
    HISTORY4 = (arguments.history4_root or PROJECT / "results/r9_followup").resolve()
    HISTORY16 = arguments.history16_root.resolve()
    DATA = (arguments.data_root or PROJECT / "data").resolve()
    globals()[arguments.stage]()
