"""Independent R13 evidence audit; no production scoring/selection imports.

CPU only. Torch is imported lazily solely to deserialize checkpoint tensors.
This checks evidence consistency and lineage, not independent training or forward
inference. Exact rational effects precede conversion to display means and SDs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = "configs/r13_early_retention/manifest.json"
SPARSE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 157, 314, 471, 628]
DENSE = sorted(set(SPARSE) | set(range(17)))
MATCHED = [0, 1, 2, 4, 8, 16]
POLICIES = ["sparse_probe", "dense_probe", "sparse_plus_actual", "dense_plus_actual"]
CONSTRAINTS = ["q20", "q20_rel5"]
METHODS = ["sgd", "er"]
SEEDS = list(range(201, 211))
RUN_IDS = [f"split_cifar10_classil_{m}_s{s}" for s in SEEDS for m in METHODS]
ANCHORS = {"step_0", "step_628", "actual"}
METRICS = ["old", "current", "all"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe(base, name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "nonportable evidence path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and str(relative) == name, "unsafe evidence path: " + name)
    path = (Path(base) / name).resolve()
    require(path.is_relative_to(Path(base).resolve()), "path escapes evidence root")
    return path


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamp lacks timezone")
    return result


def check_hashes(base, mapping):
    require(isinstance(mapping, dict) and bool(mapping), "empty hash inventory")
    for name, expected in mapping.items():
        require(sha(safe(base, name)) == expected, "hash mismatch: " + name)


def exact(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator,
            "value": float(value)}


def describe(values):
    values = [Fraction(v) for v in values]
    n = len(values)
    mean = sum(values, Fraction()) / n if n else None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else None
    return {"n": n, "mean": exact(mean) if n else None,
            "sample_variance": exact(variance) if variance is not None else None,
            "sample_sd": math.sqrt(float(variance)) if variance is not None else None,
            "positive": sum(v > 0 for v in values), "zero": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values)}


def scores(logits, labels):
    labels, logits = np.asarray(labels), np.asarray(logits)
    require(labels.ndim == 1 and labels.dtype == np.int64 and len(labels) > 0,
            "invalid mapped labels")
    require(np.all((labels >= 0) & (labels < 10)), "label outside all-ten outputs")
    require(logits.shape == (len(labels), 10) and logits.dtype == np.float32
            and np.isfinite(logits).all(), "invalid/nonfinite logits")
    counts = np.bincount(labels, minlength=10)
    require(np.all(counts > 0), "missing test/validation class")
    correct = np.bincount(labels[np.argmax(logits, axis=1) == labels], minlength=10)
    cn, cc = counts.tolist(), correct.tolist()
    old_n, current_n = sum(cn[:8]), sum(cn[8:])
    old_c, current_c = sum(cc[:8]), sum(cc[8:])
    return {"class_correct": cc, "class_n": cn, "old_correct": old_c,
            "old_n": old_n, "current_correct": current_c, "current_n": current_n,
            "old_pct": 100 * old_c / old_n, "current_pct": 100 * current_c / current_n,
            "all_pct": 100 * sum(cc) / sum(cn),
            "old_task_pct": [100 * sum(cc[j:j + 2]) / sum(cn[j:j + 2])
                             for j in (0, 2, 4, 6)]}


def metric(score, name):
    if name == "all":
        return Fraction(100 * sum(score["class_correct"]), sum(score["class_n"]))
    return Fraction(100 * score[name + "_correct"], score[name + "_n"])


def feasible(score, pre, constraint):
    require(constraint in CONSTRAINTS, "unknown constraint")
    absolute = 100 * score["old_correct"] >= 20 * score["old_n"]
    # Cross-products support differing denominators without a float tolerance.
    relative = (100 * (pre["old_correct"] * score["old_n"]
                       - score["old_correct"] * pre["old_n"])
                <= 5 * pre["old_n"] * score["old_n"])
    return absolute and (constraint == "q20" or relative)


def select(records, policy, constraint):
    require(policy in POLICIES, "unknown policy")
    pre = records["step_0"]["scores"]
    steps = DENSE if policy.startswith("dense") else SPARSE
    keys = [f"step_{step}" for step in steps]
    if policy.endswith("plus_actual"):
        keys.append("actual")
    best, rank = None, None
    for key in keys:
        row = records[key]
        require(row["scores"]["current_n"] == pre["current_n"],
                "candidate current denominators differ")
        if not feasible(row["scores"], pre, constraint):
            continue
        order = (-row["scores"]["current_correct"], row["step"], int(key == "actual"))
        if rank is None or order < rank:
            best, rank = key, order
    return best


def violations(score, pre, constraint):
    absolute = metric(score, "old") < 20
    relative = metric(pre, "old") - metric(score, "old") > 5
    return {"absolute_below_20": absolute, "relative_drop_above_5": relative,
            "constraint_violation": absolute or (constraint == "q20_rel5" and relative),
            "old_change_from_pre_pp": exact(metric(score, "old") - metric(pre, "old"))}


def state_arrays(state):
    require(isinstance(state, dict) and bool(state), "empty checkpoint model")
    result = {}
    for name, value in state.items():
        array = value.detach().cpu().numpy()
        require(array.dtype == np.float32 and np.isfinite(array).all(),
                "nonfinite/non-FP32 model tensor: " + name)
        result[name] = array
    return result


def state_digest(state):
    h = hashlib.sha256()
    for name, array in state_arrays(state).items():
        h.update(name.encode())
        h.update(hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest())
    return h.hexdigest()


def equal_states(left, right):
    a, b = state_arrays(left), state_arrays(right)
    require(list(a) == list(b), "checkpoint tensor keys/order differ")
    for name in a:
        require(a[name].shape == b[name].shape and np.array_equal(a[name], b[name]),
                "checkpoint tensor differs: " + name)


def check_replay_identity(new, old):
    require(new["candidates"] == DENSE and old["candidates"] == SPARSE,
            "replay checkpoint candidate schedule mismatch")
    require({k: v for k, v in new.items() if k != "candidates"} ==
            {k: v for k, v in old.items() if k != "candidates"},
            "replay training identity changed")


def records_by_key(rows, wanted):
    result = {}
    for row in rows:
        key = row["key"]
        require(key not in result, "duplicate endpoint: " + key)
        require(key in wanted, "unexpected endpoint: " + key)
        step = 628 if key == "actual" else int(key.removeprefix("step_"))
        require(row["step"] == step and row["kind"] == ("actual" if key == "actual" else "probe"),
                "endpoint step/kind mismatch: " + key)
        result[key] = row
    require(set(result) == set(wanted), "missing endpoint records")
    return result


def check_npz(path, records, indices, labels):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"indices", "labels"} | {"logits_" + k for k in records},
                "unexpected raw array inventory")
        require(data["indices"].dtype == np.int64 and np.array_equal(data["indices"], indices),
                "raw indices differ from fixed dataset/split")
        require(data["labels"].dtype == np.int64 and np.array_equal(data["labels"], labels),
                "raw mapped labels differ from source")
        rebuilt = {}
        for key, row in records.items():
            calculated = scores(data["logits_" + key], labels)
            require(calculated == row["scores"], "reported counts/scores differ: " + key)
            rebuilt[key] = dict(row, scores=calculated)
    return rebuilt


def summarize(all_runs):
    """Descriptive endpoints and paired effects; no test-based deletion."""
    selections, anchors, groups, comparisons, method_pairs = [], [], [], [], []
    lookup = {}
    for run in all_runs:
        test, val = run["test"], run["validation"]
        pre = test["step_0"]["scores"]
        for key in sorted(ANCHORS):
            anchors.append({"run_id": run["run_id"], "seed": run["seed"],
                            "method": run["method"], "key": key, "scores": test[key]["scores"]})
        for constraint in CONSTRAINTS:
            for policy in POLICIES:
                key = run["selection"][constraint][policy]
                item = {"run_id": run["run_id"], "seed": run["seed"], "method": run["method"],
                        "constraint": constraint, "policy": policy, "key": key,
                        "validation": None, "test": None, "violations": None,
                        "actual_minus_selected_current_pp": None,
                        "test_pre": pre, "test_actual": test["actual"]["scores"]}
                if key is not None:
                    score = test[key]["scores"]
                    item.update(validation=val[key]["scores"], test=score,
                                violations=violations(score, pre, constraint),
                                actual_minus_selected_current_pp=exact(
                                    metric(test["actual"]["scores"], "current") - metric(score, "current")))
                selections.append(item)
                lookup[run["method"], run["seed"], constraint, policy] = item
    for method, constraint, policy in itertools.product(METHODS, CONSTRAINTS, POLICIES):
        rows = [lookup[method, seed, constraint, policy] for seed in SEEDS]
        available = [row for row in rows if row["test"] is not None]
        groups.append({"method": method, "constraint": constraint, "policy": policy,
                       "available_seeds": [x["seed"] for x in available],
                       "unavailable_seeds": [x["seed"] for x in rows if x["test"] is None],
                       "metrics": {m: describe(metric(x["test"], m) for x in available) for m in METRICS},
                       "violations": {name: sum(x["violations"][name] for x in available)
                                      for name in ("absolute_below_20", "relative_drop_above_5", "constraint_violation")}})
    for method, constraint in itertools.product(METHODS, CONSTRAINTS):
        for left, right in itertools.combinations(POLICIES, 2):
            pairs = [(lookup[method, s, constraint, left], lookup[method, s, constraint, right]) for s in SEEDS]
            comparisons.append(paired_summary(pairs, {"method": method, "constraint": constraint,
                                                       "contrast": right + " minus " + left}))
    for constraint, policy in itertools.product(CONSTRAINTS, POLICIES):
        pairs = [(lookup["sgd", s, constraint, policy], lookup["er", s, constraint, policy]) for s in SEEDS]
        method_pairs.append(paired_summary(pairs, {"constraint": constraint, "policy": policy,
                                                   "contrast": "er minus sgd", "conditional_on_joint_validation_feasibility": True}))
    return {"selections": selections, "anchors": anchors, "groups": groups,
            "paired_policy_effects": comparisons, "conditional_er_minus_sgd": method_pairs}


def paired_summary(pairs, context):
    common = [(a, b) for a, b in pairs if a["test"] is not None and b["test"] is not None]
    rows = [{"seed": a["seed"], "left_key": a["key"], "right_key": b["key"],
             "effects_pp": {m: exact(metric(b["test"], m) - metric(a["test"], m)) for m in METRICS},
             "left_pre": a["test_pre"], "right_pre": b["test_pre"],
             "left_actual": a["test_actual"], "right_actual": b["test_actual"]} for a, b in common]
    return dict(context, common_seeds=[a["seed"] for a, _ in common], rows=rows,
                availability=[{"seed": a["seed"], "left": a["key"] is not None,
                               "right": b["key"] is not None, "left_key": a["key"], "right_key": b["key"]} for a, b in pairs],
                metrics={m: describe(metric(b["test"], m) - metric(a["test"], m) for a, b in common) for m in METRICS})


def audit(root=ROOT):
    import torch  # serialization only; no model construction or GPU use

    root = Path(root).resolve()
    manifest = read(root / CANONICAL)
    settings = manifest["settings"]
    for key, value in {"seeds": SEEDS, "methods": METHODS, "task": 4,
                       "sparse_steps": SPARSE, "dense_steps": DENSE, "policies": POLICIES,
                       "constraints": CONSTRAINTS, "limit_seconds": 3300,
                       "expected_optimizer_updates": 320}.items():
        require(settings[key] == value, "protocol setting drift: " + key)
    directory = safe(root, settings["output"])
    require(sha(directory / "frozen_manifest.json") == sha(root / CANONICAL), "manifest mismatch")
    for group in ("input_sha256", "source_sha256"):
        check_hashes(root, manifest[group])
    require(sha(safe(root, settings["history_manifest"])) ==
            "bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e",
            "original history manifest changed")
    require(sha(safe(root, settings["trajectory_manifest"])) ==
            "60ef2ac448e9e77c3f5285510b3a0ba937c198541f77aa73a6a3f74491261aa0",
            "original trajectory manifest changed")
    history = read(safe(root, settings["history_manifest"]))
    run_rows = [r for r in history["runs"] if r["id"].startswith("split_cifar10_classil_")]
    runs = {r["id"]: r for r in run_rows}
    require(len(run_rows) == len(runs) == 20 and set(runs) == set(RUN_IDS), "original history matrix mismatch")
    old_manifest = read(safe(root, settings["trajectory_manifest"]))
    for name, value in old_manifest["input_sha256"].items():
        if name in manifest["input_sha256"]:
            require(manifest["input_sha256"][name] == value, "old frozen input drift: " + name)
    require({p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith("split_cifar10_")} == set(RUN_IDS),
            "raw history directory matrix mismatch")
    essential = {settings[k] for k in ("validation_split", "validation_data", "validation_labels",
                                       "test_npz", "history_manifest", "trajectory_manifest")}
    essential.add("third_party/r13_cifar102_reference/SOURCE.json")
    trajectory_root = safe(root, settings["trajectory_root"])
    for run in runs.values():
        essential.add(run["config_path"])
        essential.update(run["output"] + "/" + n for n in ("pre_task_004.pt", "post_task_004.pt", "probe_task_004.pt"))
        essential.update(settings["trajectory_root"] + "/" + run["id"] + "/" + n
                         for n in ("trajectory.json", "validation.npz"))
        essential.update(settings["trajectory_root"] + "/" + run["id"] + f"/checkpoints/step_{step:06d}.pt" for step in SPARSE)
    essential.update(f"data/r9_official/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6))
    require(essential <= set(manifest["input_sha256"]), "manifest omits required input lineage")
    for name, value in history["source_sha256"].items():
        require(manifest["source_sha256"].get(name) == value, "old core source drift: " + name)
    source_dir = directory / "source"
    require(source_dir.is_dir(), "missing frozen source snapshot")
    check_hashes(source_dir, manifest["source_sha256"])

    complete, watchdog = read(directory / "complete.json"), read(directory / "watchdog.json")
    inventory = read(directory / "artifacts.json")
    require(complete["artifacts_sha256"] == sha(directory / "artifacts.json"), "artifact inventory lineage")
    excluded = {"console.log", "watchdog.json", "artifacts.json", "complete.json"}
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(set(inventory) == present - excluded, "raw artifact inventory is incomplete or stale")
    check_hashes(directory, inventory)
    require(complete["status"] == watchdog["status"] == "complete" and watchdog["exit_code"] == 0,
            "campaign did not complete")
    require(watchdog["limit_seconds"] == 3300 and 0 < watchdog["elapsed_seconds"] <= 3300,
            "watchdog budget exceeded")
    require(complete["manifest_sha256"] == sha(root / CANONICAL), "completion manifest lineage")
    seal, started = read(directory / "selection_seal.json"), read(directory / "test_started.json")
    run_start = read(directory / "started.json")
    require(run_start["manifest_sha256"] == sha(root / CANONICAL), "run start manifest lineage")
    chronology = [instant(manifest["frozen_utc"]), instant(run_start["utc"]),
                  instant(seal["created_utc"]), instant(started["utc"]), instant(complete["finished_utc"])]
    require(chronology == sorted(chronology) and chronology[2] < chronology[3], "selection/test chronology invalid")
    require((chronology[-1] - chronology[1]).total_seconds() <= watchdog["elapsed_seconds"] + 1,
            "watchdog and recorded timestamps disagree")
    seal_sha = sha(directory / "selection_seal.json")
    require(started["selection_seal_sha256"] == seal_sha, "test start seal differs")
    require(seal["selection_sha256"] == sha(directory / "selection.json"), "selection changed after seal")
    val_files = {rid + "/" + name for rid in RUN_IDS for name in ("validation.json", "validation.npz", "training.json")}
    require(set(seal["validation_files_sha256"]) == val_files, "validation seal coverage")
    check_hashes(directory, seal["validation_files_sha256"])
    choices = read(directory / "selection.json")
    require(set(choices) == set(RUN_IDS), "selection run matrix mismatch")

    original_labels = np.load(safe(root, settings["validation_labels"]), allow_pickle=False)
    require(original_labels.shape == (2000,), "CIFAR-10.1 label size")
    split = read(safe(root, settings["validation_split"]))
    require(split["labels_sha256"] == sha(safe(root, settings["validation_labels"])), "validation label provenance")
    expected_val, expected_test = [], []
    for c in range(10):
        idx = np.flatnonzero(original_labels == c)
        require(len(idx) == 200, "CIFAR-10.1 class count")
        shuffled = np.random.Generator(np.random.PCG64(20260915 + c)).permutation(idx)
        expected_val.extend(shuffled[:100].tolist())
        expected_test.extend(shuffled[100:].tolist())
    val_indices = np.array(sorted(expected_val), dtype=np.int64)
    require(split["validation_indices"] == val_indices.tolist()
            and split["test_indices"] == sorted(expected_test), "fixed validation split mismatch")
    require(instant(split["created_utc"]) < chronology[0], "validation split not frozen before R13")
    source = read(root / "third_party/r13_cifar102_reference/SOURCE.json")
    entry = next(x for x in source["downloaded"] if x["name"] == "cifar102_test.npz")
    require(entry["path"] == settings["test_npz"] and entry["sha256"] == sha(safe(root, settings["test_npz"])),
            "CIFAR-10.2 source digest mismatch")
    with np.load(safe(root, settings["test_npz"]), allow_pickle=False) as external:
        raw_test_labels = external[settings["test_labels_key"]]
        require(np.issubdtype(raw_test_labels.dtype, np.integer), "noninteger source test labels")
        test_labels = raw_test_labels.astype(np.int64)
        require(test_labels.ndim == 1 and len(test_labels) > 0
                and np.all((test_labels >= 0) & (test_labels < 10)), "invalid CIFAR-10.2 labels")
        pixels = external[settings["test_images_key"]]
        require(pixels.shape == (len(test_labels), 32, 32, 3) and pixels.dtype == np.uint8,
                "CIFAR-10.2 image shape/type")
    test_indices = np.arange(len(test_labels), dtype=np.int64)
    all_runs, total_test, total_updates = [], 0, 0
    for rid in RUN_IDS:
        run = runs[rid]
        require(sha(safe(root, run["config_path"])) == run["config_sha256"], "frozen history config drift")
        config = read(safe(root, run["config_path"]))
        seed, method = config["seed"], config["method"]
        require(rid == f"split_cifar10_classil_{method}_s{seed}", "run/config identity mismatch")
        require(config["class_order_seed"] == seed + 40000, "class-order seed mismatch")
        order = np.random.Generator(np.random.PCG64(seed + 40000)).permutation(10)
        inverse = np.argsort(order)
        rd, old_dir = directory / rid, trajectory_root / rid
        old = read(old_dir / "trajectory.json")
        require(old["validation_npz_sha256"] == sha(old_dir / "validation.npz"), "old validation archive lineage")
        require(old["train_provenance"]["class_order_original_ids"] == order.tolist(), "old class mapping mismatch")
        wanted = {f"step_{s}" for s in DENSE} | {"actual"}
        val = read(rd / "validation.json")
        require(val["run_id"] == rid and val["npz_sha256"] == sha(rd / "validation.npz"), "validation lineage mismatch")
        records = records_by_key(val["records"], wanted)
        records = check_npz(rd / "validation.npz", records, val_indices, inverse[original_labels[val_indices]].astype(np.int64))
        with np.load(old_dir / "validation.npz", allow_pickle=False) as before, np.load(rd / "validation.npz", allow_pickle=False) as after:
            for name in ("indices", "labels"):
                require(np.array_equal(before[name], after[name]), "old validation split/mapping changed")
            for step in SPARSE:
                require(np.array_equal(before[f"logits_{step}"], after[f"logits_step_{step}"]), "reused logits differ")
        old_records = {r["step"]: r for r in old["validation"]}
        models = {}
        for key, row in records.items():
            # origin describes validation evaluation, not training of the weights.
            expected_origin = "r9_reused" if key != "actual" and row["step"] in SPARSE else "r13_new"
            require(row["origin"] == expected_origin, "checkpoint origin mismatch: " + key)
            checkpoint = safe(root, row["checkpoint"])
            require(sha(checkpoint) == row["checkpoint_sha256"], "checkpoint file hash mismatch")
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = loaded["model"] if "model" in loaded else loaded
            require(state_digest(model) == row["state_sha256"], "checkpoint model hash mismatch")
            models[key] = model
            expected_path = (run["output"] + "/post_task_004.pt" if key == "actual" else
                             old_records[row["step"]]["checkpoint"] if row["step"] in SPARSE else
                             settings["output"] + f"/{rid}/checkpoints/step_{row['step']:06d}.pt")
            require(row["checkpoint"] == expected_path, "wrong checkpoint provenance: " + key)
        training = read(rd / "training.json")
        summary = training["summary"]
        require(training["matched_steps"] == MATCHED, "old/new join step inventory")
        require(summary["start_step"] == 0 and summary["stop_step"] == 16
                and summary["optimizer_steps_executed"] == 16, "training budget mismatch")
        require(summary["batch_order_sha256"] == old["summary"]["batch_order_sha256"], "training batch order changed")
        require([x["step"] for x in summary["checkpoints"]] == list(range(17)), "training checkpoint summary inventory")
        total_updates += summary["optimizer_steps_executed"]
        require({p.name for p in (rd / "checkpoints").glob("*.pt")} == {f"step_{s:06d}.pt" for s in range(17)}, "early checkpoint inventory")
        old_identity = torch.load(safe(root, old_records[0]["checkpoint"]), map_location="cpu", weights_only=False)["identity"]
        for step in range(17):
            new = torch.load(rd / "checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
            require(new["step"] == step, "replay checkpoint step changed")
            check_replay_identity(new["identity"], old_identity)
            require(new["state_sha256"] == state_digest(new["model"]), "new checkpoint model digest")
            item = summary["checkpoints"][step]
            require(item["path"] == settings["output"] + f"/{rid}/checkpoints/step_{step:06d}.pt"
                    and item["state_sha256"] == new["state_sha256"], "training checkpoint summary lineage")
            if step == 16:
                require(summary["final_state_sha256"] == new["state_sha256"], "training final state mismatch")
            if step in MATCHED:
                equal_states(new["model"], models[f"step_{step}"])
        pre_state = torch.load(safe(root, run["output"] + "/pre_task_004.pt"), map_location="cpu", weights_only=False)
        equal_states(models["step_0"], pre_state["model"] if "model" in pre_state else pre_state)
        terminal = torch.load(safe(root, run["output"] + "/probe_task_004.pt"), map_location="cpu", weights_only=False)
        equal_states(models["step_628"], terminal["model"] if "model" in terminal else terminal)
        require(set(choices[rid]) == set(CONSTRAINTS), "selection constraint inventory")
        for constraint in CONSTRAINTS:
            require(set(choices[rid][constraint]) == set(POLICIES), "selection policy inventory")
            for policy in POLICIES:
                require(choices[rid][constraint][policy] == select(records, policy, constraint), "validation selection mismatch")
        expected_keys = ANCHORS | {k for c in choices[rid].values() for k in c.values() if k is not None}
        require(len(expected_keys) <= 7, "per-history test endpoint budget")
        test = read(rd / "test.json")
        require(test["run_id"] == rid and test["selection_seal_sha256"] == seal_sha
                and test["npz_sha256"] == sha(rd / "test.npz"), "test lineage mismatch")
        test_records = records_by_key(test["records"], expected_keys)
        for key, row in test_records.items():
            require({k: v for k, v in row.items() if k != "scores"} ==
                    {k: v for k, v in records[key].items() if k != "scores"}, "test state differs from selected validation state")
        test_records = check_npz(rd / "test.npz", test_records, test_indices, inverse[test_labels].astype(np.int64))
        total_test += len(test_records)
        all_runs.append({"run_id": rid, "seed": seed, "method": method, "validation": records,
                         "test": test_records, "selection": choices[rid]})
    require(total_updates == complete["optimizer_updates"] == 320, "total optimizer budget")
    require(total_test == complete["test_endpoints"] and total_test <= 140, "total test endpoint budget")
    require(complete["histories"] == 20, "completion matrix mismatch")
    require(complete["validation_endpoints"] == 500 and complete["new_validation_endpoints"] == 240
            and complete["reused_validation_endpoints"] == 260, "validation endpoint completion counts")
    require(complete["test_images"] == len(test_labels), "test image count mismatch")
    return {"status": "passed", "scope": "independent raw counts/selection/lineage audit, not independent forward or training replay",
            "manifest_sha256": sha(root / CANONICAL), "histories": 20, "validation_endpoints": 500,
            "new_validation_endpoints": 240, "optimizer_updates": total_updates,
            "test_endpoints": total_test, "original_sparse_logits_exact": True,
            "six_old_nodes_per_history_tensor_exact": True, "analysis": summarize(all_runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "audit/r13_early_retention/analysis_v1"
    require(not output.exists(), "audit output already exists; use a new directory")
    result = audit(args.root)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "analysis"}, indent=2))


if __name__ == "__main__":
    main()
