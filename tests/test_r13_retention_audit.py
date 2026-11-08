"""CPU fixtures target failure modes, independently of the R13 runner."""
from fractions import Fraction

import numpy as np
import pytest

from analysis.audit_r13_early_retention import (
    ANCHORS, DENSE, SPARSE, check_npz, check_replay_identity, describe, feasible, metric,
    paired_summary, read, records_by_key, safe, scores, select, violations,
)


def count_score(old=160, current=20, old_n=800, current_n=200):
    # For selection only: score denominators need not be the test denominators.
    return {"old_correct": old, "old_n": old_n,
            "current_correct": current, "current_n": current_n}


def candidates():
    rows = {f"step_{s}": {"key": f"step_{s}", "step": s, "kind": "probe",
                           "scores": count_score(old=0, current=0)} for s in DENSE}
    rows["step_0"]["scores"] = count_score(old=240, current=0)
    rows["actual"] = {"key": "actual", "step": 628, "kind": "actual",
                      "scores": count_score(old=0, current=0)}
    return rows


def test_absolute_and_relative_feasibility_exact_boundaries():
    pre = count_score(old=240)
    assert feasible(count_score(old=160), pre, "q20")
    assert not feasible(count_score(old=159), pre, "q20")
    assert feasible(count_score(old=200), pre, "q20_rel5")
    assert not feasible(count_score(old=199), pre, "q20_rel5")
    # 100/400 versus pre 240/800 is exactly a 5pp loss.
    assert feasible(count_score(old=100, old_n=400), pre, "q20_rel5")
    assert not feasible(count_score(old=99, old_n=400), pre, "q20_rel5")


def test_dense_grid_and_actual_eligibility_are_distinct_changes():
    rows = candidates()
    rows["step_2"]["scores"] = count_score(current=21)
    rows["step_3"]["scores"] = count_score(current=30)
    rows["actual"]["scores"] = count_score(current=40)
    assert select(rows, "sparse_probe", "q20") == "step_2"
    assert select(rows, "dense_probe", "q20") == "step_3"
    assert select(rows, "sparse_plus_actual", "q20") == "actual"
    assert select(rows, "dense_plus_actual", "q20") == "actual"
    assert len(DENSE) == 24 and len(set(DENSE) - set(SPARSE)) == 11


def test_ties_use_earliest_then_probe_before_actual_at_628():
    rows = candidates()
    rows["step_3"]["scores"] = count_score(current=30)
    rows["step_4"]["scores"] = count_score(current=30)
    rows["step_628"]["scores"] = count_score(current=30)
    rows["actual"]["scores"] = count_score(current=30)
    assert select(rows, "dense_plus_actual", "q20") == "step_3"
    rows["step_3"]["scores"] = count_score(old=0)
    rows["step_4"]["scores"] = count_score(old=0)
    assert select(rows, "dense_plus_actual", "q20") == "step_628"


def test_null_is_not_a_fallback_and_relative_constraint_is_history_specific():
    rows = candidates()
    rows["step_0"]["scores"] = count_score(old=0)
    assert select(rows, "dense_plus_actual", "q20") is None
    rows["step_0"]["scores"] = count_score(old=320)
    rows["step_3"]["scores"] = count_score(old=240, current=100)
    assert select(rows, "dense_probe", "q20") == "step_3"
    assert select(rows, "dense_probe", "q20_rel5") == "step_0"


def test_scores_use_actual_unequal_label_counts_and_argmax_tie_rule():
    labels = np.repeat(np.arange(10), np.arange(1, 11)).astype(np.int64)
    logits = np.zeros((len(labels), 10), dtype=np.float32)
    score = scores(logits, labels)  # all ties choose output 0
    assert score["class_n"] == list(range(1, 11))
    assert score["old_n"] == 36 and score["current_n"] == 19
    assert score["old_correct"] == 1 and score["current_correct"] == 0
    assert metric(score, "old") == Fraction(25, 9)
    assert score["old_task_pct"][0] == 100 / 3
    logits[2, 4] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scores(logits, labels)


def test_test_violations_retain_failures_and_exact_five_pp_boundary():
    pre = count_score(old=240)
    edge = violations(count_score(old=200), pre, "q20_rel5")
    assert not edge["relative_drop_above_5"] and not edge["constraint_violation"]
    failure = violations(count_score(old=199), pre, "q20_rel5")
    assert failure["relative_drop_above_5"] and failure["constraint_violation"]
    assert not failure["absolute_below_20"]
    assert not violations(count_score(old=199), pre, "q20")["constraint_violation"]


def test_fraction_aggregation_exact_zero_and_unavailable_sd():
    report = describe([Fraction(1, 3), Fraction(-1, 3), Fraction(0)])
    assert report["mean"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert report["sample_variance"]["numerator"] == 1
    assert report["sample_variance"]["denominator"] == 9
    assert (report["positive"], report["negative"], report["zero"]) == (1, 1, 1)
    assert describe([])["mean"] is None
    assert describe([Fraction(1, 3)])["sample_sd"] is None


def scored(current):
    labels = np.repeat(np.arange(10), 10).astype(np.int64)
    logits = np.zeros((100, 10), np.float32)
    for i in range(80, 80 + current):
        logits[i, labels[i]] = 1
    return scores(logits, labels)


def item(seed, score):
    return {"seed": seed, "key": "step_3" if score else None, "test": score,
            "test_pre": scored(0), "test_actual": scored(10)}


def test_paired_effects_use_common_seeds_not_different_group_means():
    pairs = [(item(201, scored(1)), item(201, scored(3))),
             (item(202, None), item(202, scored(20))),
             (item(203, scored(19)), item(203, None))]
    report = paired_summary(pairs, {"contrast": "right minus left"})
    assert report["common_seeds"] == [201]
    assert report["metrics"]["current"]["mean"]["value"] == 10.0
    assert len(report["availability"]) == 3
    assert len(report["rows"]) == 1


def test_raw_npz_rejects_extra_test_oracle_array_and_wrong_source_labels(tmp_path):
    labels = np.repeat(np.arange(10), 2).astype(np.int64)
    logits = np.zeros((20, 10), dtype=np.float32)
    indices = np.arange(20, dtype=np.int64)
    rows = {"step_0": {"scores": scores(logits, labels)}}
    path = tmp_path / "raw.npz"
    np.savez(path, indices=indices, labels=labels, logits_step_0=logits)
    assert check_npz(path, rows, indices, labels)["step_0"]["scores"] == rows["step_0"]["scores"]
    with pytest.raises(ValueError, match="mapped labels"):
        check_npz(path, rows, indices, labels[::-1])
    np.savez(path, indices=indices, labels=labels, logits_step_0=logits, logits_step_3=logits)
    with pytest.raises(ValueError, match="inventory"):
        check_npz(path, rows, indices, labels)


def test_endpoint_inventory_rejects_duplicates_missing_and_wrong_kind():
    row = {"key": "actual", "kind": "actual", "step": 628}
    with pytest.raises(ValueError, match="duplicate"):
        records_by_key([row, row], {"actual"})
    with pytest.raises(ValueError, match="missing"):
        records_by_key([row], ANCHORS)
    with pytest.raises(ValueError, match="step/kind"):
        records_by_key([dict(row, kind="probe")], {"actual"})


def test_evidence_paths_and_json_reject_ambiguity(tmp_path):
    for name in ("../outside", "C:/outside", "a\\b", "/outside"):
        with pytest.raises(ValueError):
            safe(tmp_path, name)
    path = tmp_path / "duplicate.json"
    path.write_text('{"chosen":"step_0","chosen":"actual"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        read(path)


def test_replay_identity_allows_only_the_prespecified_save_grid_change():
    old = {"candidates": SPARSE, "task_id": 4, "order_sha256": ["same"], "config": {"lr": .01}}
    new = dict(old, candidates=DENSE)
    check_replay_identity(new, old)
    with pytest.raises(ValueError, match="training identity"):
        check_replay_identity(dict(new, config={"lr": .02}), old)
    with pytest.raises(ValueError, match="candidate schedule"):
        check_replay_identity(dict(new, candidates=list(range(17))), old)
