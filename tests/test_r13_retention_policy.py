"""R13 policy boundaries and nested-candidate properties; CPU only."""

import numpy as np
import pytest

from experiments.r13_early_retention.core import (
    CONSTRAINTS, DENSE, POLICIES, SPARSE, choose, scores,
)


def row(step, old, current, *, actual=False, old_n=800, current_n=200):
    return {
        'key': 'actual' if actual else f'step_{step}',
        'step': step, 'kind': 'actual' if actual else 'probe',
        'scores': {'old_correct': old, 'old_n': old_n,
                   'current_correct': current, 'current_n': current_n},
    }


def test_absolute_boundary_has_no_rounding_or_epsilon():
    candidates = [row(0, 159, 0), row(1, 160, 40), row(2, 159, 200)]
    assert choose(candidates, 'sparse_probe', 'q20') == 'step_1'
    assert choose(candidates, 'sparse_probe', 'q20_rel5') == 'step_1'
    assert choose([row(0, 159, 0), row(1, 159, 200)], 'sparse_probe', 'q20') is None


def test_relative_boundary_and_old_accuracy_improvement():
    candidates = [row(0, 320, 0), row(1, 280, 40), row(2, 279, 200)]
    assert choose(candidates, 'sparse_probe', 'q20_rel5') == 'step_1'
    assert choose(candidates, 'sparse_probe', 'q20') == 'step_2'
    assert choose([row(0, 159, 0), row(1, 170, 5)],
                  'sparse_probe', 'q20_rel5') == 'step_1'


def test_relative_constraint_does_not_replace_absolute_floor():
    candidates = [row(0, 120, 0), row(1, 100, 150)]
    assert choose(candidates, 'sparse_probe', 'q20_rel5') is None


def test_dynamic_denominators_retain_exact_relative_boundary():
    candidates = [row(0, 52, 0, old_n=160, current_n=31),
                  row(1, 44, 20, old_n=160, current_n=31),
                  row(2, 43, 31, old_n=160, current_n=31)]
    assert choose(candidates, 'sparse_probe', 'q20_rel5') == 'step_1'


def test_ties_prefer_earlier_step_then_probe_at_628():
    candidates = [row(0, 160, 0), row(628, 160, 100, actual=True),
                  row(628, 160, 100), row(16, 160, 99)]
    assert choose(candidates, 'dense_plus_actual', 'q20') == 'step_628'
    candidates[-1]['scores']['current_correct'] = 100
    assert choose(candidates, 'dense_plus_actual', 'q20') == 'step_16'


def test_new_dense_states_do_not_change_sparse_choice_or_create_fallback():
    candidates = [row(0, 159, 0), row(1, 159, 100), row(3, 160, 99),
                  row(628, 159, 200, actual=True)]
    assert choose(candidates, 'sparse_probe', 'q20') is None
    assert choose(candidates, 'sparse_plus_actual', 'q20') is None
    assert choose(candidates, 'dense_probe', 'q20') == 'step_3'
    candidates[1]['scores']['old_correct'] = 160
    assert choose(candidates, 'sparse_probe', 'q20') == 'step_1'


def test_nested_policies_preserve_feasibility_and_bound_possible_winners():
    rng = np.random.default_rng(18497)
    for _ in range(80):
        candidates = [row(s, int(rng.integers(801)), int(rng.integers(201))) for s in DENSE]
        candidates.append(row(628, int(rng.integers(801)), int(rng.integers(201)), actual=True))
        lookup = {r['key']: r for r in candidates}
        winners = set()
        for constraint in CONSTRAINTS:
            selected = {p: choose(candidates, p, constraint) for p in POLICIES}
            for prefix in ['sparse', 'dense']:
                probe = selected[prefix + '_probe']
                augmented = selected[prefix + '_plus_actual']
                assert augmented in {probe, 'actual'}
                if probe is not None:
                    assert augmented is not None
            for suffix in ['probe', 'plus_actual']:
                sparse = selected['sparse_' + suffix]
                dense = selected['dense_' + suffix]
                if sparse is not None:
                    assert dense is not None
                    assert lookup[dense]['scores']['current_correct'] >= lookup[sparse]['scores']['current_correct']
            winners.update(v for v in selected.values() if v not in {None, 'actual'})
        assert len(winners) <= 4


def test_scores_use_actual_class_counts_and_all_ten_outputs():
    labels = np.repeat(np.arange(10, dtype=np.int64), np.arange(1, 11))
    logits = np.zeros((55, 10), dtype=np.float32)
    logits[:, 9] = 1
    result = scores(logits, labels)
    assert result['class_n'] == list(range(1, 11))
    assert result['class_correct'] == [0] * 9 + [10]
    assert result['old_n'] == 36 and result['current_n'] == 19
    assert result['old_pct'] == 0
    assert result['current_pct'] == pytest.approx(1000 / 19)
    assert result['all_pct'] == pytest.approx(1000 / 55)
    assert result['old_task_pct'] == [0, 0, 0, 0]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_any_nonfinite_logit_is_rejected_even_if_not_the_winner(bad):
    labels = np.arange(10, dtype=np.int64)
    logits = np.eye(10, dtype=np.float32)
    logits[2, 9] = bad
    with pytest.raises(ValueError):
        scores(logits, labels)


@pytest.mark.parametrize('labels', [
    np.arange(10, dtype=np.int32), np.arange(10, dtype=np.float32),
    np.arange(10, dtype=np.int64).reshape(2, 5),
    np.arange(-1, 9, dtype=np.int64), np.arange(1, 11, dtype=np.int64),
    np.zeros(10, dtype=np.int64),
])
def test_invalid_label_schema_is_rejected(labels):
    with pytest.raises(ValueError):
        scores(np.eye(10, dtype=np.float32), labels)


@pytest.mark.parametrize('logits', [np.eye(10, dtype=np.float64),
                                    np.zeros((10, 9), dtype=np.float32),
                                    np.zeros((9, 10), dtype=np.float32)])
def test_invalid_logit_schema_is_rejected(logits):
    with pytest.raises(ValueError):
        scores(logits, np.arange(10, dtype=np.int64))


@pytest.mark.parametrize('mutation', ['nan_count', 'float_count', 'excess_count',
                                    'bad_kind', 'bad_key', 'bad_actual_step', 'denominator'])
def test_invalid_excluded_candidate_is_rejected_before_policy_filter(mutation):
    candidates = [row(0, 200, 0), row(628, 200, 200, actual=True)]
    bad = candidates[1]
    if mutation == 'nan_count': bad['scores']['old_correct'] = np.nan
    elif mutation == 'float_count': bad['scores']['old_correct'] = 200.0
    elif mutation == 'excess_count': bad['scores']['current_correct'] = 201
    elif mutation == 'bad_kind': bad['kind'] = 'fresh'
    elif mutation == 'bad_key': bad['key'] = 'step_628'
    elif mutation == 'bad_actual_step': bad['step'] = 629
    elif mutation == 'denominator': bad['scores']['old_n'] = 900
    with pytest.raises(ValueError):
        choose(candidates, 'sparse_probe', 'q20')


def test_duplicate_or_missing_origin_and_unknown_policy_fail():
    with pytest.raises(ValueError): choose([row(1, 200, 0)], 'sparse_probe', 'q20')
    with pytest.raises(ValueError): choose([row(0, 200, 0)] * 2, 'sparse_probe', 'q20')
    with pytest.raises(ValueError): choose([row(0, 200, 0)], 'picked_later', 'q20')
    with pytest.raises(ValueError): choose([row(0, 200, 0)], 'sparse_probe', 'q19')


def test_protocol_candidate_sets_are_nested_with_eleven_missing_early_steps():
    assert len(SPARSE) == 13 and len(DENSE) == 24
    assert set(DENSE) - set(SPARSE) == {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}
