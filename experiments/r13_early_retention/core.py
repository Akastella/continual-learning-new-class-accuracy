"""Exact count-based selection and all-class scoring for the R13 protocol."""

from numbers import Integral

import numpy as np


SPARSE = (0, 1, 2, 4, 8, 16, 32, 64, 128, 157, 314, 471, 628)
DENSE = tuple(sorted(set(SPARSE) | set(range(17))))
POLICIES = (
    'sparse_probe', 'dense_probe', 'sparse_plus_actual', 'dense_plus_actual',
)
CONSTRAINTS = ('q20', 'q20_rel5')


def scores(logits, labels):
    """Score all ten outputs; denominators come from the supplied labels.

    Every class must be represented, as in both fixed R13 evaluation datasets.
    No equal-class-count assumption is made. Ties use NumPy's first argmax.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype != np.int64 or not len(labels):
        raise ValueError('Expected a nonempty int64 label vector')
    if np.any((labels < 0) | (labels >= 10)):
        raise ValueError('Labels must be mapped to the ten output classes')
    if logits.dtype != np.float32 or logits.shape != (len(labels), 10):
        raise ValueError('Expected aligned FP32 logits with ten columns')
    if not np.isfinite(logits).all():
        raise ValueError('Every candidate logit must be finite')
    class_n = np.bincount(labels, minlength=10).astype(np.int64).tolist()
    if min(class_n) <= 0:
        raise ValueError('Evaluation must contain each of the ten classes')
    prediction = logits.argmax(axis=1)
    class_correct = np.bincount(labels[prediction == labels], minlength=10).tolist()
    old_correct, old_n = sum(class_correct[:8]), sum(class_n[:8])
    current_correct, current_n = sum(class_correct[8:]), sum(class_n[8:])
    return {
        'class_correct': class_correct, 'class_n': class_n,
        'old_correct': old_correct, 'old_n': old_n,
        'current_correct': current_correct, 'current_n': current_n,
        'old_pct': 100 * old_correct / old_n,
        'current_pct': 100 * current_correct / current_n,
        'all_pct': 100 * sum(class_correct) / len(labels),
        'old_task_pct': [
            100 * sum(class_correct[i:i + 2]) / sum(class_n[i:i + 2])
            for i in range(0, 8, 2)
        ],
    }


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} is out of range')
    return value


def _counts(row):
    """Validate even rows excluded by the requested candidate policy."""
    try:
        step = _integer(row['step'], 'step')
        kind, key = row['kind'], row['key']
        if kind == 'probe':
            if step not in DENSE or key != f'step_{step}':
                raise ValueError('Invalid probe identity')
        elif kind == 'actual':
            if step != 628 or key != 'actual':
                raise ValueError('Actual must have identity actual and step 628')
        else:
            raise ValueError('Unknown candidate kind')
        result = row['scores']
        old_n = _integer(result['old_n'], 'old_n', 1)
        current_n = _integer(result['current_n'], 'current_n', 1)
        old = _integer(result['old_correct'], 'old_correct')
        current = _integer(result['current_correct'], 'current_correct')
        if old > old_n or current > current_n:
            raise ValueError('Correct count exceeds its denominator')
        return old, old_n, current, current_n
    except (KeyError, TypeError) as error:
        raise ValueError('Malformed candidate row') from error


def choose(rows, policy, constraint):
    """Return a candidate key, or None when the fixed feasible set is empty.

    All rows belong to one history and one evaluation set. This helper permits
    partial grids for unit-level checks; the experiment runner audits full-grid
    completeness. At least the historical step-0 row is always required.
    """
    if policy not in POLICIES or constraint not in CONSTRAINTS:
        raise ValueError('Unknown R13 policy or constraint')
    validated = []
    seen = set()
    for row in rows:
        counts = _counts(row)
        if row['key'] in seen:
            raise ValueError('Duplicate candidate key')
        seen.add(row['key'])
        validated.append((row, counts))
    starts = [counts for row, counts in validated if row['key'] == 'step_0']
    if len(starts) != 1:
        raise ValueError('Exactly one historical step-0 row is required')
    start_old, start_n, _, start_current_n = starts[0]
    if any(counts[1] != start_n or counts[3] != start_current_n
           for _, counts in validated):
        raise ValueError('Candidates must share evaluation denominators')
    grid = DENSE if policy.startswith('dense_') else SPARSE
    include_actual = policy.endswith('_plus_actual')
    feasible = []
    for row, (old, old_n, current, _) in validated:
        if row['kind'] == 'actual':
            if not include_actual:
                continue
        elif row['step'] not in grid:
            continue
        if 100 * old < 20 * old_n:
            continue
        if constraint == 'q20_rel5' and (
            100 * (start_old * old_n - old * start_n) > 5 * start_n * old_n
        ):
            continue
        feasible.append(((-current, row['step'], int(row['kind'] == 'actual')), row['key']))
    return min(feasible)[1] if feasible else None
