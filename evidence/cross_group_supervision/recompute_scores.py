"""Recompute the separate calibration/test offsets and paired seed effects."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import numpy as np

METHODS = ('usual_er', 'current_mask', 'mixed_full', 'mixed_restricted', 'old_blend05')
PRIMARY_TARGETS = (5, 10, 15, 20)
SEEDS = tuple(range(3101, 3106))
METRICS = ('old_pct', 'new_pct', 'all_pct') + tuple(f'old_task{i}_pct' for i in range(9))


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def stats(values, seeds):
    a = np.asarray(values, dtype=np.float64)
    return {'n': len(a), 'seed_ids': seeds, 'values': a.tolist(),
            'mean': float(a.mean()) if len(a) else None,
            'sample_sd': float(a.std(ddof=1)) if len(a) > 1 else None,
            'positive': int((a > 0).sum()), 'zero': int((a == 0).sum()),
            'negative': int((a < 0).sum())}


def paired_effects(rows, targets):
    lookup = {(int(r['target_pct']), r['model'], int(r['seed'])): r for r in rows}
    effects, model_summaries = {}, []
    for q in targets:
        selected = {name: [s for s in SEEDS if lookup[q, name, s]['available'] == 'True']
                    for name in METHODS}
        common = [s for s in SEEDS if all(s in selected[name] for name in METHODS)]
        if q in PRIMARY_TARGETS:
            assert common == list(SEEDS), ('Expected all five paired seeds', q, common)
        effects[str(q)] = {}
        for name in METHODS:
            seeds = selected[name]
            model_summaries.append({'model': name, 'target_pct': q,
                'available_seeds': len(seeds), 'unavailable_seeds': len(SEEDS) - len(seeds),
                'seed_ids': seeds,
                'test_violations': sum(lookup[q, name, s]['test_violation'] == 'True' for s in seeds),
                **{m: stats([float(lookup[q, name, s]['test_' + m]) for s in seeds], seeds)
                   for m in METRICS}})
        for metric in METRICS:
            def difference(a, b, seeds):
                return [float(lookup[q, a, s]['test_' + metric]) -
                        float(lookup[q, b, s]['test_' + metric]) for s in seeds]
            comparisons = {}
            # Keep the recorded comparison names and add every pair of the five methods.
            named = {'old_only_restricted_minus_full': ('current_mask', 'usual_er'),
                     'mixed_restricted_minus_full': ('mixed_restricted', 'mixed_full'),
                     'blend05_minus_restricted': ('old_blend05', 'current_mask'),
                     'blend05_minus_full': ('old_blend05', 'usual_er')}
            named.update({f'{a}_minus_{b}': (a, b) for a, b in itertools.combinations(METHODS, 2)})
            for name, (a, b) in named.items():
                seeds = [s for s in SEEDS if s in selected[a] and s in selected[b]]
                comparisons[name] = stats(difference(a, b, seeds), seeds)
            seeds = [s for s in SEEDS if all(s in selected[name] for name in METHODS[:4])]
            mixed = difference('mixed_restricted', 'mixed_full', seeds)
            old = difference('current_mask', 'usual_er', seeds)
            comparisons['policy_interaction'] = stats([a - b for a, b in zip(mixed, old)], seeds)
            effects[str(q)][metric] = comparisons
    return effects, model_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'score_analysis')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the originally executed calculate/measure functions byte for byte.
    spec = importlib.util.spec_from_file_location('executed_analysis', data / 'executed_analysis.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    (output / 'inputs').mkdir()
    hashes = {}
    reference_split = None
    for seed in SEEDS:
        for name in original.MODELS:
            path = data / 'inputs' / f's{seed}_{name}.npz'
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as d:
                y, labels, split = d['mapped_labels'], d['original_labels'], d['calibration']
                assert int(d['seed']) == seed and str(d['model']) == name
                assert y.shape == labels.shape == split.shape == (10000,)
                assert split.dtype == np.bool_ and int(split.sum()) == 2000
                expected = np.zeros(10000, dtype=bool)
                for c in range(100):
                    ids = np.flatnonzero(labels == c)
                    assert len(ids) == 100
                    generator = np.random.Generator(np.random.PCG64(2026110800 + c))
                    expected[generator.permutation(ids)[:20]] = True
                assert np.array_equal(split, expected)
                if reference_split is None:
                    reference_split = split.copy()
                else:
                    assert np.array_equal(split, reference_split)
                for score in ('old_score', 'current_score'):
                    assert d[score].shape == (10000,) and np.isfinite(d[score]).all()
            shutil.copy2(path, output / 'inputs' / path.name)
    original.calculate(output)
    rows = read_rows(output / 'offset_results.csv')
    assert len(rows) == 180
    for filename in ('offset_results.csv', 'unadjusted_results.csv'):
        assert read_rows(output / filename) == read_rows(data / filename), filename
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary == json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    effects, allq = paired_effects(rows, original.TARGETS)
    recorded = json.loads((data / 'recorded_paired_effects.json').read_text(encoding='utf-8'))
    for q, metrics in recorded.items():
        for metric, comparisons in metrics.items():
            for name, values in comparisons.items():
                got = effects[q][metric][name]
                for key, value in values.items():
                    assert got[key] == value, (q, metric, name, key)
    (output / 'paired_effects.json').write_text(json.dumps(effects, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'allq_seed_summary.json').write_text(json.dumps(allq, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    verification = {'status': 'passed', 'group_score_files': len(hashes),
        'fixed_calibration_images': 2000, 'fixed_test_images': 8000,
        'offset_attempts': len(rows), 'available_attempts': sum(r['available'] == 'True' for r in rows),
        'all_offset_and_old_task_rows_match': True, 'all_unadjusted_rows_match': True,
        'all_summary_values_match': True, 'recorded_paired_effects_exactly_match': True,
        'primary_paired_targets_pct': list(PRIMARY_TARGETS), 'primary_paired_seed_n': 5,
        'group_score_sha256': hashes, 'training_performed': False, 'inference_performed': False}
    (output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'group_score_sha256'}))


if __name__ == '__main__':
    main()
