"""Export group scores and fit uniform offsets on separate calibration images."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

MODELS = ('usual_er', 'current_mask', 'mixed_full', 'mixed_restricted',
          'old_blend05', 'common_sgd')
TARGETS = (5, 10, 15, 20, 25, 30)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export(run, prior, output):
    assert json.loads((run / 'evaluation_complete.json').read_text())['status'] == 'complete'
    inputs = json.loads((run / 'inputs.json').read_text())['seeds']
    directory = output / 'inputs'; directory.mkdir(parents=True, exist_ok=True)
    for item in inputs:
        seed = item['seed']; original = Path(item['directory']) / 'evaluation'
        with np.load(original / 'actual_task_09.npz', allow_pickle=False) as data:
            labels = data['mapped_labels'].copy(); original_labels = data['original_labels'].copy()
            order = data['class_order'].copy(); indices = data['indices'].copy()
        assert np.array_equal(indices, np.arange(10000))
        assert np.array_equal(order[labels], original_labels)
        calibration = np.zeros(10000, dtype=bool)
        for c in range(100):
            ids = np.flatnonzero(original_labels == c); assert len(ids) == 100
            generator = np.random.Generator(np.random.PCG64(2026110800 + c))
            calibration[generator.permutation(ids)[:20]] = True
        for name in MODELS:
            if name in ('usual_er', 'common_sgd'):
                path = original / ('actual_task_09.npz' if name == 'usual_er' else 'probe_task_09.npz')
            elif name == 'current_mask':
                path = prior / 'runs' / f'seed{seed}' / 'logits.npz'
            else:
                path = run / 'runs' / name / f'seed{seed}' / 'logits.npz'
            with np.load(path, allow_pickle=False) as data:
                z = data['logits'].astype(np.float32)
            assert z.shape == (10000, 100) and np.isfinite(z).all()
            old_prediction = z[:, :90].argmax(1)
            current_prediction = 90 + z[:, 90:].argmax(1)
            old_score = z[np.arange(10000), old_prediction]
            current_score = z[np.arange(10000), current_prediction]
            original_prediction = z.argmax(1)
            rebuilt = np.where(current_score > old_score, current_prediction, old_prediction)
            assert np.array_equal(rebuilt, original_prediction)
            np.savez_compressed(directory / f's{seed}_{name}.npz',
                old_score=old_score, current_score=current_score,
                old_prediction=old_prediction.astype(np.int16),
                current_prediction=current_prediction.astype(np.int16),
                mapped_labels=labels.astype(np.int16), original_labels=original_labels.astype(np.int16),
                class_order=order.astype(np.int16), official_test_indices=indices.astype(np.int16),
                calibration=calibration, seed=np.int64(seed), model=np.asarray(name))


def measure(data, subset, scalar):
    y = data['mapped_labels']; old = subset & (y < 90); new = subset & (y >= 90)
    prediction = np.where(data['current_score'].astype(float) + scalar > data['old_score'].astype(float),
                          data['current_prediction'], data['old_prediction'])
    correct = prediction == y
    current_correct = data['current_prediction'] == y
    result = {'old_n': int(old.sum()), 'new_n': int(new.sum()),
              'old_correct': int(correct[old].sum()), 'new_correct': int(correct[new].sum()),
              'old_pct': float(correct[old].mean() * 100),
              'new_pct': float(correct[new].mean() * 100),
              'all_pct': float(correct[subset].mean() * 100),
              'old_within_pct': float((data['old_prediction'][old] == y[old]).mean() * 100),
              'current_within_pct': float(current_correct[new].mean() * 100)}
    for task in range(9):
        take = subset & (y // 10 == task)
        result[f'old_task{task}_pct'] = float(correct[take].mean() * 100)
    return result


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            'sample_sd': float(a.std(ddof=1)) if len(a) > 1 else None,
            'values': a.tolist()}


def calculate(output):
    rows, raw = [], []
    for seed in range(3101, 3106):
        reference_labels = reference_split = None
        for name in MODELS:
            with np.load(output / 'inputs' / f's{seed}_{name}.npz', allow_pickle=False) as loaded:
                data = {k: loaded[k].copy() for k in loaded.files}
            y = data['mapped_labels']; calibration = data['calibration']; test = ~calibration
            assert y.shape == calibration.shape == (10000,)
            assert np.array_equal(data['class_order'][y], data['original_labels'])
            assert np.array_equal(data['official_test_indices'], np.arange(10000))
            assert np.array_equal(np.bincount(y[calibration], minlength=100), np.full(100, 20))
            assert np.array_equal(np.bincount(y[test], minlength=100), np.full(100, 80))
            if reference_labels is None:
                reference_labels = y; reference_split = calibration
            else:
                assert np.array_equal(y, reference_labels) and np.array_equal(calibration, reference_split)
            for part, subset in [('calibration', calibration), ('test', test)]:
                raw.append({'seed': seed, 'model': name, 'images': part, **measure(data, subset, 0.)})
            correct_old = calibration & (y < 90) & (data['old_prediction'] == y)
            thresholds = data['old_score'][correct_old].astype(float) - data['current_score'][correct_old].astype(float)
            for q in TARGETS:
                required = int(np.ceil((q + 3) * 1800 / 100 - 1e-10))
                available = len(thresholds) >= required
                row = {'seed': seed, 'model': name, 'target_pct': q,
                       'calibration_target_pct': q + 3, 'required_old_correct': required,
                       'available': available, 'scalar': None, 'test_violation': None}
                if available:
                    scalar = float(np.sort(thresholds)[-required]); row['scalar'] = scalar
                    cm = measure(data, calibration, scalar)
                    assert cm['old_correct'] >= required
                    assert sum(thresholds >= scalar) >= required
                    assert sum(thresholds > scalar) < required
                    tm = measure(data, test, scalar)
                    row['test_violation'] = tm['old_pct'] < q - 1e-12
                    row.update({'calibration_' + k: v for k, v in cm.items()})
                    row.update({'test_' + k: v for k, v in tm.items()})
                rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    rows = [{k: r.get(k) for k in fields} for r in rows]
    write_csv(output / 'offset_results.csv', rows); write_csv(output / 'unadjusted_results.csv', raw)
    summaries = []
    for name in MODELS:
        for q in TARGETS:
            all_rows = [r for r in rows if r['model'] == name and r['target_pct'] == q]
            available = [r for r in all_rows if r['available']]
            summary = {'model': name, 'target_pct': q, 'available_seeds': len(available),
                       'unavailable_seeds': 5 - len(available),
                       'test_violations': sum(r['test_violation'] for r in available),
                       'seed_ids': [r['seed'] for r in available]}
            summary.update({k: summarize([r['test_' + k] for r in available])
                            for k in ('old_pct', 'new_pct', 'all_pct')})
            summaries.append(summary)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'models': 30, 'offset_attempts': len(rows),
                      'available_attempts': sum(r['available'] for r in rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-from', type=Path)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.export_from:
        assert args.prior is not None
        plan = args.export_from / 'SCORE_PLAN.md'
        meta = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'calibration_per_class': 20, 'test_per_class': 80,
                'validation_margin_pp': 3, 'targets_pct': list(TARGETS)}
        (args.output / 'started.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        shutil.copy2(plan, args.output / 'recorded_design.md')
        shutil.copy2(Path(__file__), args.output / 'executed_analysis.py')
        export(args.export_from, args.prior, args.output)
    calculate(args.output)


if __name__ == '__main__':
    main()
