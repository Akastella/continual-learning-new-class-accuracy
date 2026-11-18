"""Recompute all task-7 counts and fixed score analyses with NumPy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS = tuple(range(3101, 3106))

METHODS = ('usual_er', 'old_restricted', 'old_blend05', 'mixed_full', 'mixed_restricted', 'common_sgd')

TARGETS = (5, 10, 15, 20, 25, 30)

METRICS = ('new_pct', 'current10_pct', 'competition_pp', 'old_pct', 'all_pct')

PAIRS = {'old_restricted_minus_full': ('old_restricted', 'usual_er'),
         'mixed_restricted_minus_full': ('mixed_restricted', 'mixed_full'),
         'blend05_minus_zero': ('old_blend05', 'old_restricted'),
         'blend05_minus_full': ('old_blend05', 'usual_er'),
         'mixed_full_minus_old_full': ('mixed_full', 'usual_er'),
         'mixed_restricted_minus_old_restricted': ('mixed_restricted', 'old_restricted')}

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def statistics(numerators, denominator):
    values = np.asarray(numerators, dtype=np.float64)/denominator
    return {'n': len(numerators), 'values': values.tolist(),
            'mean': sum(numerators)/(len(numerators)*denominator) if len(numerators) else None,
            'sample_sd': float(values.std(ddof=1)) if len(values)>1 else None,
            'positive_seeds': sum(v>0 for v in numerators),
            'zero_seeds': sum(v==0 for v in numerators), 'negative_seeds': sum(v<0 for v in numerators)}

def split_mask(original_labels):
    calibration = np.zeros(10000, dtype=bool)
    for original_class in range(100):
        ids = np.flatnonzero(original_labels == original_class)
        assert len(ids) == 100
        generator = np.random.Generator(np.random.PCG64(2026110800+original_class))
        calibration[generator.permutation(ids)[:20]] = True
    return calibration

def measure(data, subset, bias=0.):
    labels = data['mapped_labels']
    old = subset & (labels<70)
    new = subset & (labels>=70) & (labels<80)
    predicted = np.where(data['current_score'].astype(np.float64)+bias > data['old_score'].astype(np.float64),
                         data['predictions_current10'], data['old_prediction'])
    correct = predicted == labels
    old_n, new_n = int(old.sum()), int(new.sum())
    old_correct, new_correct = int(correct[old].sum()), int(correct[new].sum())
    current_correct = int((data['predictions_current10'][new] == labels[new]).sum())
    within_old_correct = int((data['old_prediction'][old] == labels[old]).sum())
    result = {'old_n': old_n, 'new_n': new_n, 'all_n': old_n+new_n,
        'old_correct': old_correct, 'new_correct': new_correct, 'all_correct': old_correct+new_correct,
        'current10_correct': current_correct, 'competition_lost_correct': current_correct-new_correct,
        'within_old_correct': within_old_correct,
        'old_pct': 100*old_correct/old_n, 'new_pct': 100*new_correct/new_n,
        'all_pct': 100*(old_correct+new_correct)/(old_n+new_n),
        'current10_pct': 100*current_correct/new_n,
        'competition_pp': 100*(current_correct-new_correct)/new_n,
        'within_old_pct': 100*within_old_correct/old_n}
    for task in range(7):
        take = subset & (labels//10 == task)
        result[f'old_task{task}_n'] = int(take.sum())
        result[f'old_task{task}_correct'] = int(correct[take].sum())
        result[f'old_task{task}_pct'] = 100*int(correct[take].sum())/int(take.sum())
    return result

def calculate():
    raw, offsets, unadjusted = [], [], []
    for seed in SEEDS:
        reference = None
        for method in METHODS:
            path = ROOT / 'reader_data' / f's{seed}_{method}.npz'
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            labels = data['mapped_labels']
            assert labels.shape == (10000,) and np.array_equal(data['official_test_indices'], np.arange(10000))
            assert np.array_equal(data['class_order'][labels], data['original_labels'])
            assert np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100))
            assert int(data['seed']) == seed and str(data['model']) == method
            if reference is None:
                reference = [data[key] for key in ('mapped_labels', 'original_labels', 'class_order')]
            else:
                assert all(np.array_equal(data[key], ref) for key, ref in zip(('mapped_labels', 'original_labels', 'class_order'), reference))
            rebuilt = np.where(data['current_score'] > data['old_score'], data['predictions_current10'], data['old_prediction'])
            assert np.array_equal(rebuilt, data['predictions_all80'])
            seen = labels < 80
            calibration = split_mask(data['original_labels']) & seen
            test = seen & ~calibration
            assert calibration.sum() == 1600 and test.sum() == 6400
            assert int((calibration & (labels<70)).sum()) == 1400
            assert int((test & (labels<70)).sum()) == 5600
            assert np.array_equal(np.bincount(labels[calibration], minlength=100), np.r_[np.full(80, 20), np.zeros(20)])
            assert np.array_equal(np.bincount(labels[test], minlength=100), np.r_[np.full(80, 80), np.zeros(20)])
            raw.append({'seed': seed, 'method': method, **measure(data, seen)})
            for part, subset in (('calibration', calibration), ('test', test)):
                unadjusted.append({'seed': seed, 'method': method, 'images': part, **measure(data, subset)})
            correct_old = calibration & (labels<70) & (data['old_prediction']==labels)
            thresholds = data['old_score'][correct_old].astype(np.float64)-data['current_score'][correct_old].astype(np.float64)
            for target in TARGETS:
                required = (target+3)*14
                available = len(thresholds) >= required
                row = {'seed': seed, 'method': method, 'q': target, 'calibration_q': target+3,
                       'required_old_correct': required, 'available': available, 'bias': None, 'test_violation': None}
                if available:
                    bias = float(np.sort(thresholds)[-required])
                    assert int((thresholds >= bias).sum()) >= required and int((thresholds > bias).sum()) < required
                    calibration_scores = measure(data, calibration, bias)
                    test_scores = measure(data, test, bias)
                    assert calibration_scores['old_correct'] >= required
                    row.update(bias=bias, test_violation=test_scores['old_correct']*100 < target*5600)
                    row.update({f'cal_{key}': value for key, value in calibration_scores.items()})
                    row.update({f'test_{key}': value for key, value in test_scores.items()})
                offsets.append(row)
    return raw, offsets, unadjusted

def summarize(raw, offsets, unadjusted):
    lookup = {(row['seed'], row['method']): row for row in raw}
    fields = {'new_pct': ('new_correct', 10), 'current10_pct': ('current10_correct', 10),
              'competition_pp': ('competition_lost_correct', 10), 'old_pct': ('old_correct', 70), 'all_pct': ('all_correct', 80)}
    methods = {method: {metric: statistics([lookup[(seed, method)][field] for seed in SEEDS], denominator)
                         for metric, (field, denominator) in fields.items()} for method in METHODS}
    effects = {name: {metric: statistics([lookup[(seed, a)][field]-lookup[(seed, b)][field] for seed in SEEDS], denominator)
                      for metric, (field, denominator) in fields.items()} for name, (a, b) in PAIRS.items()}
    interaction = {metric: statistics([lookup[(seed, 'mixed_restricted')][field]-lookup[(seed, 'mixed_full')][field]
                    -lookup[(seed, 'old_restricted')][field]+lookup[(seed, 'usual_er')][field] for seed in SEEDS], denominator)
                    for metric, (field, denominator) in fields.items()}
    new_interaction, competition = interaction['new_pct'], interaction['competition_pp']
    h1 = {'new_interaction': new_interaction, 'competition_interaction': competition,
          'supported': new_interaction['mean']>0 and new_interaction['positive_seeds']>=3
                       and competition['mean']<0 and competition['negative_seeds']>=3}
    recovery = statistics([2*lookup[(seed, 'old_blend05')]['new_correct']-lookup[(seed, 'usual_er')]['new_correct'] for seed in SEEDS], 20)
    old_advantage = effects['blend05_minus_full']['old_pct']
    h2 = {'coefficient': .05, 'fraction_of_original_er': .5, 'new_minus_half_er': recovery,
          'old_minus_er': old_advantage,
          'recovery_seeds': recovery['positive_seeds']+recovery['zero_seeds'],
          'supported': recovery['mean']>=0 and recovery['positive_seeds']+recovery['zero_seeds']>=3
                       and old_advantage['mean']>0 and old_advantage['positive_seeds']>=3}
    offset_lookup = {(row['seed'], row['method'], row['q']): row for row in offsets}
    adjusted = []
    score_fields = {'new_pct': ('new_correct', 8), 'old_pct': ('old_correct', 56), 'all_pct': ('all_correct', 64),
                    'current10_pct': ('current10_correct', 8), 'within_old_pct': ('within_old_correct', 56),
                    **{f'old_task{task}_pct': (f'old_task{task}_correct', 8) for task in range(7)}}
    unadjusted_test = {method: {metric: statistics([row[field] for row in unadjusted if row['method']==method and row['images']=='test'], denominator)
                               for metric, (field, denominator) in score_fields.items()} for method in METHODS}
    for method in METHODS:
        for target in TARGETS:
            rows = [offset_lookup[(seed, method, target)] for seed in SEEDS]
            available = [row for row in rows if row['available']]
            adjusted.append({'method': method, 'q': target, 'total_seed_units': 5,
                'available_seeds': len(available), 'unavailable_seeds': [row['seed'] for row in rows if not row['available']],
                'test_violations': [row['seed'] for row in available if row['test_violation']],
                **{metric: statistics([row['test_'+field] for row in available], denominator)
                   for metric, (field, denominator) in score_fields.items()}})
    paired_q20 = {}
    for name, (a, b) in PAIRS.items():
        usable = [seed for seed in SEEDS if offset_lookup[(seed, a, 20)]['available'] and offset_lookup[(seed, b, 20)]['available']]
        per_seed = []
        for seed in SEEDS:
            row = {'seed': seed, 'available': seed in usable}
            if seed in usable:
                row.update({metric: (offset_lookup[(seed, a, 20)]['test_'+field]-offset_lookup[(seed, b, 20)]['test_'+field])/denominator
                            for metric, (field, denominator) in score_fields.items()})
            per_seed.append(row)
        paired_q20[name] = {'total_seed_units': 5, 'paired_available_seeds': usable,
            'unavailable_seeds': [seed for seed in SEEDS if seed not in usable],
            'per_seed': per_seed,
            'differences': {metric: statistics([offset_lookup[(seed, a, 20)]['test_'+field]-offset_lookup[(seed, b, 20)]['test_'+field]
                                               for seed in usable], denominator)
                            for metric, (field, denominator) in score_fields.items()}}
    return {'boundary': 7, 'independent_seed_units': 5, 'seeds': list(SEEDS), 'old_new_weights': [7/8, 1/8],
        'methods': methods, 'paired_raw_effects': effects, 'restriction_interaction': interaction,
        'H1': h1, 'H2': h2, 'score_targets': list(TARGETS), 'score_margin_pp': 3,
        'score_split_seed_base': 2026110800, 'score_calibration_n': 1600, 'score_test_n': 6400,
        'unadjusted_test': unadjusted_test, 'adjusted_test': adjusted, 'paired_q20': paired_q20}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('recomputed_task7'))
    args = parser.parse_args()
    raw, offsets, unadjusted = calculate()
    summary = summarize(raw, offsets, unadjusted)
    assert summary == json.loads((ROOT / 'summary.json').read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fields = [key for key in raw[0] if key in ('seed', 'method') or key.endswith('_correct') or key.endswith('_n')]
    write_csv(output / 'counts.csv', [{key: row[key] for key in fields} for row in raw])
    write_csv(output / 'seed_scores.csv', raw)
    write_csv(output / 'unadjusted_scores.csv', unadjusted)
    write_csv(output / 'score_counts.csv', offsets)
    write_json(output / 'summary.json', summary)
    print(json.dumps({'status':'passed','seed_units':5,'raw_endpoints':len(raw),'offset_attempts':len(offsets),'H1':summary['H1']['supported'],'H2':summary['H2']['supported']}))
if __name__ == '__main__':
    main()
