"""Compare one fixed five-point validation margin at the 30% retention target."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from recompute_score_frontier import all_names, loadval, calc, optimal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    chosen = []
    for seed in range(201, 211):
        for policy in ('raw30', 'offset_usual', 'offset30'):
            names = ['er_standard', 'sgd_standard'] if policy == 'offset_usual' else all_names()
            eligible = []
            for name in names:
                z, y = loadval(a.artifact, seed, name)
                b = optimal(z, y, 35) if policy.startswith('offset') else 0.
                if b is None: continue
                score = calc(z, y, b)
                if score['old'] >= 35 - 1e-10:
                    eligible.append((name, b, score['current']))
            winner = min(eligible, key=lambda r: (-r[2], r[0])) if eligible else None
            chosen.append({'seed': seed, 'policy': policy, 'target': 30, 'margin': 5,
                           'candidate': winner[0] if winner else None,
                           'scalar': winner[1] if winner else None})
    rows = []
    for dataset in ('cifar101', 'cifar102'):
        for choice in chosen:
            row = {'dataset': dataset, **choice, 'available': choice['candidate'] is not None,
                   'old': None, 'current': None, 'all': None, 'violation': None}
            if row['available']:
                filename = f"{dataset}_s{row['seed']}_{row['candidate']}.npz"
                file = a.data / 'diagnostics/test_logits' / filename
                if not file.exists(): file = a.data / 'test_logits' / filename
                with np.load(file, allow_pickle=False) as d:
                    score = calc(d['logits'].astype(float), d['labels'], row['scalar'])
                row.update(score); row['violation'] = row['old'] < 30 - 1e-10
            rows.append(row)
    with (a.output / 'test_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for dataset in ('cifar101', 'cifar102'):
        for policy in ('raw30', 'offset_usual', 'offset30'):
            selected = [r for r in rows if r['dataset'] == dataset and r['policy'] == policy and r['available']]
            s = {'dataset': dataset, 'policy': policy, 'available': len(selected),
                 'unavailable': 10 - len(selected), 'violations': sum(r['violation'] for r in selected),
                 'seed_ids': [r['seed'] for r in selected]}
            for key in ('old', 'current', 'all'):
                v = np.asarray([r[key] for r in selected], dtype=float)
                s[key] = {'values': v.tolist(), 'mean': float(v.mean()) if len(v) else None,
                          'sample_sd': float(v.std(ddof=1)) if len(v) > 1 else None}
            summaries.append(s)
    (a.output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps(summaries))


if __name__ == '__main__': main()
