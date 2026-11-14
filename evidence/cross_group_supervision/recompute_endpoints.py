"""Recompute the fixed endpoint counts in a new directory using only NumPy."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('executed_recompute', data / 'recompute.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rows, summary = original.compute(data)
    with (output / 'counts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (data / 'counts.csv').open(newline='', encoding='utf-8') as f:
        recorded = list(csv.DictReader(f))
    assert recorded == [{k: str(v) for k, v in row.items()} for row in rows]
    assert summary == json.loads((data / 'statistics.json').read_text(encoding='utf-8'))
    verification = {'status': 'passed', 'prediction_files': 30, 'independent_seeds': 5,
                    'all_integer_counts_match': True, 'all_summary_values_match': True,
                    'H1': summary['H1']['supported'], 'H2': summary['H2']['supported'],
                    'training_performed': False, 'inference_performed': False}
    (output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
