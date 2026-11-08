"""Run the fixed matrix and deterministic recomputation, with no paper claims."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(manifest, results, analysis, generated):
    args = [sys.executable, '-s', '-X', 'utf8']
    env = os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    subprocess.run(args + ['tools/run_r15_normal_budget.py', '--run', '--manifest', str(manifest), '--output', str(results)],
                   cwd=ROOT, env=env, check=True)
    batch = json.loads((results/'batch.json').read_text(encoding='utf-8'))
    if batch['status'] != 'complete':
        raise RuntimeError('Fixed experiment matrix incomplete; no statistical finalization')
    subprocess.run(args + ['analysis/analyze_r15_normal_budget.py', '--results', str(results), '--output', str(analysis)],
                   cwd=ROOT, env=env, check=True)
    subprocess.run(args + ['tools/build_r15_evidence.py', '--analysis', str(analysis/'analysis.json'), '--output', str(generated)],
                   cwd=ROOT, env=env, check=True)
    marker = {'status': 'raw_and_statistics_complete_pending_independent_audit',
              'utc': datetime.now(timezone.utc).isoformat(), 'results': str(results),
              'analysis': str(analysis), 'generated': str(generated),
              'paper_changed': False, 'acceptance_claimed': False}
    (results/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(marker, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(marker, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--generated', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest.resolve(), a.results.resolve(), a.analysis.resolve(), a.generated.resolve())
