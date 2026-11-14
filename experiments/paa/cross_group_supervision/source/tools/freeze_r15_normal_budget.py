"""Seal final R15 protocol, implementation and analysis before scientific runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.r15_normal_budget.core import SETTINGS, REQUIRED_SOURCES, SCHEMA, ARCHIVE_SHA256, validate_manifest


def freeze(output):
    if output.exists():
        raise FileExistsError('Frozen manifests are immutable; use a new version')
    sources = set(REQUIRED_SOURCES) | {
        'protocol/R15_NORMAL_BUDGET.md', 'analysis/analyze_r15_normal_budget.py',
        'tools/build_r15_evidence.py', 'tools/freeze_r15_normal_budget.py',
        'tools/finalize_r15_statistics.py', 'tests/test_r15_analysis.py',
    }
    manifest = {'schema': SCHEMA, 'status': 'frozen',
                'frozen_utc': datetime.now(timezone.utc).isoformat(),
                'settings': SETTINGS, 'job_timeout_seconds': 5400,
                'dataset': {'archive_sha256': ARCHIVE_SHA256},
                'sources_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)},
                'scientific_role': 'Fixed external long-budget falsification on previously accessed CIFAR-100 test; not an unseen-dataset confirmation.',
                'analysis_unit': 'Five paired seeds, equal average of diagnostic tasks 6,7,8,9 before seed aggregation.',
                'strict_gate': 'Every seed: d_actual<0 AND d_probe>0 AND d_masked>0; exact integer-derived signs; zeros fail.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    validate_manifest(output)
    print(json.dumps({'path': output.as_posix(), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'sources': len(sources), 'frozen_utc': manifest['frozen_utc']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    freeze(p.parse_args().output)
