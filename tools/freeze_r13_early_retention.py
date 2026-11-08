"""Exclusive source and input freeze before the R13 experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_r13_early_retention as r

SOURCES = [
    'tools/run_r13_early_retention.py', 'tools/freeze_r13_early_retention.py',
    'tools/prepare_r13_external_data.py',
    'tools/measure_r13_trainonly.py',
    'experiments/r13_early_retention/__init__.py', 'experiments/r13_early_retention/core.py',
    'analysis/audit_r13_early_retention.py', 'analysis/replay_r13_early_retention.py',
    'tests/test_r13_retention_policy.py', 'tests/test_r13_retention_audit.py',
    'tests/test_r13_early_runner.py',
    'protocol/R13_EARLY_RETENTION_PROTOCOL.md', 'requirements-r9-windows-lock.txt',
]


def main():
    if (ROOT / r.MANIFEST).exists() or (ROOT / r.OUTPUT).exists():
        raise FileExistsError('R13 freeze/output already exists; preserve it')
    witness = r.read(ROOT / 'audit/r13_early_retention/trainonly_resource/RESOURCE.json')
    if witness['status'] != 'passed_train_only_prefix_and_resource_witness' or not witness['within_3300_seconds']:
        raise ValueError('Resource witness failed')
    for name, h in witness['source_sha256'].items():
        if r.sha(ROOT / name) != h:
            raise ValueError('Source changed since resource witness: ' + name)
    source = {}
    # Original sources retain their original pins; unchanged imported modules
    # are included even where Python only uses a subset of their functions.
    for name in [r.ORIGINAL, r.OLD_MANIFEST]:
        for p, h in r.read(ROOT / name)['source_sha256'].items():
            if p in source and source[p] != h:
                raise ValueError('Conflicting frozen source: ' + p)
            if r.sha(ROOT / p) != h:
                raise ValueError('Historical source drift: ' + p)
            source[p] = h
    source.update({p: r.sha(ROOT / p) for p in SOURCES})
    m = dict(settings=r.settings('images', 'labels'), source_sha256=source,
        input_sha256={p: r.sha(ROOT / p) for p in r.input_names()}, frozen_utc=r.utc(),
        disclosure='Post-hoc study motivated by prior CIFAR10.1 results; reused validation; first CIFAR10.2 model access after this freeze and global selection seal; no preregistration or original-claim confirmation')
    r.validate(m)
    r.save(ROOT / r.MANIFEST, m)
    print({'manifest_sha256': r.sha(ROOT / r.MANIFEST), 'frozen_utc': m['frozen_utc'],
           'inputs': len(m['input_sha256']), 'sources': len(source)})


if __name__ == '__main__':
    main()
