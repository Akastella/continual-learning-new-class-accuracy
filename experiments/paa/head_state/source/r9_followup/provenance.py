"""Completion requires all recorded artifacts, not only a status string."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def validate_manifest(manifest):
    runs=manifest['runs']
    if not runs or manifest['run_count']!=len(runs): raise ValueError('Manifest has invalid run count')
    for key in ('id','config_path','output'):
        if len({r[key] for r in runs})!=len(runs): raise ValueError(f'Manifest has duplicate {key}')


def required_paths(result):
    paths={'run_started.json','data_manifest.json','initial.pt','result.json'}
    paths.update('source/'+name for name in result['source_sha256'])
    for t in range(result['config']['tasks']):
        paths.add(f'task_{t:03d}.json')
        for prefix in ('pre','post','fresh','probe'): paths.add(f'{prefix}_task_{t:03d}.pt')
        if result['config'].get('release_current_weight_control',False):
            paths.update(f'{prefix}_task_{t:03d}.pt' for prefix in ('weighted_fresh','weighted_probe','compute_fresh','compute_probe'))
            if result['config']['method']=='er':paths.add(f'compensated_er_task_{t:03d}.pt')
        if result['config']['method'] in ('er','derpp'):
            paths.update(f'{prefix}_task_{t:03d}_memory.pt' for prefix in ('pre','post'))
    return paths


def write_artifact_manifest(output):
    result=json.loads((output/'result.json').read_text())
    doc={'schema':1,'files':{name:file_hash(output/name) for name in sorted(required_paths(result))}}
    path=output/'artifacts.json'; temp=output/'artifacts.json.tmp'
    temp.write_text(json.dumps(doc,indent=2),encoding='utf-8'); temp.replace(path)


def validate_finished_run(output,config,source_sha256):
    output=Path(output); result=json.loads((output/'result.json').read_text())
    if result.get('status')!='complete' or result.get('config')!=config: raise ValueError('Incomplete or mismatched result/config')
    expected={Path(name).name:h for name,h in source_sha256.items()}
    if result.get('source_sha256')!=expected: raise ValueError('Result source hashes differ from frozen source')
    artifacts=json.loads((output/'artifacts.json').read_text())
    if artifacts.get('schema')!=1 or set(artifacts['files'])!=required_paths(result): raise ValueError('Artifact inventory incomplete')
    for name,h in artifacts['files'].items():
        if file_hash(output/name)!=h: raise ValueError(f'Artifact hash mismatch: {name}')
    for name,h in expected.items():
        if artifacts['files']['source/'+name]!=h: raise ValueError('Archived source differs from freeze')
    if [t['task'] for t in result['tasks']]!=list(range(config['tasks'])): raise ValueError('Task sequence incomplete')
    for t,row in enumerate(result['tasks']):
        if json.loads((output/f'task_{t:03d}.json').read_text())!=row: raise ValueError('Task evidence differs from result')
    return result
