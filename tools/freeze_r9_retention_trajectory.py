"""Exclusive pre-external-evaluation freeze."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_r9_retention_trajectory as r
if __name__=='__main__':
    witness=r.read(ROOT/'audit/r9/RETENTION_TRAJECTORY_RESOURCE.json')
    if witness['status']!='passed_train_only_resource_witness' or not witness['within_55min_estimate_with_margin']:raise ValueError('No feasible measured resource gate')
    witness_hashes={k.replace('\\','/'):v for k,v in witness['source_sha256'].items()}
    if witness_hashes['experiments/r9_retention_trajectory/core.py']!=r.sha(ROOT/'experiments/r9_retention_trajectory/core.py'):raise ValueError('Core changed after resource witness')
    source=dict(r.read(ROOT/r.ORIGINAL)['source_sha256'])
    source.update({n:r.sha(ROOT/n) for n in r.SOURCES})
    m={'settings':r.SETTINGS,'source_sha256':source,'input_sha256':{n:r.sha(ROOT/n) for n in sorted(r.inputs())},'frozen_utc':r.utc(),'disclosure':'Post-hoc question after prior CIFAR-10 results; no CIFAR-10.1 model inference before freeze; shared external image split; descriptive inference only'}
    r.validate(m);p=ROOT/r.MANIFEST;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(m,f,indent=2)
    print(json.dumps({'manifest_sha256':r.sha(p),'frozen_utc':m['frozen_utc'],'inputs':len(m['input_sha256'])}))
