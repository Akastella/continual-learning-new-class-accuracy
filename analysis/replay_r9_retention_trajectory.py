"""Independent functional forward of every fixed seed201 test checkpoint."""
import argparse,hashlib,json,os,time,platform
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def forward(x,s):
    for i in [0,3,6]:x=F.max_pool2d(F.relu(F.conv2d(x,s[f'backbone.{i}.weight'],s[f'backbone.{i}.bias'],padding=1)),2)
    return F.linear(F.relu(F.linear(x.flatten(1),s['backbone.10.weight'],s['backbone.10.bias'])),s['head.weight'],s['head.bias'])
def main(out):
    started=time.monotonic();d=ROOT/'results/r9_retention_trajectory';m=read(d/'frozen_manifest.json')
    if out.exists():raise FileExistsError(out)
    assert sha(__file__)==m['source_sha256']['analysis/replay_r9_retention_trajectory.py']
    data=ROOT/'data/r9_cifar101/cifar10.1_v6_data.npy';lab=ROOT/'data/r9_cifar101/cifar10.1_v6_labels.npy'
    for p in [data,lab]:assert sha(p)==m['input_sha256'][p.relative_to(ROOT).as_posix()]
    pixels=np.load(data,allow_pickle=False);labels=np.load(lab,allow_pickle=False);ids=[]
    for c in range(10):
        order=np.random.Generator(np.random.PCG64(20260915+c)).permutation(np.flatnonzero(labels==c));ids.extend(order[100:])
    ids=np.sort(ids).astype(np.int64)
    x=torch.from_numpy(pixels[ids]).permute(0,3,1,2).float()/255
    x=(x-torch.tensor([.4914,.4822,.4465]).reshape(1,3,1,1))/torch.tensor([.2470,.2435,.2616]).reshape(1,3,1,1)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for method in ['sgd','er']:
        run=f'split_cifar10_classil_{method}_s201';r=read(d/run/'test.json');assert sha(d/run/'test.npz')==r['npz_sha256']
        with np.load(d/run/'test.npz',allow_pickle=False) as z:
            assert np.array_equal(z['indices'],ids)
            for rec in r['records']:
                cp=ROOT/rec['checkpoint'];assert sha(cp)==rec['checkpoint_sha256'];s=torch.load(cp,map_location='cpu',weights_only=True)
                if rec['key']!='actual':s=s['model']
                s={k:v.cuda() for k,v in s.items()};parts=[]
                with torch.no_grad():
                    for b in range(0,1000,512):parts.append(forward(x[b:b+512].cuda(),s).cpu().numpy())
                logits=np.concatenate(parts);saved=z['logits_'+rec['key']]
                if not np.array_equal(logits,saved):raise ValueError((run,rec['key'],float(np.abs(logits-saved).max())))
                rows.append({'run':run,'key':rec['key'],'checkpoint_sha256':sha(cp),'image_evaluations':1000,'logit_elements':10000,'max_absolute_error':0})
    result={'status':'independent_functional_forward_passed','manifest_sha256':sha(d/'frozen_manifest.json'),'source_sha256':sha(__file__),'scope':'all actually test-scored fixed seed201 checkpoints for both methods; no training replay','environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'threads':2,'tf32':False},'rows':rows,'elapsed_seconds':time.monotonic()-started}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'checkpoints':len(rows),'seconds':result['elapsed_seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
