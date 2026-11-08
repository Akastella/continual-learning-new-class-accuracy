"""Same four-epoch CIFAR-10 comparison using only seen outputs in the loss.

The original engine is reused unchanged. Inactive classifier outputs receive a
constant value, so they contribute neither probabilities nor parameter updates.
The output tensor remains ten-dimensional for the existing replay storage.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse,json,sys,hashlib,time
import torch

p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path)
p.add_argument('--seed',required=True,type=int);p.add_argument('--method',choices=['er','sgd'],required=True)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
sys.path.insert(0,str(a.project.resolve()))
from experiments.r9_followup import engine

OriginalNetwork=engine.Network
OriginalTrain=engine.train_branch
class SeenNetwork(OriginalNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.active_classes=2
    def forward(self,x):
        z=super().forward(x)
        if self.active_classes<10:
            z=z.clone();z[:,self.active_classes:]=-1e9
        return z
def train_seen(model,task,config,*args,**kwargs):
    model.active_classes=2*(task['id']+1)
    before=model.head.weight[model.active_classes:].detach().clone()
    before_bias=model.head.bias[model.active_classes:].detach().clone()
    result=OriginalTrain(model,task,config,*args,**kwargs)
    assert torch.equal(before,model.head.weight[model.active_classes:])
    assert torch.equal(before_bias,model.head.bias[model.active_classes:])
    return result
engine.Network=SeenNetwork;engine.train_branch=train_seen
cfg=json.loads((a.project/f'configs/r9_followup/split_cifar10_classil_{a.method}_s{a.seed}.json').read_text())
cfg.update(release_current_weight_control=False,run_type='seen_classes_training_control_20261107')
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
engine.execute(cfg,a.project/'data',a.output)
