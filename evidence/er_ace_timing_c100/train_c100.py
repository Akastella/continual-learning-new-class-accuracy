"""Test loss placement with the Split CIFAR-100 ER-ACE model and data stream."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR100
from torchvision.models.resnet import BasicBlock


SEEDS = range(7101,7106)
POLICIES = ("retrieval","same_step","expected_count","first_presentation","half_at_first")
TASK_SIZE = 5
BATCH = 10


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,20,3,padding=1,bias=False),
                                  nn.BatchNorm2d(20),nn.ReLU())
        stages = []
        previous = 20
        for i,width in enumerate((20,40,80,160)):
            stride = 1 if i == 0 else 2
            downsample = None
            if stride == 2:
                downsample = nn.Sequential(nn.Conv2d(previous,width,1,stride=2,bias=False),
                                           nn.BatchNorm2d(width))
            stages.extend((BasicBlock(previous,width,stride,downsample), BasicBlock(width,width)))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(160,100,bias=False)

    def forward(self,x):
        hidden = F.avg_pool2d(self.stages(self.stem(x)),4).flatten(1)
        weight = self.classifier.weight
        features = hidden / (hidden.norm(dim=1,keepdim=True)+1e-5)
        weights = weight / (weight.norm(dim=1,keepdim=True)+1e-5)
        return 10*F.linear(features,weights)


def log(**record):
    print(json.dumps(record),flush=True)


def data(root,device,train):
    source = CIFAR100(str(root),train=train,download=False)
    labels = np.asarray(source.targets,dtype=np.int64)
    images = torch.as_tensor(source.data,device=device).permute(0,3,1,2).float().div_(127.5).sub_(1)
    return images,torch.as_tensor(labels,device=device),labels


def plan(labels,seed,capacity):
    storage = np.random.default_rng(np.random.SeedSequence([seed,1]))
    samples = np.random.default_rng(np.random.SeedSequence([seed,2]))
    order = np.random.default_rng(seed+30000)
    members = np.empty(capacity,dtype=np.int64)
    seen, size = 0,0
    histories = []
    for task in range(20):
        start = task*TASK_SIZE
        inputs = np.flatnonzero((labels>=start)&(labels<start+TASK_SIZE))
        groups = order.permutation(inputs).reshape(-1,BATCH)
        draws = []
        weights = np.zeros(len(labels),dtype=np.float64)
        for row in groups:
            if size:
                replay = samples.choice(members[:size],size=BATCH,replace=False)
                now = replay[labels[replay]>=start]
                weights[now] += 1/BATCH
            else:
                replay = np.empty(0,dtype=np.int64)
            draws.append(replay)
            for index in row:
                seen += 1
                slot = size if size<capacity else storage.integers(seen)
                if slot < capacity:
                    members[slot] = index
                    if slot == size:
                        size += 1
        histories.append((groups,draws,weights))
    return histories


@torch.inference_mode()
def evaluate(model,test,task,path):
    images,labels,_ = test
    old_n = TASK_SIZE*task
    n_seen = old_n+TASK_SIZE
    rows = torch.where(labels<n_seen)[0]
    model.eval()
    z = torch.cat([model(images[batch]) for batch in rows.split(256)])
    if not task:
        np.savez_compressed(path,labels=labels[rows].cpu().numpy(),
                            predicted=z[:,:n_seen].argmax(1).cpu().numpy())
        return
    old_scores, old_pred = z[:,:old_n].max(1)
    new_scores, new_pred = z[:,old_n:n_seen].max(1)
    np.savez_compressed(path, labels=labels[rows].cpu().numpy(),
                        predicted=z[:,:n_seen].argmax(1).cpu().numpy(),
                        old_prediction=old_pred.cpu().numpy(),
                        new_prediction=(new_pred+old_n).cpu().numpy(),
                        old_score=old_scores.cpu().numpy(),new_score=new_scores.cpu().numpy(),
                        new_classes=np.arange(old_n,n_seen))


def run_task(model,optimizer,training,test,task,schedule,policy,output):
    x,y,labels = training
    device = x.device
    scheduled,retrieved,weights = schedule
    old_n = task*TASK_SIZE
    group = torch.arange(100,device=device)>=old_n
    seen_max = old_n-1
    measured = {}
    dose, realized = 0.,0.
    for j,(incoming,replay) in enumerate(zip(scheduled,retrieved),1):
        model.train()
        seen_max = max(seen_max,int(labels[incoming].max()))
        y_in = y[incoming]
        current = torch.arange(100,device=device)>=seen_max
        current[y_in] = True
        incoming_scores = model(x[incoming])
        loss = F.cross_entropy(incoming_scores.masked_fill(~current,-1e9),y_in)
        if len(replay):
            y_re = y[replay]
            replay_scores = model(x[replay])
            if not task or policy == "retrieval":
                loss += F.cross_entropy(replay_scores,y_re)
            else:
                now = y_re >= old_n
                allowed = group | (~now[:,None])
                loss += F.cross_entropy(replay_scores.masked_fill(~allowed,-1e9),y_re)
            if task:
                m = int((labels[replay]>=old_n).sum())
                realized += m/BATCH
                if policy == "retrieval":
                    dose += m/BATCH
                elif policy in ("same_step","expected_count"):
                    transfer = m/BATCH if policy == "same_step" else (j-1)/(task*len(scheduled)+j-1)
                    penalty = incoming_scores.logsumexp(1)-incoming_scores[:,old_n:].logsumexp(1)
                    loss += transfer*penalty.mean()
                    dose += transfer
                else:
                    early = 0.5 if policy == "half_at_first" else 1.
                    per_image = torch.as_tensor(weights[incoming],device=device,dtype=incoming_scores.dtype)
                    penalty = incoming_scores.logsumexp(1)-incoming_scores[:,old_n:].logsumexp(1)
                    late = replay_scores.logsumexp(1)-replay_scores[:,old_n:].logsumexp(1)
                    loss += early*(per_image*penalty).sum()
                    loss += (1-early)*((y_re>=old_n)*late).mean()
                    dose += early*float(weights[incoming].sum()) + (1-early)*m/BATCH
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if task and j in (25,len(scheduled)):
            evaluate(model,test,task,output/f"{j}.npz")
            measured[j] = dose
    if task and policy != "expected_count":
        assert abs(dose-realized)<1e-9,(dose,realized)
    return measured


def model_for(seed,device):
    torch.manual_seed(seed+10000)
    return SmallResNet().to(device,memory_format=torch.channels_last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--capacity",type=int,choices=(2000,10000),default=10000)
    parser.add_argument("--policies",nargs="+",choices=POLICIES,default=POLICIES)
    parser.add_argument("--first-task-only",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    train = data(args.data_root,device,True)
    test = data(args.data_root,device,False)
    args.output.mkdir(parents=True)
    schedules = {seed:plan(train[2],seed,args.capacity) for seed in SEEDS}
    if args.first_task_only:
        for seed in SEEDS:
            model = model_for(seed,device)
            optimizer = torch.optim.SGD(model.parameters(),lr=.1)
            run_task(model,optimizer,train,test,0,schedules[seed][0],"retrieval",args.output)
            evaluate(model,test,0,args.output/f"s{seed}_t1.npz")
        log(status="complete",histories=len(SEEDS))
        return

    trial = args.output / "runtime_trial"
    trial.mkdir()
    net = model_for(SEEDS.start,device)
    start = time.monotonic()
    run_task(net,torch.optim.SGD(net.parameters(),lr=.1),train,test,1,
             schedules[SEEDS.start][1],"half_at_first",trial)
    seconds = time.monotonic()-start
    log(status="trial_completed",seconds=seconds,estimated_condition_seconds=1.5*seconds*100)
    if seconds*150 >= 3000:
        raise TimeoutError("The measured task exceeds the 50-minute per-condition estimate.")
    del net

    records = []
    for policy in args.policies:
        start = time.monotonic()
        for seed in SEEDS:
            model = model_for(seed,device)
            optimizer = torch.optim.SGD(model.parameters(),lr=.1)
            for task,schedule in enumerate(schedules[seed]):
                if time.monotonic()-start >= 3000:
                    raise TimeoutError("The condition exceeded 50 minutes.")
                folder = args.output / f"s{seed}_t{task+1}_{policy}"
                if task:
                    folder.mkdir()
                doses = run_task(model,optimizer,train,test,task,schedule,policy,folder)
                if task:
                    records.append(dict(seed=seed,task=task+1,policy=policy,dose_by_update=doses))
            log(policy=policy,seed=seed,status="trained",seconds=time.monotonic()-start)
            with (args.output/"training.json").open("w",encoding="utf-8") as stream:
                json.dump(records,stream,indent=2)
        log(policy=policy,status="condition_complete",seconds=time.monotonic()-start)
    log(status="complete",histories=len(SEEDS)*len(args.policies))


if __name__ == "__main__":
    main()
