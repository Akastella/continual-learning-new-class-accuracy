"""Fixed CIFAR-10.1 split and validation-only checkpoint selection."""
from pathlib import Path
import numpy as np
import torch

def split_indices(labels):
    labels=np.asarray(labels)
    if labels.shape!=(2000,) or labels.dtype.kind not in 'iu' or not np.array_equal(np.bincount(labels),np.full(10,200)):
        raise ValueError('Expected balanced v6 labels')
    validation=[];test=[]
    for c in range(10):
        ids=np.flatnonzero(labels==c)
        ids=np.random.Generator(np.random.PCG64(20260915+c)).permutation(ids)
        validation.extend(ids[:100]);test.extend(ids[100:])
    return np.sort(validation).astype(np.int64),np.sort(test).astype(np.int64)

def external_data(root):
    root=Path(root)
    images=np.load(root/'cifar10.1_v6_data.npy',allow_pickle=False)
    labels=np.load(root/'cifar10.1_v6_labels.npy',allow_pickle=False)
    if images.shape!=(2000,32,32,3) or images.dtype!=np.uint8:raise ValueError('v6 image schema')
    val,test=split_indices(labels)
    x=torch.from_numpy(images).permute(0,3,1,2).float()/255
    x=(x-torch.tensor([.4914,.4822,.4465]).view(1,3,1,1))/torch.tensor([.2470,.2435,.2616]).view(1,3,1,1)
    return x,labels.astype(np.int64),val,test

def scores(logits,labels):
    logits=np.asarray(logits);labels=np.asarray(labels)
    if logits.shape!=(1000,10) or logits.dtype!=np.float32 or not np.isfinite(logits).all():raise ValueError('logit schema')
    if labels.shape!=(1000,) or labels.dtype!=np.int64 or not np.array_equal(np.bincount(labels),np.full(10,100)):raise ValueError('mapped label schema')
    pred=logits.argmax(1);correct=[int(np.sum((labels==c)&(pred==labels))) for c in range(10)]
    return {'class_correct':correct,'class_n':[100]*10,'old_correct':sum(correct[:8]),'old_n':800,'current_correct':sum(correct[8:]),'current_n':200,'old_pct':sum(correct[:8])/8,'current_pct':sum(correct[8:])/2,'all_pct':sum(correct)/10,'old_task_pct':[(correct[i]+correct[i+1])/2 for i in range(0,8,2)]}

def choose(rows,threshold):
    feasible=[r for r in rows if 100*r['scores']['old_correct']>=threshold*r['scores']['old_n']]
    if not feasible:return None
    return min(feasible,key=lambda r:(-r['scores']['current_correct'],r['step']))['step']
