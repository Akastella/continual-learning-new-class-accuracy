"""Freeze explicit split indices using official labels only, without model/image access."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.r9_retention_trajectory.evaluation import split_indices
if __name__=='__main__':
    p=ROOT/'data/r9_cifar101/cifar10.1_v6_labels.npy';h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!='ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448':raise ValueError('official labels drift')
    v,t=split_indices(np.load(p,allow_pickle=False));dest=ROOT/'configs/r9_retention_trajectory/split.json';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:json.dump({'created_utc':datetime.now(timezone.utc).isoformat(),'labels_sha256':h,'model_forward_calls':0,'images_decoded':False,'rule':'PCG64(20260915+c) permutation of original-class indices; first100 validation, remaining100 test; each union sorted','validation_indices':v.tolist(),'test_indices':t.tolist()},f,indent=2)
    print(json.dumps({'validation':len(v),'test':len(t),'split_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
