"""Replay source-group counts, independent of image contents and minibatch order."""
import json
import argparse
from pathlib import Path
import numpy as np
rows=[]
for seed in range(201,211):
    storage=np.random.default_rng(np.random.SeedSequence([seed+77001,1]))
    sampling=np.random.default_rng(np.random.SeedSequence([seed+77001,2]))
    memory=np.full(500,-1,dtype=int); n=seen=0
    for task in range(5):
        current=old=0
        for epoch in range(4):
            for size in [64]*156+[16]:
                if n:
                    ids=sampling.choice(n,size=min(32,n),replace=False)
                    current+=int((memory[ids]==task).sum());old+=int((memory[ids]<task).sum())
                for _ in range(size):
                    seen+=1
                    j=n if n<500 else int(storage.integers(seen))
                    if j<500:
                        memory[j]=task
                        if j==n:n+=1
        assert current+old==(20064 if task==0 else 20096)
        rows.append(dict(seed=seed,task=task,scheduled_current=40000,replayed_current=current,replayed_old=old,total_current=40000+current))
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('replay_presentations.json'))
path=parser.parse_args().output
path.write_text(json.dumps(rows,indent=2)+'\n')
for t in range(5):
    v=[r for r in rows if r['task']==t]
    print(t,{k:float(np.mean([r[k] for r in v])) for k in ['scheduled_current','replayed_current','replayed_old','total_current']})
