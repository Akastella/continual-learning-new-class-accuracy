import numpy as np
import pytest
from experiments.r9_retention_trajectory.evaluation import split_indices,choose,scores

def test_fixed_global_disjoint_balanced_split():
    y=np.tile(np.arange(10),200);v,t=split_indices(y)
    assert len(set(v)&set(t))==0
    assert np.array_equal(np.sort(np.r_[v,t]),np.arange(2000))
    assert np.array_equal(np.bincount(y[v]),np.full(10,100))
    assert np.array_equal(v,split_indices(y)[0])
    assert not np.array_equal(v,split_indices(y[::-1])[0])

def test_selection_rejects_better_current_when_old_fails_and_ties_earliest():
    rows=[{'step':s,'scores':{'old_correct':o,'old_n':800,'current_correct':c}} for s,o,c in [(0,200,0),(1,160,70),(2,160,70),(4,159,200)]]
    assert choose(rows,20)==1
    assert choose(rows,10)==4
    assert choose(rows,30) is None

def test_selection_exact_boundary_no_float_tolerance():
    rows=[{'step':1,'scores':{'old_correct':79,'old_n':800,'current_correct':200}},{'step':2,'scores':{'old_correct':80,'old_n':800,'current_correct':1}}]
    assert choose(rows,10)==2

def test_scores_use_global_candidates_including_current_outputs():
    y=np.repeat(np.arange(10),100).astype(np.int64);z=np.zeros((1000,10),np.float32);z[:,9]=1
    r=scores(z,y);assert r['old_pct']==0 and r['current_pct']==50 and r['all_pct']==10
    assert r['old_task_pct']==[0,0,0,0]
    z[0,0]=np.nan
    with pytest.raises(ValueError):scores(z,y)
