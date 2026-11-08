import copy
import pytest
import torch
import numpy as np
from experiments.r9_retention_trajectory.core import epoch_order, candidate_steps, train_trajectory, state_hash, cifar_images_from_flat


def config():
    return {'benchmark':'split_cifar10_classil','seed':201,'order_seed':30201,
            'batch_size':3,'epochs':4,'lr':.01}


def toy():
    torch.manual_seed(91)
    return torch.nn.Linear(2,10),torch.randn(10,2),torch.arange(10,dtype=torch.int64)


def test_orders_and_real_grid():
    c=config()
    assert torch.equal(epoch_order(10,c,4,2),torch.randperm(10,generator=torch.Generator().manual_seed(3020100000+402)))
    c['batch_size']=64
    assert candidate_steps(10000,c)==[0,1,2,4,8,16,32,64,128,157,314,471,628]


def test_training_memory_layout_matches_torchvision_pipeline():
    flat=np.arange(2*3072,dtype=np.uint8).reshape(2,3072)
    reference=torch.as_tensor(flat.reshape(-1,3,32,32).transpose(0,2,3,1)).clone()
    actual=cifar_images_from_flat(flat)
    assert torch.equal(reference,actual)
    assert actual.stride()==reference.stride()==(3072,32,1,1024)
    assert actual.permute(0,3,1,2).float().is_contiguous()


def test_matches_plain_sgd_and_callbacks_are_rng_mode_neutral(tmp_path):
    m,x,y=toy(); baseline=copy.deepcopy(m); c=config()
    opt=torch.optim.SGD(baseline.parameters(),lr=.01)
    for e in range(4):
        ids=epoch_order(10,c,4,e)
        for start in range(0,10,3):
            ix=ids[start:start+3];opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(baseline(x[ix]),y[ix]).backward();opt.step()
    calls=[]
    def callback(step,model,path):
        assert not model.training and not torch.is_grad_enabled()
        torch.rand(100);calls.append(step)
    rng=torch.random.get_rng_state().clone()
    result=train_trajectory(m,x,y,c,4,tmp_path,candidate_steps(10,c),callback)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert state_hash(m)==state_hash(baseline)
    assert calls==candidate_steps(10,c)
    assert result['optimizer_steps_executed']==16


def test_resume_exact(tmp_path):
    m,x,y=toy(); c=config(); other=copy.deepcopy(m)
    train_trajectory(m,x,y,c,4,tmp_path/'full',candidate_steps(10,c))
    train_trajectory(other,x,y,c,4,tmp_path/'part',candidate_steps(10,c),stop_step=5)
    result=train_trajectory(other,x,y,c,4,tmp_path/'resume',candidate_steps(10,c),resume=tmp_path/'part/step_000005.pt')
    assert result['optimizer_steps_executed']==11
    assert state_hash(m)==state_hash(other)


def test_callback_mutation_rejected_and_restored(tmp_path):
    m,x,y=toy(); before=state_hash(m)
    def bad(step,model,path): model.weight.add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        train_trajectory(m,x,y,config(),4,tmp_path,candidate_steps(10,config()),bad)
    assert state_hash(m)==before


def test_resume_mismatch_rejected(tmp_path):
    m,x,y=toy();c=config()
    train_trajectory(m,x,y,c,4,tmp_path/'first',candidate_steps(10,c),stop_step=5)
    c['lr']=.02
    with pytest.raises(ValueError,match='identity'):
        train_trajectory(m,x,y,c,4,tmp_path/'bad',candidate_steps(10,c),resume=tmp_path/'first/step_000005.pt')
