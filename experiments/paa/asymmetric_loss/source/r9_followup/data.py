"""Official splits, explicit task transforms, and content-addressed data provenance."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST


def digest_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_stream(config, root):
    name = config['benchmark']
    cifar = name.startswith('split_cifar10')
    class_order=np.arange(10)
    if cifar and config.get('class_order_seed') is not None:
        class_order=np.random.default_rng(config['class_order_seed']).permutation(10)
    inverse_order=np.argsort(class_order)
    cls = CIFAR10 if cifar else MNIST
    path = Path(root)/('r9_official' if cifar else '')
    tensors = {}
    provenance = {'dataset': 'CIFAR10' if cifar else 'MNIST', 'source': 'torchvision official train/test splits', 'splits': {},
                  'class_order_original_ids':class_order.tolist() if cifar else None,
                  'original_class_to_global_output':inverse_order.tolist() if cifar else None}
    for split, is_train in [('train', True), ('test', False)]:
        ds = cls(path, train=is_train, download=True)
        images = torch.as_tensor(np.asarray(ds.data)).clone()
        labels = torch.as_tensor(np.asarray(ds.targets)).long()
        provenance['splits'][split] = {'official_n': len(labels), 'images_sha256': digest_tensor(images), 'labels_sha256': digest_tensor(labels)}
        x = images.permute(0,3,1,2).float()/255 if cifar else images[:,None].float()/255
        mu = torch.tensor([.4914,.4822,.4465] if cifar else [.1307]).view(1,-1,1,1)
        sd = torch.tensor([.2470,.2435,.2616] if cifar else [.3081]).view(1,-1,1,1)
        tensors[split] = ((x-mu)/sd, labels)
    tasks = []
    seed = config.get('stream_seed',config['seed'])
    for t in range(config['tasks']):
        # Separate streams: varying one transformation never changes the other.
        pixel_rng = np.random.default_rng(np.random.SeedSequence([seed, t, 1101]))
        label_rng = np.random.default_rng(np.random.SeedSequence([seed, t, 2202]))
        pix = torch.arange(784)
        lab = torch.tensor(inverse_order,dtype=torch.long) if cifar else torch.arange(10)
        if not cifar and t:
            if name == 'permuted_mnist':
                pix = torch.tensor(pixel_rng.permutation(784),dtype=torch.long)
            elif name == 'target_remapping':
                lab = torch.tensor(label_rng.permutation(10),dtype=torch.long)
            elif name != 'stationary_mnist':
                raise ValueError(name)
        original_classes=class_order[2*t:2*t+2].tolist() if cifar else list(range(10))
        task = {'id': t, 'classes': [2*t,2*t+1] if cifar else list(range(10)), 'original_classes':original_classes,'pixel_permutation': pix.tolist() if not cifar else None,
                'label_permutation': lab.tolist(), 'splits': {}}
        for split in ('train','test'):
            x,y = tensors[split]
            ids = torch.where((y==original_classes[0])|(y==original_classes[1]))[0] if cifar else torch.arange(len(y))
            cap = config.get('max_'+split)
            if cap and len(ids)>cap:
                # Fixed dataset subsample, independent of model/stream seed; never selected by outcomes.
                selection_seed=config.get('data_seed',1729)+(0 if split=='train' else 1)+(t*10 if cifar else 0)
                g = torch.Generator().manual_seed(selection_seed)
                ids=ids[torch.randperm(len(ids),generator=g)[:cap]].sort().values
            tx,ty=x[ids],y[ids]
            if cifar:ty=torch.tensor(inverse_order,dtype=torch.long)[ty]
            if not cifar:
                tx=tx.reshape(-1,784)[:,pix].reshape(-1,1,28,28)
                ty=lab[ty]
            elif name=='split_cifar10_shared2':
                ty=ty%2
            task[split]=(tx,ty)
            task['splits'][split]={'n':len(ids),'official_indices':ids.tolist(),'images_sha256':digest_tensor(tx),'labels_sha256':digest_tensor(ty)}
        tasks.append(task)
    return tasks, provenance
