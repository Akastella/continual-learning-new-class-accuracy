"""Verify pinned official CIFAR-10.2 test bytes and exact image overlap, CPU only.

Default: read-only verification and JSON on stdout. --download-missing fetches
only the pinned README and test array when absent; it never overwrites a file.
No model, GPU, training set download, filtering, or data redistribution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '521467a52ad494b78ff9c70051099c0d9ba74d40'
FILES = {
    'README.md': ('third_party/r13_cifar102_reference/README.md',
                  'fa5d8ee410de8b99a892e6a0ad67706ef6e6aaac550f91e69c6542654fb08ebb'),
    'cifar102_test.npz': ('data/r13_cifar102/cifar102_test.npz',
                        'e4fd6462bd5141ed293acd52ea570952f65d1bfd4ec9b068f87742f75a7a632e'),
}
CIFAR_MD5 = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
}
CIFAR101_HASHES = {
    'cifar10.1_v6_data.npy': '2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6',
    'cifar10.1_v6_labels.npy': 'ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448',
}
NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def image_hash(image):
    require(image.shape == (32,32,3) and image.dtype == np.uint8, 'RGB byte schema')
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()

def fetch_missing():
    for name,(relative,expected) in FILES.items():
        target=ROOT/relative
        if target.exists():
            require(sha(target)==expected, 'Existing file hash mismatch: '+relative)
            continue
        url=f'https://raw.githubusercontent.com/modestyachts/cifar-10.2/{COMMIT}/{name}'
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-data-audit'}),timeout=40) as r:
                    data=r.read()
                require(hashlib.sha256(data).hexdigest()==expected,'Downloaded file hash mismatch: '+name)
                break
            except Exception:
                if attempt==2:
                    raise
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:
            f.write(data)

def main(download_missing=False):
    if download_missing:
        fetch_missing()
    inputs={}
    for name,(relative,expected) in FILES.items():
        inputs[relative]=sha(ROOT/relative)
        require(inputs[relative]==expected,'Pinned source hash mismatch: '+relative)
    source_path=ROOT/'third_party/r13_cifar102_reference/SOURCE.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    require(source['commit']==COMMIT,'Source commit mismatch')
    require(source['image_key']=='images' and source['label_key']=='labels','Source NPZ key mismatch')
    for r in source['downloaded']:
        require(r['name'] in FILES and r['sha256']==FILES[r['name']][1], 'Source metadata hash mismatch')
        require(r['url']==f'https://raw.githubusercontent.com/modestyachts/cifar-10.2/{COMMIT}/{r["name"]}', 'Nonpinned source URL')
    path=ROOT/FILES['cifar102_test.npz'][0]
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'images','labels','ti_indices','keywords','label_names'},'Unexpected NPZ keys')
        arrays={key:z[key] for key in z.files}
    images,labels=arrays['images'],arrays['labels']
    require(images.shape==(2000,32,32,3) and images.dtype==np.uint8,'Image schema mismatch')
    require(labels.shape==(2000,) and labels.dtype==np.int64,'Label schema mismatch')
    require(np.array_equal(np.unique(labels),np.arange(10)), 'Class identities mismatch')
    counts=np.bincount(labels,minlength=10)
    require(np.array_equal(counts,np.full(10,200)), 'Class balance mismatch')
    require(arrays['ti_indices'].shape==(2000,) and arrays['ti_indices'].dtype==np.int64,'TinyImages index schema')
    require(arrays['keywords'].shape==(2000,) and arrays['keywords'].dtype.kind=='U','Keyword schema')
    require(arrays['label_names'].shape==(10,) and arrays['label_names'].tolist()==NAMES,'Class names/order mismatch')
    # Shape/range check only: experiment code must separately preserve tensor
    # layout and exact FP32 operation order. No CNN inference is performed.
    normalized=images.transpose(0,3,1,2).astype(np.float32)/np.float32(255)
    normalized=(normalized-np.array([.4914,.4822,.4465],np.float32)[None,:,None,None])/np.array([.2470,.2435,.2616],np.float32)[None,:,None,None]
    require(normalized.shape==(2000,3,32,32) and normalized.dtype==np.float32 and np.isfinite(normalized).all(),'Preprocessing shape/range')
    index=defaultdict(list)
    for i,image in enumerate(images):
        index[image_hash(image)].append(i)
    internal=[{'sha256':h,'test_indices':ii,'labels':labels[ii].tolist()} for h,ii in index.items() if len(ii)>1]
    matches={name:[] for name in ('cifar10_train','cifar10_test','cifar101_v6')}
    def compare(name,x,y,offset=0):
        for j,image in enumerate(x):
            h=image_hash(image)
            for i in index.get(h,[]):
                # Verify exact equality even after digest match.
                require(np.array_equal(images[i],image),'SHA256 match without equal pixels')
                matches[name].append({'cifar102_test_index':i,'reference_index':offset+j,
                    'cifar102_label':int(labels[i]),'reference_label':int(y[j]),'image_sha256':h})
    train_count=0;test_count=0
    folder=ROOT/'data/r9_official/cifar-10-batches-py'
    for name,expected in CIFAR_MD5.items():
        p=folder/name;raw=p.read_bytes()
        require(hashlib.md5(raw).hexdigest()==expected,'Official CIFAR-10 checksum mismatch: '+name)
        inputs[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
        item=pickle.loads(raw,encoding='latin1')
        x=np.asarray(item['data']).reshape(-1,3,32,32).transpose(0,2,3,1)
        y=np.asarray(item['labels'],dtype=np.int64)
        require(x.shape==(10000,32,32,3) and y.shape==(10000,),'CIFAR batch schema')
        if name=='test_batch':
            compare('cifar10_test',x,y);test_count+=len(x)
        else:
            compare('cifar10_train',x,y,train_count);train_count+=len(x)
    for name,expected in CIFAR101_HASHES.items():
        p=ROOT/'data/r9_cifar101'/name;inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        require(inputs[p.relative_to(ROOT).as_posix()]==expected,'Official CIFAR-10.1 hash mismatch')
    x101=np.load(ROOT/'data/r9_cifar101/cifar10.1_v6_data.npy',allow_pickle=False)
    y101=np.load(ROOT/'data/r9_cifar101/cifar10.1_v6_labels.npy',allow_pickle=False)
    require(x101.shape==(2000,32,32,3) and y101.shape==(2000,),'CIFAR-10.1 schema')
    compare('cifar101_v6',x101,y101)
    return {'status':'pinned_bytes_schema_and_exact_pixel_overlap_verified','commit':COMMIT,
        'script_sha256':sha(Path(__file__)),'input_sha256':inputs,
        'npz_arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
        'class_counts':counts.tolist(),'class_names':NAMES,'tinyimages_unique_indices':int(len(np.unique(arrays['ti_indices']))),
        'normalized_check':{'shape':list(normalized.shape),'dtype':str(normalized.dtype),'min':float(normalized.min()),'max':float(normalized.max()),'all_finite':True,'layout_equivalence_to_experiment_not_asserted':True},
        'reference_images_checked':{'cifar10_train':train_count,'cifar10_test':test_count,'cifar101_v6':len(x101)},
        'exact_duplicate_groups_within_cifar102_test':internal,
        'cross_dataset_overlap':{name:{'pair_count':len(rows),'unique_cifar102_test_images':len({r['cifar102_test_index'] for r in rows}),'pairs':rows} for name,rows in matches.items()},
        'sample_filtering':'none; original 2000 images retained','model_inference_performed':False,
        'limits':['Byte-identical pixel overlap only; no perceptual, semantic, source-family independence or near-duplicate guarantee.',
                  'No model evaluation or assertion that this dataset was never previously viewed elsewhere.',
                  'Only official test archive downloaded; no original images redistributed.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-missing',action='store_true')
    args=parser.parse_args()
    print(json.dumps(main(args.download_missing),indent=2,ensure_ascii=False))
