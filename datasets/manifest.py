"""Read-only raw asset loading. No old dataset classes or code are imported."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def read_yolo(path,size,source_class,allow_outside=True):
    """Explicit source classes: person=0, patch=1; new detector positive=1."""
    w,h=size
    boxes=[]
    for lineno,line in enumerate(Path(path).read_text().splitlines(),1):
        fields=line.split()
        if not fields:continue
        if len(fields)!=5:raise ValueError(f'{path}:{lineno}: expected class cx cy w h')
        c,cx,cy,bw,bh=map(float,fields)
        if c!=int(c):raise ValueError(f'nonintegral class: {path}')
        if int(c)!=source_class:continue
        if not np.isfinite([cx,cy,bw,bh]).all() or bw<=0 or bh<=0:
            raise ValueError(f'invalid box: {path}:{lineno}')
        box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
        # Do not repair/clip original GT silently. Tolerate serialization at boundary.
        # Amodal CityPersons GT can legitimately extend beyond image bounds.
        # Preserve original full boxes for evaluation instead of clipping labels.
        if not allow_outside and (min(box)<-.01 or box[2]>w+.01 or box[3]>h+.01):
            raise ValueError(f'out-of-bounds GT: {path}:{lineno} {box} for {size}')
        boxes.append(box)
    return torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)


def read_rgb(path,size=None):
    with Image.open(path) as im:
        im=im.convert('RGB')
        original=im.size
        if size:im=im.resize((size,size),Image.Resampling.BILINEAR)
        image=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1)/255.
    return image,original


class ManifestDataset(Dataset):
    def __init__(self,path,size=416,require_clean=False,task='patch'):
        self.path=Path(path)
        self.records=load_jsonl(path)
        # Public manifests use paths relative to their own location. Absolute
        # caller-provided paths remain supported, without embedding any here.
        for record in self.records:
            for key in ['image','clean','patch_labels','person_labels']:
                if record.get(key) and not Path(record[key]).is_absolute():record[key]=str((self.path.resolve().parent/record[key]).resolve())
        self.size,self.require_clean,self.task=size,require_clean,task
        if not self.records:raise ValueError(f'Empty manifest {path}')
        ids=[r['id'] for r in self.records]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        x,original=read_rgb(r['image'],self.size)
        label_key='patch_labels' if self.task=='patch' else 'person_labels'
        label_path=r.get(label_key)
        if self.task=='person' and r.get('person_gt_available') is False:
            raise ValueError(f'Patch-only annotation is not person GT: {r["id"]}')
        inline_key='patch_boxes_xyxy' if self.task=='patch' else 'person_boxes_xyxy'
        if inline_key in r:
            boxes=torch.tensor(r[inline_key],dtype=torch.float32).reshape(-1,4)
        elif label_path:boxes=read_yolo(label_path,original,1 if self.task=='patch' else 0)
        else:raise ValueError(f'Missing {label_key}: {r["id"]}')
        boxes=boxes*boxes.new_tensor([self.size/original[0],self.size/original[1]]*2)
        target={'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),
                'image_id':torch.tensor(index),'iscrowd':torch.tensor(r.get('person_iscrowd',[0]*len(boxes)) if self.task=='person' else [0]*len(boxes),dtype=torch.int64),
                'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])}
        clean=None
        if r.get('clean'):
            clean,shape=read_rgb(r['clean'],self.size)
            if shape!=original:raise ValueError('Clean/attacked dimensions differ')
        elif self.require_clean:raise ValueError(f'No paired clean image: {r["id"]}')
        return {'image':x,'clean':clean,'target':target,'record':r}


def collate(batch):
    return {'image':torch.stack([b['image'] for b in batch]),
            'clean':torch.stack([b['clean'] for b in batch]) if all(b['clean'] is not None for b in batch) else None,
            'target':[b['target'] for b in batch], 'record':[b['record'] for b in batch]}
