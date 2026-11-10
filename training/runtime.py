"""Training logs, project-local provenance and checkpoint validation."""
import datetime,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_ORIGIN='defense.training.run'
PREVIOUS_PUBLIC_ORIGIN_SHA256='11e46958ab07e4328152898edfeca9e86b27dbd5360fe6c798d357b7244e4187'
# Retain validation for checkpoints produced before the public package was named.
LEGACY_ORIGIN_SHA256='05e7009cccffbf0b98ab905f24346c591710647a54c98728dcd81b6184195986'

def example_seed(seed,epoch,position):
    return (int(seed)+1_000_003*int(epoch)+97_409*int(position))%(2**31-1)

class EpochPermutationSampler(torch.utils.data.Sampler):
    """Keep sample order independent of worker-base-seed generator consumption."""
    def __init__(self,data_source,seed):self.size=len(data_source);self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        return iter(torch.randperm(self.size,generator=generator).tolist())
    def __len__(self):return self.size

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_fingerprint():
    dirs=['models','losses','datasets','training','attacks','evaluation']
    files=[p for d in dirs for p in (ROOT/d).glob('*.py')]
    return hashlib.sha256(''.join(str(p.relative_to(ROOT))+sha256(p) for p in sorted(files)).encode()).hexdigest()

def jsonify(x):
    if isinstance(x,torch.Tensor):return x.detach().float().cpu().item() if x.numel()==1 else x.detach().float().cpu().tolist()
    if isinstance(x,dict):return {k:jsonify(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonify(v) for v in x]
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jsonify(obj),indent=2,allow_nan=False));tmp.replace(path)

def append_jsonl(path,obj):
    with Path(path).open('a') as f:f.write(json.dumps(jsonify(obj),allow_nan=False)+'\n');f.flush()

def utcnow():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save_checkpoint(path,model,optimizer,scaler,epoch,cfg,stage,run_id,best_metric,complete=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
        'epoch':epoch,'config':cfg,'stage':stage,'run_id':run_id,'best_validation_metric':best_metric,
        'provenance':{'generated_by':CHECKPOINT_ORIGIN,'schema':1,'code_sha256':cfg.get('_code_sha256',code_fingerprint()),
        'created_utc':utcnow(),'seed':cfg['seed'],'category':cfg.get('category','defense_training'),'complete':complete},
        'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
               'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    write_json(str(path)+'.json',{'sha256':sha256(path),'epoch':epoch,'stage':stage,'run_id':run_id,'provenance':payload['provenance']})

def load_checkpoint(path,allow_smoke=False):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT/'checkpoints') or p.is_relative_to(ROOT/'checkpoints/public'):
        raise ValueError('Defense checkpoints must originate from this project')
    sidecar=Path(str(p)+'.json')
    if not sidecar.exists():raise ValueError('Missing checkpoint provenance sidecar')
    meta=json.loads(sidecar.read_text())
    if meta['sha256']!=sha256(p):raise ValueError('Checkpoint integrity mismatch')
    data=torch.load(p,map_location='cpu')
    origin=data.get('provenance',{}).get('generated_by')
    if origin!=CHECKPOINT_ORIGIN and (not isinstance(origin,str) or hashlib.sha256(origin.encode('utf-8')).hexdigest() not in (LEGACY_ORIGIN_SHA256, PREVIOUS_PUBLIC_ORIGIN_SHA256)):
        raise ValueError('Unknown checkpoint origin')
    if data['provenance']['category']=='smoke' and not allow_smoke:
        raise ValueError('Smoke checkpoints cannot be used as formally trained models')
    return data
