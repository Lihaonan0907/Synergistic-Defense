"""Strict, portable complete-defense loading; labels never enter this API."""
import json
from pathlib import Path
import torch
from models.localizer import SpectralPatchLocalizer
from models.repair import DualDomainRepair
from models.detectors import build_detector
from models.defense import DefenseSystem,boxes_to_mask
from utils.io import sha256

ROOT=Path(__file__).resolve().parents[1]

def load_settings(config=None):
    path=Path(config) if config else ROOT/'configs/inference.json'
    settings=json.loads(path.read_text())
    expected={'size':416,'mask_threshold':.25,'detector_score_threshold':.001,'detector_nms_threshold':.6,'max_detections':100}
    if any(settings.get(k)!=v for k,v in expected.items()):raise ValueError('Config changes the frozen published inference/evaluation protocol')
    return settings

def load_model(path,device='cpu',expected_stage=None):
    path=Path(path);sidecar=json.loads(Path(str(path)+'.json').read_text())
    if sha256(path)!=sidecar['sha256']:raise ValueError('Weight SHA256 mismatch')
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    if bundle.get('schema')!='tdsc_inference_bundle_v1':raise ValueError('Expected exported inference bundle, not a training checkpoint')
    if expected_stage and bundle['stage']!=expected_stage:raise ValueError('Wrong weight stage')
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in bundle['model'].values()):raise ValueError('Nonfinite weight tensor')
    c=bundle['config'];localizer=SpectralPatchLocalizer(**c['localizer'])
    if bundle['stage']=='stage1':model=localizer
    elif bundle['stage']=='stage2':model=DefenseSystem(localizer,DualDomainRepair(**c['repair']),build_detector(**c['person']),c['mask_threshold'])
    else:raise ValueError('Unknown weight stage')
    model.load_state_dict(bundle['model'],strict=True);model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,bundle

class DefensePredictor:
    def __init__(self,stage2_weights=None,device='cpu',stage1_weights=None,config=None):
        self.settings=load_settings(config);self.device=torch.device(device)
        self.stage2_path=Path(stage2_weights) if stage2_weights else ROOT/self.settings['stage2_weights']
        self.model,self.bundle=load_model(self.stage2_path,self.device,'stage2');self.size=self.bundle['config']['size']
        if self.size!=self.settings['size'] or self.bundle['config']['mask_threshold']!=self.settings['mask_threshold']:raise ValueError('Bundle and inference config disagree')
        self.stage1_path=Path(stage1_weights) if stage1_weights else ROOT/self.settings['stage1_weights']
        localizer,stage1=load_model(self.stage1_path,'cpu','stage1')
        frozen=self.model.localizer.state_dict()
        if set(frozen)!=set(stage1['model']) or any(not torch.equal(v.detach().cpu(),stage1['model'][k]) for k,v in frozen.items()):raise ValueError('Stage1 is incompatible with the complete Stage2 frozen localizer')
        del localizer,stage1
    @torch.inference_mode()
    def __call__(self,rgb):
        if rgb.ndim==3:rgb=rgb[None]
        if rgb.ndim!=4 or rgb.shape[1]!=3 or rgb.shape[-2:]!=(self.size,self.size):raise ValueError('Expected Bx3x416x416 normalized RGB')
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:raise ValueError('Expected finite floating RGB in [0,1]')
        x=rgb.to(self.device);patch=self.model.localizer.predict(x)
        mask=boxes_to_mask(patch,x.shape[-2:],self.model.mask_threshold);repaired=self.model.repair(x,mask)
        person=self.model.person.predict(repaired)
        if not torch.isfinite(repaired).all() or any(not torch.isfinite(p[k]).all() for p in [*patch,*person] for k in ['boxes','scores']):raise FloatingPointError('Nonfinite inference output')
        return {'patch':patch,'mask':mask,'repaired':repaired,'person':person}
