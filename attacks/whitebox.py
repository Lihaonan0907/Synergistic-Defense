"""Eq1–3 defense-aware image-specific RGB-patch optimization. Not universal attack."""
import math
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def transform_patch(patch,params):
    angle,scale,brightness,offset=params
    a=angle*math.pi/180
    theta=patch.new_tensor([[math.cos(a)/scale,-math.sin(a)/scale,0],
                            [math.sin(a)/scale,math.cos(a)/scale,0]]).unsqueeze(0)
    p=(patch*brightness+offset).clamp(0,1)
    # Pad to fit rotated/scaled support; actual original optimized texture remains64².
    pad=patch.shape[-1]//4
    p=F.pad(p,(pad,)*4);alpha=F.pad(torch.ones_like(patch[:,:1]),(pad,)*4)
    grid=F.affine_grid(theta,p.shape,align_corners=False)
    return F.grid_sample(p,grid,align_corners=False),F.grid_sample(alpha,grid,align_corners=False)


def place_patch(image,patch,center,params=(0.,1.,1.,0.)):
    texture,alpha=transform_patch(patch,params)
    h,w=image.shape[-2:];ph,pw=texture.shape[-2:]
    x0=int(round(center[0]-pw/2));y0=int(round(center[1]-ph/2))
    x1,y1=max(0,x0),max(0,y0);x2,y2=min(w,x0+pw),min(h,y0+ph)
    if x2<=x1 or y2<=y1:raise ValueError('Patch is outside image')
    texture=texture[...,y1-y0:y2-y0,x1-x0:x2-x0]
    alpha=alpha[...,y1-y0:y2-y0,x1-x0:x2-x0]
    canvas=F.pad(texture,(x1,w-x2,y1,h-y2));mask=F.pad(alpha,(x1,w-x2,y1,h-y2))
    return image*(1-mask)+canvas,mask


def sample_transform(generator):
    u=torch.rand(4,generator=generator).tolist()
    return (20*u[0]-10,.9+.2*u[1],.8+.4*u[2],-.05+.1*u[3])


@contextmanager
def frozen_system(system):
    states=[(p,p.requires_grad) for p in system.parameters()]
    modes=[(m,m.training) for m in system.modules()]
    system.eval()
    for p,_ in states:p.requires_grad_(False)
    try:yield
    finally:
        for p,state in states:p.requires_grad_(state)
        for m,state in modes:m.training=state


def attack_objective(system,attacked,bpda=True):
    hard,_=system.mask(attacked)
    # Evasion term remains differentiable in the no-BPDA ablation.
    soft=system.localizer.soft_mask(attacked)
    mask=hard+(soft-soft.detach()) if bpda else hard
    repaired=system.repair(attacked,mask)
    empty=[{'boxes':attacked.new_zeros((0,4)),'labels':torch.zeros(0,dtype=torch.long,device=attacked.device)}]
    bg,_=system.person.loss(repaired,empty)
    return bg+.1*soft.square().mean()


def optimize_image_patch(system,image,center,cfg):
    if image.shape[0]!=1:raise ValueError('Separate patch per image; batch must be1')
    seed=int(cfg.get('seed',2026));rng=torch.Generator().manual_seed(seed)
    restarts=cfg.get('restarts',3);iterations=cfg.get('iterations',100);eot=cfg.get('eot',8)
    selection_rng=torch.Generator().manual_seed(seed+1000003)
    bank=([sample_transform(selection_rng) for _ in range(cfg.get('selection_eot',8))]
            if cfg.get('use_eot',True) else [(0.,1.,1.,0.)])
    best=None;logs=[]
    devices=[image.device.index or 0] if image.device.type=='cuda' else []
    with frozen_system(system),torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for restart in range(restarts):
            torch.manual_seed(seed+restart*100003)
            patch=torch.rand((1,3,cfg.get('patch_size',64),cfg.get('patch_size',64)),generator=rng).to(image.device).requires_grad_()
            optimizer=torch.optim.Adam([patch],lr=cfg.get('lr',.03))
            history=[]
            for iteration in range(iterations):
                optimizer.zero_grad(set_to_none=True);loss_sum=0.
                # Accumulate gradients sequentially so8 EOT samples do not retain8 full graphs.
                for _ in range(eot):
                    params=sample_transform(rng) if cfg.get('use_eot',True) else (0.,1.,1.,0.)
                    attacked,_=place_patch(image,patch,center,params)
                    objective=attack_objective(system,attacked,cfg.get('bpda',True))/eot
                    objective.backward();loss_sum+=float(objective.detach())
                grad=float(patch.grad.norm());optimizer.step()
                with torch.no_grad():patch.clamp_(0,1)
                history.append({'iteration':iteration+1,'objective':loss_sum,'patch_grad_norm':grad})
            scores=[]
            # Also fix stochastic detector proposal sampling during selection.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed+1000003)
                for params in bank:
                    attacked,_=place_patch(image,patch,center,params)
                    scores.append(float(attack_objective(system,attacked,cfg.get('bpda',True)).detach()))
            score=sum(scores)/len(scores)
            logs.append({'restart':restart,'selection_objective':score,'history':history})
            if best is None or score<best[0]:best=(score,patch.detach().clone(),restart)
        attacked,mask=place_patch(image,best[1],center)
    return {'image':attacked.detach(),'texture':best[1],'patch_mask':mask.detach(),
            'center':center,'selected_restart':best[2],'selection_objective':best[0],
            'history':logs,'config':dict(cfg),'seed':seed,'selection_bank':bank}
