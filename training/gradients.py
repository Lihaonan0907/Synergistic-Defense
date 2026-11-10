"""Explicit optimization variants; objectives and AFAL derivatives are unchanged."""
import torch

def stage1_optimizer(model,cfg):
    lr=cfg.get('lr',.01);scale=cfg.get('enhancement_lr_scale',1.)
    if not 0<scale<=1:raise ValueError('H must remain trainable with positive LR multiplier <=1')
    if scale==1.:return torch.optim.SGD(model.parameters(),lr=lr,momentum=.937,weight_decay=.0005)
    h=list(model.enhancement.parameters());ids={id(p) for p in h}
    return torch.optim.SGD([{'params':[p for p in model.parameters() if id(p) not in ids],'lr':lr},
                            {'params':h,'lr':lr*scale}],lr=lr,momentum=.937,weight_decay=.0005)


def group_name(name):
    if name.startswith('enhancement.'):return 'H'
    if '.backbone.' in name:return 'backbone'
    if '.rpn.' in name:return 'rpn'
    if '.roi_heads.' in name:return 'roi'
    return 'spectral_fusion'


def gradient_groups(model):
    groups={}
    for name,p in model.named_parameters():
        if p.requires_grad and p.grad is not None:groups.setdefault(group_name(name),[]).append(p)
    return groups


def group_norms(model):
    return {k:float(torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in ps])))
            for k,ps in gradient_groups(model).items()}


def clip_gradients(model,cfg,diagnose=False,error_if_nonfinite=True):
    policy=cfg.get('clip_policy','global')
    if policy not in ['global','parameter_groups']:raise ValueError('Unknown clipping policy: '+policy)
    groups=gradient_groups(model)
    before=group_norms(model) if diagnose or policy=='parameter_groups' else {}
    if policy=='global':
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('clip_norm',1.),error_if_nonfinite=error_if_nonfinite)
    else:
        # The total norm is checked BEFORE any group is mutated. Otherwise an
        # overflowing group could leave finite groups changed on a skipped step.
        norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for ps in groups.values() for p in ps]))
        if bool(torch.isfinite(norm)):
            for ps in groups.values():torch.nn.utils.clip_grad_norm_(ps,cfg.get('clip_norm',1.),error_if_nonfinite=True)
        elif error_if_nonfinite:raise FloatingPointError('Nonfinite group gradients')
    after=group_norms(model) if diagnose and bool(torch.isfinite(norm)) else {}
    kernel=getattr(getattr(model,'enhancement',None),'kernel',None)
    h=kernel.detach().float() if kernel is not None else None
    return norm,{'policy':policy,'preclip_group_norms':before,'postclip_group_norms':after,
        'H_effective_kernel_norm':float((h-h.mean((-1,-2),keepdim=True)).norm()) if h is not None else None}
