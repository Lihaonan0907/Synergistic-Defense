"""Paper objectives with separately named assumptions, never any AP targets."""
import torch
from losses.afal import vulnerability_scores,adaptive_weights,alignment_loss
from losses.repair import frequency_reconstruction,dependency_loss


def _rng_snapshot(device):
    return torch.random.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None


def _rng_restore(states,device):
    torch.random.set_rng_state(states[0])
    if device.type=='cuda':torch.cuda.set_rng_state(states[1],device)


def stage1_objective(model,x,targets,cfg):
    x=x.detach().requires_grad_(True)
    base=model.coefficients(x)
    enhanced=model.coefficients(x,enhance=True)
    state=_rng_snapshot(x.device)
    lb,parts=model.band_loss(base,x.shape[-2:],targets)
    _rng_restore(state,x.device)
    le,_=model.band_loss(enhanced,x.shape[-2:],targets)
    weighting=cfg.get('weighting','gradient')
    if weighting=='gradient':
        v=vulnerability_scores(lb,le,base,enhanced,eps=cfg.get('eps',1e-6),higher_order=cfg.get('higher_order',False))
        weights=adaptive_weights(v,cfg.get('base_weight',10.),cfg.get('kappa',10.),cfg.get('tau',.8))
    elif weighting=='energy':
        v=torch.stack([b.detach().square().mean((1,2,3)).sqrt()/(a.detach().square().mean((1,2,3)).sqrt()+1e-6) for a,b in zip(base,enhanced)],1)
        weights=adaptive_weights(v,cfg.get('base_weight',10.),cfg.get('kappa',10.),cfg.get('tau',.8))
    elif weighting in ['fixed','none']:
        v=x.new_zeros((len(x),len(base)))
        weights=torch.full_like(v,cfg.get('base_weight',10.) if weighting=='fixed' else 0.)
    else:raise ValueError(weighting)
    align,stats=alignment_loss(base,enhanced,weights,cfg.get('alpha',.6),cfg.get('beta_align',.4),cfg.get('eps',1e-6),cfg.get('reduction','batch_mean'))
    detection=lb if model.mode=='paper_literal' else (lb+le)/2
    loss=detection+cfg.get('lambda_align',10.)*align
    return loss,{'detection':detection,'detection_base':lb,'detection_enhanced':le,'alignment':align,
                'vulnerability':v.detach(),'weights':weights.detach(),**{f'patch/{k}':v for k,v in parts.items()}}


def stage2_objective(system,adv,clean,targets,cfg,phase='joint',mask_override=None):
    mask,_=system.mask(adv) if mask_override is None else (mask_override,None)
    repaired=system.repair(adv,mask)
    reconstruction,terms=frequency_reconstruction(repaired,clean,system.repair.wavelet,
            cfg.get('lambda_low',1.),cfg.get('lambda_high',1.),cfg.get('reduction','batch_mean'))
    zero=reconstruction.new_zeros(())
    lp,lu,dep=zero,zero,zero
    if phase!='initial':
        state=_rng_snapshot(adv.device)
        lp,_=system.person.loss(repaired,targets)
        _rng_restore(state,adv.device)
        lu,_=system.person.loss(adv,targets)
        dep=dependency_loss(lp,lu,cfg.get('gamma',.2),cfg.get('detach_unrepaired',False))
    total=cfg.get('lambda_repair',1.)*reconstruction+cfg.get('lambda_person',1.)*lp+cfg.get('lambda_dep',.1)*dep
    return total,{'person':lp,'unrepaired':lu,'dependency':dep,'reconstruction':reconstruction,
                  'mask_coverage':mask.mean(),**terms}
