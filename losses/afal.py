"""Literal real-coefficient AFAL; scores are detection-loss gradients, Eq10–13."""
import torch


def signed_normalize(c,eps=1e-6):
    return c/torch.sqrt(c.square()+eps)


def vulnerability_scores(loss_base,loss_enhanced,base,enhanced,eps=1e-6,higher_order=False):
    # As with ordinary AMP backward, derivative construction runs outside
    # autocast; forward-recorded dtypes remain in the autograd graph.
    with torch.autocast(device_type=loss_base.device.type,enabled=False):
        gb=torch.autograd.grad(loss_base,base,retain_graph=True,create_graph=higher_order)
        ge=torch.autograd.grad(loss_enhanced,enhanced,retain_graph=True,create_graph=higher_order)
    def norms(gs):
        return torch.stack([torch.linalg.vector_norm(g.float().flatten(1),dim=1) for g in gs],1)
    v=norms(ge)/(norms(gb)+eps)
    return v if higher_order else v.detach()


def adaptive_weights(v,base_weight=10.,kappa=10.,tau=.8):
    return base_weight*torch.sigmoid(kappa*(v-tau))


def alignment_loss(base,enhanced,weights,alpha=.6,beta=.4,eps=1e-6,reduction='batch_mean'):
    mag=[];signed=[]
    for a,b in zip(base,enhanced):
        m=(a.abs()-b.abs()).square().flatten(1)
        s=(signed_normalize(a,eps)-signed_normalize(b,eps)).square().flatten(1)
        mag.append(m.mean(1) if reduction=='band_mean' else m.sum(1))
        signed.append(s.mean(1) if reduction=='band_mean' else s.sum(1))
    magnitude=torch.stack(mag,1);polarity=torch.stack(signed,1)
    terms=weights*(alpha*magnitude+beta*polarity)
    loss=terms.sum() if reduction=='literal_sum' else terms.sum(1).mean()
    return loss,{'magnitude':magnitude.detach(),'signed':polarity.detach(),'weights':weights.detach()}
