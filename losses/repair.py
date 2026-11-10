"""Equations 19–24, including literal dependency gradients on both inputs."""
import torch


def frequency_reconstruction(repaired,clean,wavelet,lambda_low=1.,lambda_high=1.,reduction='batch_mean'):
    a,b=wavelet(repaired),wavelet(clean)
    errors=[]
    for c,d in zip(a,b):
        e=(c-d).square().flatten(1)
        errors.append(e.mean(1) if reduction=='band_mean' else e.sum(1))
    low=errors[0].mean()
    high=torch.stack(errors[1:],1).sum(1).mean()
    if reduction=='literal_sum':low=low*repaired.shape[0];high=high*repaired.shape[0]
    return lambda_low*low+lambda_high*high,{'low':low,'high':high}


def dependency_loss(loss_repaired,loss_unrepaired,gamma=.2,detach_unrepaired=False):
    if detach_unrepaired:loss_unrepaired=loss_unrepaired.detach()
    return torch.relu(gamma+loss_repaired-loss_unrepaired)
