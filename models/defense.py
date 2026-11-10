"""Complete frozen-parameter deployment and white-box differentiable surrogate."""
import torch
from torch import nn


def boxes_to_mask(predictions,size,threshold=.25):
    h,w=size
    device=predictions[0]['boxes'].device
    masks=torch.zeros((len(predictions),1,h,w),device=device)
    for i,p in enumerate(predictions):
        for box in p['boxes'][p['scores']>=threshold]:
            x1,y1=box[:2].floor().long().tolist();x2,y2=box[2:].ceil().long().tolist()
            masks[i,:,max(y1,0):min(y2,h),max(x1,0):min(x2,w)]=1
    return masks


def bpda_mask(hard,soft):
    return hard+(soft-soft.detach())


class DefenseSystem(nn.Module):
    def __init__(self,localizer,repair,person,mask_threshold=.25):
        super().__init__()
        self.localizer,self.repair,self.person=localizer,repair,person
        self.mask_threshold=mask_threshold

    def mask(self,x,bpda=False):
        with torch.no_grad():
            hard=boxes_to_mask(self.localizer.predict(x),x.shape[-2:],self.mask_threshold)
        if bpda:
            soft=self.localizer.soft_mask(x)
            return bpda_mask(hard,soft),soft
        return hard,None

    def forward(self,x,mask_override=None):
        mask,_=self.mask(x) if mask_override is None else (mask_override,None)
        repaired=self.repair(x,mask)
        return {'mask':mask,'repaired':repaired,'person':self.person.predict(repaired)}
