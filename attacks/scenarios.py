"""Executable multi-patch/environment controls; unspecified details are assumptions."""
import torch
import torch.nn.functional as F
from .whitebox import place_patch


def multi_patch(image,person_boxes,textures,heterogeneous=False):
    result=image;mask=torch.zeros_like(image[:,:1]);used=[]
    for index,box in enumerate(person_boxes):
        patch=textures[index%len(textures)] if heterogeneous else textures[0]
        center=[float((box[0]+box[2])/2),float((box[1]+box[3])/2)]
        result,m=place_patch(result,patch,center)
        mask=torch.maximum(mask,m);used.append({'texture_index':index%len(textures) if heterogeneous else 0,'center':center})
    return result,mask,used


def environmental_corruption(x,kind,seed=2026,severity=.3):
    """A15: illustrative levels, not falsely labeled as the undisclosed Table X levels."""
    generator=torch.Generator(device=x.device).manual_seed(seed)
    if kind=='normal':return x
    if kind=='low_light':
        noise=torch.randn(x.shape,device=x.device,generator=generator)*(.05*severity)
        return (x*(1-severity)+noise).clamp(0,1)
    if kind=='fog':
        h,w=x.shape[-2:];yy=torch.linspace(-1,1,h,device=x.device)[:,None];xx=torch.linspace(-1,1,w,device=x.device)[None,:]
        alpha=(severity*(.6+.4*torch.exp(-(xx.square()+yy.square()))))[None,None]
        return x*(1-alpha)+alpha
    if kind=='rain':
        points=(torch.rand(x[:,:1].shape,device=x.device,generator=generator)<.003*severity).float()
        kernel=x.new_zeros(1,1,11,5)
        for i in range(11):kernel[0,0,i,min(4,i//3)]=1/4
        rain=F.conv2d(points,kernel,padding=(5,2))
        return (x*(1-rain.clamp(0,1))+rain.clamp(0,1)).clamp(0,1)
    raise ValueError(kind)
