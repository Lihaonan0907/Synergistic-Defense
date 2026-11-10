"""Equations 14–18. Clean reference is deliberately absent from this API."""
import torch
from torch import nn
from .wavelet import WaveletPyramid,BandEncoder


class DualDomainRepair(nn.Module):
    def __init__(self,wave='db6',levels=4,width=32,strength=.7,branches='both'):
        super().__init__()
        if branches not in ['both','frequency','spatial']:raise ValueError(branches)
        self.branches,self.strength=branches,strength
        self.wavelet=WaveletPyramid(wave,levels)
        self.frequency=BandEncoder(levels,width)
        self.spatial=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.SiLU(),
                nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.fusion=nn.Sequential(nn.Conv2d(width*2,width,3,padding=1),nn.SiLU(),
                    nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,3,3,padding=1))

    def frequency_features(self,x):
        f=self.frequency(self.wavelet(x),(max(1,x.shape[-2]//2),max(1,x.shape[-1]//2)))
        return nn.functional.interpolate(f,size=x.shape[-2:],mode='bilinear',align_corners=False)

    def delta(self,x):
        f=self.frequency_features(x)
        s=self.spatial(x)
        if self.branches=='frequency':s=s*0
        if self.branches=='spatial':f=f*0
        return self.fusion(torch.cat([f,s],1))

    def forward(self,x,mask):
        if mask.shape!=(x.shape[0],1,*x.shape[-2:]):raise ValueError('mask must be B×1×H×W')
        return x+self.strength*self.delta(x)*mask
