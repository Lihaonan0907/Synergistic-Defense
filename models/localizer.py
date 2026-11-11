"""Equations 4,6–13 plus explicitly labeled A01 spectral detector interface."""
import torch
from torch import nn
from .wavelet import WaveletPyramid,BandEncoder
from .detectors import build_detector


class HighPassEnhancement(nn.Module):
    def __init__(self,beta=1.):
        super().__init__()
        self.beta=beta
        weight=torch.full((3,1,3,3),-1/8)
        weight[:,:,1,1]=1.
        self.kernel=nn.Parameter(weight)
    def forward(self,x):
        # Literal coefficient sums have large gradients. Preserve float32 along
        # the direct AFAL -> H route while detector convolutions use AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            kernel=self.kernel.float()-self.kernel.float().mean((-2,-1),keepdim=True)
            high=nn.functional.conv2d(nn.functional.pad(x.float(),(1,1,1,1),mode='reflect'),kernel,groups=3)
            return x.float()+self.beta*high


class SpectralPatchLocalizer(nn.Module):
    def __init__(self,size=416,wave='db2',levels=4,width=16,beta=1.,mode='shared_enhancement',tiny=False,roi_backend='native'):
        super().__init__()
        if mode not in ['paper_literal','shared_enhancement']:raise ValueError(mode)
        self.mode=mode
        self.wavelet=WaveletPyramid(wave,levels)
        self.enhancement=HighPassEnhancement(beta)
        self.band_encoder=BandEncoder(levels,width)
        self.spectral_projection=nn.Conv2d(width,3,1)
        self.fusion=nn.Conv2d(6,3,1)
        with torch.no_grad():
            self.fusion.weight.zero_();self.fusion.bias.zero_()
            for i in range(3):
                self.fusion.weight[i,i,0,0]=1.
                self.fusion.weight[i,3+i,0,0]=.1
        self.detection=build_detector(size=size,tiny=tiny,roi_backend=roi_backend)

    def coefficients(self,x,enhance=False):
        return self.wavelet(self.enhancement(x) if enhance else x)

    def detection_input(self,bands,size):
        spatial=self.wavelet.inverse(bands,size)
        spectral=self.band_encoder(bands,(max(1,size[0]//2),max(1,size[1]//2)))
        spectral=nn.functional.interpolate(self.spectral_projection(spectral),size=size,mode='bilinear',align_corners=False)
        return self.fusion(torch.cat([spatial,spectral],1))

    def band_loss(self,bands,size,targets):
        return self.detection.loss(self.detection_input(bands,size),targets)

    def predict(self,x):
        bands=self.coefficients(x,enhance=self.mode=='shared_enhancement')
        return self.detection.predict(self.detection_input(bands,x.shape[-2:]))

    def soft_mask(self,x):
        bands=self.coefficients(x,enhance=self.mode=='shared_enhancement')
        return self.detection.soft_objectness(self.detection_input(bands,x.shape[-2:]))
