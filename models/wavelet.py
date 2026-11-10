"""Equations 6/7/9: image-domain, real-valued differentiable multilevel DWT."""
import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletPyramid(nn.Module):
    def __init__(self, wave='db6', levels=4, mode='symmetric'):
        super().__init__()
        self.levels, self.wave, self.mode = levels, wave, mode
        self.analysis = DWTForward(J=levels, wave=wave, mode=mode)
        self.synthesis = DWTInverse(wave=wave, mode=mode)

    @property
    def names(self):
        # pytorch_wavelets orientation order: LH, HL, HH; high list finest first.
        return [f'LL{self.levels}'] + [f'{b}{l}' for l in range(1, self.levels+1) for b in ['LH','HL','HH']]

    def forward(self, x):
        # DWT and AFAL remain float32 under AMP for small signed coefficients.
        with torch.autocast(device_type=x.device.type, enabled=False):
            low, high = self.analysis(x.float())
        return [low] + [h[:, :, o] for h in high for o in range(3)]

    def inverse(self, bands, size):
        high = [torch.stack(bands[1+3*l:4+3*l], dim=2) for l in range(self.levels)]
        with torch.autocast(device_type=bands[0].device.type, enabled=False):
            x = self.synthesis((bands[0].float(), [h.float() for h in high]))
        return x[..., :size[0], :size[1]]


class BandEncoder(nn.Module):
    """A01/A06: process every band before multiscale channel fusion."""
    def __init__(self, levels=4, width=16):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
                          nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
            for _ in range(1+3*levels)])
        self.project = nn.Conv2d((1+3*levels)*width, width, 1)

    def forward(self, bands, size):
        features = [torch.nn.functional.interpolate(enc(b), size=size, mode='bilinear', align_corners=False)
                    for enc,b in zip(self.encoders,bands)]
        return self.project(torch.cat(features,dim=1))
