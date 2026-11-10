"""Full-image PSNR/SSIM to paired clean reference, not a detection surrogate."""
import numpy as np
from skimage.metrics import structural_similarity

def image_quality(repaired,clean):
    a=repaired.detach().cpu().permute(1,2,0).numpy()
    b=clean.detach().cpu().permute(1,2,0).numpy()
    mse=np.mean((a-b)**2)
    return {'PSNR_dB':float(-10*np.log10(mse)) if mse else None,
            'perfect_reconstruction':bool(mse==0),'zero_MSE_PSNR_convention':'null denotes positive infinity',
            'SSIM':float(structural_similarity(a,b,data_range=1.,channel_axis=-1))}
