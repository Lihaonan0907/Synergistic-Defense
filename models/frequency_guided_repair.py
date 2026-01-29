"""


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from pytorch_wavelets import DWTForward
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    import pywt
    import numpy as np
    PYTORCH_WAVELETS_AVAILABLE = False


class CompleteWaveletTransform(nn.Module):
    """（）- """
    
    def __init__(self, wavelet='db6', levels=3):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        
        if PYTORCH_WAVELETS_AVAILABLE:
            # pytorch_wavelets
            self.dwt = DWTForward(J=levels, wave=wavelet, mode='zero')
            self.use_differentiable = True
        else:
            self.use_differentiable = False
    
    def forward(self, x):
        """
        
        x: [B, C, H, W]
        : (LL, [YH1, YH2, ...])
        """
        if self.use_differentiable:
            # 🔧 pytorch_wavelets
            # DWTForward: (LL, [YH_list])
            # LL: [B, C, H', W']
            # YH_list: [(B, C, 3, H_i, W_i) for each level]
            ll, yh_list = self.dwt(x)
            return ll, yh_list
        else:
            # 🔧 numpy（，）
            return self._numpy_dwt(x)
    
    def _numpy_dwt(self, x):
        """numpyDWT（，）"""
        coeffs = []
        current = x
        
        for level in range(self.levels):
            # 2D DWT
            batch_coeffs = []
            for b in range(current.shape[0]):
                channel_coeffs = []
                for c in range(current.shape[1]):
                    # numpy
                    img_tensor = current[b, c]
                    img = img_tensor.detach().cpu().numpy() if img_tensor.requires_grad else img_tensor.cpu().numpy()
                    coeff = pywt.dwt2(img, self.wavelet)
                    channel_coeffs.append(coeff)
                batch_coeffs.append(channel_coeffs)
            

            # LL: [B, C, H/2, W/2]
            # YH: [B, C, 3, H/2, W/2] (LH, HL, HH)
            ll_list = []
            yh_list = []
            
            for b_coeffs in batch_coeffs:
                ll_channels = []
                lh_channels = []
                hl_channels = []
                hh_channels = []
                
                for (ll, (lh, hl, hh)) in b_coeffs:
                    ll_channels.append(ll)
                    lh_channels.append(lh)
                    hl_channels.append(hl)
                    hh_channels.append(hh)
                
                # numpy.array，tensor
                ll_list.append(torch.from_numpy(
                    np.array(ll_channels)  # numpy
                ).unsqueeze(0).float())
                
                yh_list.append(torch.stack([
                    torch.from_numpy(np.array(lh_channels)).float(),
                    torch.from_numpy(np.array(hl_channels)).float(),
                    torch.from_numpy(np.array(hh_channels)).float()
                ], dim=1))  # [C, 3, H/2, W/2]
            
            ll = torch.cat(ll_list, dim=0).to(x.device)
            yh = torch.stack([y.unsqueeze(0) for y in yh_list], dim=0).squeeze(1).to(x.device)
            
            coeffs.append(yh)
            current = ll
        
        return current, coeffs  # LL, [YH1, YH2, YH3]


class FrequencyGuidedRepairNet(nn.Module):
    """"""
    
    def __init__(self, wavelet='db6', levels=3):
        super().__init__()
        self.wavelet_transform = CompleteWaveletTransform(wavelet, levels)
        self.levels = levels
        
        # : LL + YH
        # YH: levels * 3 ，3
        # LL: 3
        # : 3 + levels * 3 * 3 = 3 + 9 * levels
        freq_channels = 3 + 9 * levels
        

        self.freq_encoder = nn.Sequential(
            nn.Conv2d(freq_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        

        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.repair_strength = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, adv_img, patch_masks=None, clean_img=None):
        """
        
        adv_img:  [B, 3, H, W]
        patch_masks:  [B, 1, H, W] ()
        clean_img:  [B, 3, H, W] ()
        
        :
        - repaired: 
        - info_dict: 
        """
        batch_size = adv_img.shape[0]
        
        # 1. 
        ll_adv, yh_adv = self.wavelet_transform(adv_img)
        
        # 2. 
        # LL: [B, 3, H/2^levels, W/2^levels]
        # YH: list of [B, 3, 3, H/2^i, W/2^i]
        freq_features = []
        
        # LL
        ll_upsampled = F.interpolate(
            ll_adv, 
            size=adv_img.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        freq_features.append(ll_upsampled)
        
        # YH
        for level_idx, yh in enumerate(yh_adv):
            # yh: [B, 3, 3, H/2^(level_idx+1), W/2^(level_idx+1)]
            for band_idx in range(3):  # LH, HL, HH
                band = yh[:, :, band_idx, :, :]  # [B, 3, H, W]

                band_upsampled = F.interpolate(
                    band,
                    size=adv_img.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                freq_features.append(band_upsampled)
        

        freq_input = torch.cat(freq_features, dim=1)  # [B, freq_channels, H, W]
        
        # 3. 
        freq_feat = self.freq_encoder(freq_input)
        
        # 4. 
        spatial_feat = self.spatial_encoder(adv_img)
        
        # 5. 
        combined = torch.cat([freq_feat, spatial_feat], dim=1)
        repair_delta = self.fusion(combined)
        
        # 6. 
        alpha = torch.sigmoid(self.repair_strength)
        
        if patch_masks is not None and patch_masks.sum() > 0:

            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
            
            repaired = adv_img + alpha * repair_delta * patch_masks_3c
        else:

            repaired = adv_img + alpha * repair_delta
        
        repaired = torch.clamp(repaired, 0, 1)
        
        # 7. 
        info_dict = {
            'freq_features': freq_feat,
            'spatial_features': spatial_feat,
            'repair_delta': repair_delta,
            'repair_strength': alpha.item(),
            'll_component': ll_upsampled,
        }
        
        return repaired, info_dict


class FrequencyConsistencyLoss(nn.Module):
    """"""
    
    def __init__(self, wavelet='db6', levels=3):
        super().__init__()
        self.wavelet_transform = CompleteWaveletTransform(wavelet, levels)
        self.levels = levels
    
    def compute(self, repaired_img, clean_img, patch_masks=None):
        """
        
        
        :
        - repaired_img:  [B, 3, H, W]
        - clean_img:  [B, 3, H, W]
        - patch_masks:  [B, 1, H, W] ()
        
        :
        - loss_dict: 
        """

        ll_rep, yh_rep = self.wavelet_transform(repaired_img)
        ll_clean, yh_clean = self.wavelet_transform(clean_img)
        
        # 1. 
        low_freq_loss = F.mse_loss(ll_rep, ll_clean)
        
        # 2. （）
        high_freq_loss = 0.0
        high_freq_losses = []
        
        for level in range(self.levels):
            for band in range(3):  # LH, HL, HH
                # ，
                weight = 1.0 / (level + 1)
                
                band_rep = yh_rep[level][:, :, band, :, :]
                band_clean = yh_clean[level][:, :, band, :, :]
                
                band_loss = F.mse_loss(band_rep, band_clean)
                weighted_loss = weight * band_loss
                
                high_freq_loss += weighted_loss
                high_freq_losses.append(weighted_loss.item())
        
        # 3. ，
        if patch_masks is not None and patch_masks.sum() > 0:
            # LL
            mask_for_ll = F.interpolate(
                patch_masks,
                size=ll_rep.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            

            mask_weight = mask_for_ll.mean()
            low_freq_loss = low_freq_loss * (1.0 + mask_weight)
        
        # 4. 
        total_freq_loss = low_freq_loss + high_freq_loss
        
        return {
            'total_freq_loss': total_freq_loss,
            'low_freq_loss': low_freq_loss,
            'high_freq_loss': high_freq_loss,
            'high_freq_details': high_freq_losses
        }
    
    def __call__(self, repaired_img, clean_img, patch_masks=None):
        """"""
        return self.compute(repaired_img, clean_img, patch_masks)


if __name__ == '__main__':
    """"""
    print("=" * 80)
    print("")
    print("=" * 80)
    

    model = FrequencyGuidedRepairNet(wavelet='db6', levels=3)
    freq_loss = FrequencyConsistencyLoss(wavelet='db6', levels=3)
    

    batch_size = 2
    adv_img = torch.rand(batch_size, 3, 416, 416)
    clean_img = torch.rand(batch_size, 3, 416, 416)
    patch_masks = torch.zeros(batch_size, 1, 416, 416)
    patch_masks[:, :, 100:200, 100:200] = 1.0
    
    print(f"\n:")
    print(f"  : {adv_img.shape}")
    print(f"  : {clean_img.shape}")
    print(f"  : {patch_masks.shape}")
    

    print(f"\n...")
    repaired, info = model(adv_img, patch_masks, clean_img)
    
    print(f"\n:")
    print(f"  : {repaired.shape}")
    print(f"  : {info['repair_strength']:.4f}")
    print(f"  : {info['freq_features'].shape}")
    print(f"  : {info['spatial_features'].shape}")
    

    print(f"\n...")
    loss_dict = freq_loss(repaired, clean_img, patch_masks)
    
    print(f"\n:")
    print(f"  : {loss_dict['total_freq_loss'].item():.6f}")
    print(f"  : {loss_dict['low_freq_loss'].item():.6f}")
    print(f"  : {loss_dict['high_freq_loss'].item():.6f}")
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n:")
    print(f"  : {total_params:,}")
    print(f"  : {trainable_params:,}")
    
    print(f"\n{'='*80}")
    print("✅ ！")
    print("=" * 80)
