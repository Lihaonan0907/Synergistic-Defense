#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

：
：（+）
"""

import argparse
import gc
import math
import os
import sys
import time
import logging
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda import amp
from torch.optim import lr_scheduler
import numpy as np
import yaml
from tqdm import tqdm
import cv2


try:
    import pywt
    from pytorch_wavelets import DWTForward, DWTInverse
    HAS_WAVELETS = True
except ImportError as e:
    raise ImportError("pytorch_waveletspywt: pip install PyWavelets pytorch-wavelets")


def clear_cuda_cache():
    """CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device, de_parallel, smart_optimizer, ModelEMA
from utils.general import LOGGER, check_yaml, colorstr, increment_path, init_seeds, non_max_suppression, xywh2xyxy, box_iou, scale_boxes
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.callbacks import Callbacks
import val_Defense as validate

# Ultralytics YOLO (YOLOv11/v12)
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    HAS_ULTRALYTICS = True
    LOGGER.info("✓ Ultralytics YOLO (YOLOv11/v12)")
except ImportError:
    HAS_ULTRALYTICS = False
    LOGGER.warning("⚠️ Ultralytics，YOLOv5")

# =============================================================================
#  - 
# =============================================================================

class CompleteWaveletTransform(nn.Module):
    """ - """
    
    def __init__(self, wavelet='db6', levels=2, mode='zero'):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        

        self.dwt = DWTForward(J=levels, wave=wavelet, mode=mode)
        self.idwt = DWTInverse(wave=wavelet, mode=mode)
        
    def forward(self, x):
        """"""

        x_float = x.float() if x.dtype != torch.float32 else x
        
        # 🚀 torch.no_grad()（）

        yl, yh = self.dwt(x_float)
        

        if x_float is not x:
            del x_float
        
        return yl, yh
    
    def inverse(self, yl, yh):
        """"""

        reconstructed = self.idwt((yl, yh))
        

        return reconstructed.type_as(yl) if hasattr(yl, 'dtype') else reconstructed

    def get_band_count(self):
        """"""
        return 1 + 3 * self.levels  # LL + 3 * levels (LH, HL, HH)

    def analyze_frequency_bands(self, x):
        """"""
        yl, yh = self.forward(x)
        
        band_energies = {}
        
        # LL
        band_energies['LL'] = torch.mean(yl ** 2)
        

        for level in range(self.levels):
            coeffs = yh[level]
            band_energies[f'LH_{level}'] = torch.mean(coeffs[:, :, 0] ** 2)
            band_energies[f'HL_{level}'] = torch.mean(coeffs[:, :, 1] ** 2)
            band_energies[f'HH_{level}'] = torch.mean(coeffs[:, :, 2] ** 2)
        
        return band_energies

# =============================================================================
#  - AFAL
# =============================================================================

class WaveletEnhance(nn.Module):
    """ - train_Defense.py"""
    def __init__(self, kernel_size=3, enhance_factor=1.5, device='cuda'):
        super(WaveletEnhance, self).__init__()
        self.enhance_factor = enhance_factor
        self.device = device
        self.kernel_size = kernel_size
        # kernel，forward
        self.kernel = None
    
    def _create_highpass_kernel(self, size, device):
        """"""
        kernel = torch.ones(size, size, dtype=torch.float32, device=device)
        center = size // 2
        kernel[center, center] = - (size * size - 1)
        return kernel.view(1, 1, size, size).repeat(3, 1, 1, 1)
    
    def forward(self, x):
        """"""
        current_device = x.device
        
        # kernel
        if self.kernel is None or self.kernel.device != current_device:
            self.kernel = self._create_highpass_kernel(self.kernel_size, current_device)
        
        high_pass = F.conv2d(
            input=x,
            weight=self.kernel,
            padding=self.kernel.shape[-1] // 2,
            groups=3
        )
        enhanced = x + self.enhance_factor * high_pass
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced

# =============================================================================
# -AFAL
# =============================================================================

class AdaptiveFreqAFALFusion(nn.Module):
    """-AFAL - Full"""
    
    def __init__(self, wavelet='coif1', freq_levels=4, afal_levels=3):
        super().__init__()
        from pytorch_wavelets import DWTForward, DWTInverse
        
        # AFAL（coif1，3）
        self.afal_dwt_forward = DWTForward(wave=wavelet, J=afal_levels, mode='zero')
        self.afal_dwt_inverse = DWTInverse(wave=wavelet, mode='zero')
        self.wavelet_enhance = None
        

        
    def forward(self, x, mode='afal_only'):
        """
        
        Args:
            x:  [B, C, H, W]
            mode: 'afal_only' (AFAL)
        """
        # AFAL
        if self.wavelet_enhance is not None:
            with torch.amp.autocast('cuda', enabled=False):
                output = self.wavelet_enhance(x.float())
        else:
            output = x
        
        return output

# =============================================================================
# =============================================================================

# =============================================================================

# =============================================================================
# （1+2）- 
# =============================================================================

class FrequencyGuidedRepairNet(nn.Module):
    """ - 12（）"""
    
    def __init__(self, wavelet='db6', levels=3, enable_frequency=True, repair_strength_init=0.3):
        super().__init__()
        

        self.wavelet = wavelet
        self.levels = levels
        self.enable_frequency = enable_frequency
        self.wavelet_transform = None
        self._freq_cache = {}
        
        # 🔥 1: 
        freq_channels = 3 + 9 * levels
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(freq_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
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
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
        #  s ( --repair-strength-init )
        self.repair_strength = nn.Parameter(torch.tensor(repair_strength_init))
        
    def _init_wavelet_transform(self, device):
        """"""
        if self.wavelet_transform is None and self.enable_frequency:
            try:
                from pytorch_wavelets import DWTForward
                self.wavelet_transform = DWTForward(
                    J=self.levels, 
                    wave=self.wavelet, 
                    mode='zero'
                ).to(device)
            except (ImportError, Exception):
                self.wavelet_transform = None
                self.enable_frequency = False
    
    def _extract_frequency_features(self, img):
        """（：）"""
        if not self.enable_frequency:
            return None
            
        if self.wavelet_transform is None:
            self._init_wavelet_transform(img.device)
        
        if self.wavelet_transform is None:
            return None
        
        try:
            with torch.amp.autocast('cuda', enabled=False):
                ll, yh = self.wavelet_transform(img.float())
            
            # 🔥 ：
            target_size = (ll.shape[2], ll.shape[3])
            
            freq_features = [ll]
            for level in range(self.levels):
                for band in range(3):
                    hf_band = yh[level][:, :, band, :, :]

                    if hf_band.shape[2:] != target_size:
                        hf_band = F.adaptive_avg_pool2d(hf_band, target_size)
                    freq_features.append(hf_band)
            
            freq_input = torch.cat(freq_features, dim=1)
            return freq_input
            
        except Exception as e:
            self.enable_frequency = False
            return None
    
    def forward(self, adv_img, patch_masks=None, clean_img=None):
        """（）"""

        if self.enable_frequency:
            freq_features_adv = self._extract_frequency_features(adv_img)
        else:
            freq_features_adv = None
        

        if freq_features_adv is not None:

            freq_features_adv = F.interpolate(
                freq_features_adv,
                size=adv_img.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            freq_encoded = self.freq_encoder(freq_features_adv)
        else:
            # ：
            freq_encoded = torch.zeros(
                adv_img.shape[0], 128, adv_img.shape[2], adv_img.shape[3],
                device=adv_img.device, dtype=adv_img.dtype
            )
        

        spatial_encoded = self.spatial_encoder(adv_img)
        

        combined = torch.cat([freq_encoded, spatial_encoded], dim=1)
        repair_delta = self.fusion(combined)
        
        # （）
        if patch_masks is not None and patch_masks.sum() > 0:
            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
            repaired = adv_img + self.repair_strength * repair_delta * patch_masks_3c
        else:
            # ✅ ：，
            # mask，
            repaired = adv_img + self.repair_strength * repair_delta * 0.0
        
        repaired = torch.clamp(repaired, 0, 1)
        
        aux_outputs = {
            'freq_features': freq_encoded,
            'spatial_features': spatial_encoded,
            'repair_delta': repair_delta
        }
        
        return repaired, aux_outputs


# =============================================================================
# 2: （）
# =============================================================================

class FrequencyConsistencyLoss:
    """（）"""
    
    def __init__(self, wavelet='db6', levels=3, device='cuda', enable_frequency=True):
        self.wavelet = wavelet
        self.levels = levels
        self.device = device
        self.enable_frequency = enable_frequency
        self.wavelet_transform = None
        
    def _init_wavelet_transform(self):
        """"""
        if self.wavelet_transform is None and self.enable_frequency:
            try:
                from pytorch_wavelets import DWTForward
                self.wavelet_transform = DWTForward(
                    J=self.levels,
                    wave=self.wavelet,
                    mode='zero'
                ).to(self.device)
            except (ImportError, Exception):
                self.wavelet_transform = None
                self.enable_frequency = False
    
    def __call__(self, repaired_img, clean_img, patch_masks=None):
        """（）"""
        if not self.enable_frequency:
            # 🔧 tensor，tensor
            return {
                'total_freq_loss': repaired_img.sum() * 0.0,
                'low_freq_loss': repaired_img.sum() * 0.0,
                'high_freq_loss': repaired_img.sum() * 0.0
            }
        
        if self.wavelet_transform is None:
            self._init_wavelet_transform()
        
        if self.wavelet_transform is None:
            self.enable_frequency = False
            # 🔧 tensor
            return {
                'total_freq_loss': repaired_img.sum() * 0.0,
                'low_freq_loss': repaired_img.sum() * 0.0,
                'high_freq_loss': repaired_img.sum() * 0.0
            }
        
        try:
            with torch.amp.autocast('cuda', enabled=False):
                ll_rep, yh_rep = self.wavelet_transform(repaired_img.float())
                ll_clean, yh_clean = self.wavelet_transform(clean_img.float())
            

            low_freq_loss = F.mse_loss(ll_rep, ll_clean)
            
            # （）
            high_freq_loss = torch.tensor(0.0, device=repaired_img.device)
            
            for level in range(self.levels):
                weight = 1.0 / (level + 1)
                for band in range(3):
                    band_loss = F.mse_loss(
                        yh_rep[level][:, :, band, :, :],
                        yh_clean[level][:, :, band, :, :]
                    )
                    high_freq_loss = high_freq_loss + weight * band_loss
            

            if patch_masks is not None and patch_masks.sum() > 0:
                patch_masks_resized = F.interpolate(
                    patch_masks.float(),
                    size=ll_rep.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                mask_weight = patch_masks_resized.mean()
                low_freq_loss = low_freq_loss * mask_weight
                high_freq_loss = high_freq_loss * mask_weight
            
            return {
                'total_freq_loss': low_freq_loss + high_freq_loss,
                'low_freq_loss': low_freq_loss,
                'high_freq_loss': high_freq_loss
            }
            
        except Exception as e:
            self.enable_frequency = False
            # 🔧 tensor，
            return {
                'total_freq_loss': repaired_img.sum() * 0.0,
                'low_freq_loss': repaired_img.sum() * 0.0,
                'high_freq_loss': repaired_img.sum() * 0.0
            }


# =============================================================================
# （）
# =============================================================================

class LightweightRepairNet(FrequencyGuidedRepairNet):
    """ - """
    
    def __init__(self, wavelet='db6', levels=3):
        super().__init__(wavelet=wavelet, levels=levels, enable_frequency=True)
        
    def forward(self, x, patch_masks=None, clean_img=None):
        """："""
        repaired, aux_outputs = super().forward(x, patch_masks, clean_img)
        return repaired

# =============================================================================

# =============================================================================

class DynamicTrainingStrategy:
    """🔥  - """
    
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_history = []
        
    def get_strategy(self, epoch, current_adv_map=None):
        """
        🔥 ：
        
        ：
        - （0-40%）：，
        - （40-70%）： - ，
        - （70-100%）： - +
        
        ：
        1. ****（1/100），
        2. ：
        3. ：
        """
        progress = epoch / max(self.total_epochs, 1)
        
        # ============ 1：（0-40%）============
        if progress < 0.4:
            strategy = {
                'stage': '1-',
                'detector_trainable': False,
                'detector_lr_scale': 0.0,     # 0
                'repair_strength': 0.7,
                'patch_visibility': 0.8,      # 🔧 : 0.60.8，
                'repair_weight': 1.0,
                'detection_weight': 0.0,
                'description': '，'
            }
        
        # ============ 2：（40-70%）============
        elif progress < 0.7:
            # 🔥 ：，
            strategy = {
                'stage': '2-',
                'detector_trainable': True,
                'detector_lr_scale': 0.01,    # 🔥 1/100
                'repair_strength': 0.8,
                'patch_visibility': 0.8,
                'repair_weight': 0.7,
                'detection_weight': 0.3,
                'description': '，'
            }
        
        # ============ 3：（70-100%）============
        else:
            strategy = {
                'stage': '3-',
                'detector_trainable': True,
                'detector_lr_scale': 0.05,    # 🔥 1/20
                'repair_strength': 0.9,
                'patch_visibility': 1.0,
                'repair_weight': 0.5,         # 🔧 : 0.30.5，
                'detection_weight': 0.5,
                'description': '：+'
            }
        
        # 🔥 ：mAP
        if current_adv_map is not None:
            if current_adv_map > 0.75 and progress < 0.6:
                # 🚨 ：
                LOGGER.warning(f"🚨 Epoch {epoch}: (mAP={current_adv_map:.2f})！")
                LOGGER.warning(f"   → ：，")
                strategy['detector_trainable'] = False
                strategy['detector_lr_scale'] = 0.0
                strategy['patch_visibility'] = max(0.3, strategy['patch_visibility'] - 0.3)
                strategy['repair_weight'] = 1.0
                strategy['detection_weight'] = 0.0
                strategy['description'] += ' [🚨]'
            
            elif current_adv_map > 0.6 and progress < 0.6:
                # ⚠️ ：
                LOGGER.warning(f"⚠️ Epoch {epoch}: (mAP={current_adv_map:.2f})")
                LOGGER.warning(f"   → 1/10")
                strategy['detector_lr_scale'] *= 0.1
                strategy['detection_weight'] *= 0.5
                strategy['description'] += ' [⚠️LR]'
            
            elif 0.35 <= current_adv_map <= 0.55 and progress > 0.4:
                # ✅ ：，
                LOGGER.info(f"✅ Epoch {epoch}: (mAP={current_adv_map:.2f})，")

            
            elif current_adv_map < 0.3:
                # 💪 ，
                LOGGER.info(f"💪 Epoch {epoch}: (mAP={current_adv_map:.2f})，")
                if progress > 0.5:
                    strategy['detector_lr_scale'] = min(0.1, strategy['detector_lr_scale'] * 1.5)
                    strategy['detection_weight'] = min(0.6, strategy['detection_weight'] * 1.2)
        
        self.epoch_history.append({
            'epoch': epoch,
            'strategy': strategy,
            'adv_map': current_adv_map
        })
        
        return strategy

# =============================================================================
#  - 
# =============================================================================

class RepairDependencyEnforcement:
    """ - """
    
    def __init__(self, stage1_detector, device, detector_type='yolov5'):
        self.stage1_detector = stage1_detector
        self.device = device
        self.detector_type = detector_type.lower()
        self.repair_dependency_threshold = 0.1
    
    def _convert_yolov11_to_yolov5_format(self, pred):
        """YOLOv11/v12YOLOv5 NMS"""
        # 🔥 tuple
        if isinstance(pred, (tuple, list)):
            if len(pred) > 0:
                pred = pred[0]
            else:
                return pred
        
        if not isinstance(pred, torch.Tensor):
            return pred
        
        # YOLOv5
        if pred.dim() == 3 and pred.shape[2] == 85:
            return pred
        
        # YOLOv11
        if pred.dim() == 3 and pred.shape[1] == 84:
            B, C, N = pred.shape
            pred = pred.permute(0, 2, 1)  # [B, N, 84]
            
            bbox = pred[:, :, :4]
            class_logits = pred[:, :, 4:84]
            
            # class logitobjectness
            objectness = class_logits.max(dim=2, keepdim=True)[0].sigmoid()
            class_probs = class_logits.sigmoid()
            
            pred_converted = torch.cat([bbox, objectness, class_probs], dim=2)
            return pred_converted
        
        return pred
    
    def enforce_repair_dependency(self, detector, repair_module, imgs, targets, epoch):
        """（🚀 ）"""
        
        # 🚀 ：mini-batchbatch
        batch_size = min(4, imgs.shape[0])  # 4
        imgs_mini = imgs[:batch_size]
        
        # targets
        targets_mini = targets[targets[:, 0] < batch_size]
        
        if len(targets_mini) == 0:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0
        
        with torch.no_grad():
            # 1. （mini-batch）
            orig_pred = detector(imgs_mini)
            # 🔥 YOLOv11
            if self.detector_type in ['yolov11', 'yolo11', 'yolov12', 'yolo12']:
                orig_pred = self._convert_yolov11_to_yolov5_format(orig_pred)
            orig_detections = non_max_suppression(orig_pred, conf_thres=0.25)
            orig_performance = self._evaluate_detection_performance(orig_detections, targets_mini)
            
            # 2. （mini-batch）
            patch_pred = self.stage1_detector(imgs_mini)
            patch_masks = self._create_patch_masks(patch_pred, imgs_mini.shape)
        
        # 3. （mini-batch）
        # 🔥 （）
        repair_result = repair_module(imgs_mini, patch_masks)
        if isinstance(repair_result, tuple):
            strongly_repaired = repair_result[0]
        else:
            strongly_repaired = repair_result
        
        # 4. （mini-batch）
        with torch.no_grad():
            repaired_pred = detector(strongly_repaired)
            # 🔥 YOLOv11
            if self.detector_type in ['yolov11', 'yolo11', 'yolov12', 'yolo12']:
                repaired_pred = self._convert_yolov11_to_yolov5_format(repaired_pred)
            repaired_detections = non_max_suppression(repaired_pred, conf_thres=0.25)
            repaired_performance = self._evaluate_detection_performance(repaired_detections, targets_mini)
        
        # 5. （）
        dependency_loss = self._compute_dependency_loss(
            orig_performance, repaired_performance, epoch
        )
        

        del orig_pred, patch_pred, patch_masks, strongly_repaired, repaired_pred
        del orig_detections, repaired_detections
        
        return dependency_loss, orig_performance, repaired_performance
    
    def _evaluate_detection_performance(self, detections, targets):
        """（mAP）"""
        if len(targets) == 0:
            return 0.0
        
        total_targets = 0
        total_correct = 0
        

        for img_idx in targets[:, 0].unique():
            img_targets = targets[targets[:, 0] == img_idx]
            total_targets += len(img_targets)
            

            if img_idx.item() < len(detections) and detections[int(img_idx.item())] is not None:
                dets = detections[int(img_idx.item())]
                if len(dets) > 0:
                    # ：
                    total_correct += min(len(dets), len(img_targets))
        
        if total_targets > 0:
            return total_correct / total_targets
        return 0.0
    
    def _create_patch_masks(self, patch_pred, img_shape):
        """"""
        batch_size, _, img_h, img_w = img_shape
        patch_masks = torch.zeros((batch_size, 1, img_h, img_w), device=self.device)
        
        detections = non_max_suppression(patch_pred, conf_thres=0.05)
        
        for i, det in enumerate(detections):
            if det is not None and len(det) > 0:
                for *box, conf, cls in det:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    patch_masks[i, 0, y1:y2, x1:x2] = 1.0
        
        return patch_masks
    
    def _compute_dependency_loss(self, orig_perf, repaired_perf, epoch):
        """"""

        expected_gap = 0.2
        
        actual_gap = repaired_perf - orig_perf
        gap_shortfall = max(0, expected_gap - actual_gap)
        
        # ：，
        dependency_loss = gap_shortfall * 10.0
        
        # ，
        if epoch > 50:
            dependency_loss *= 0.5
        elif epoch > 30:
            dependency_loss *= 0.7
            
        return torch.tensor(dependency_loss, device=self.device)

# =============================================================================
#  - 
# =============================================================================

class ProgressiveChallengeIncrement:
    """ - """
    
    def __init__(self):
        self.challenge_level = 1
        self.last_repair_gain = 0.0
        self.consecutive_high_gain = 0
    
    def get_challenge_parameters(self, epoch, current_repair_gain):
        """"""
        

        if current_repair_gain > self.last_repair_gain + 0.02:
            self.consecutive_high_gain += 1
        else:
            self.consecutive_high_gain = max(0, self.consecutive_high_gain - 1)
        
        self.last_repair_gain = current_repair_gain
        

        if self.consecutive_high_gain >= 3 and self.challenge_level < 3:
            self.challenge_level += 1
            self.consecutive_high_gain = 0
            LOGGER.info(f"🎯  {self.challenge_level}")
        
        challenge_params = {
            1: {
                'detector_trainable': False, 
                'patch_visibility': 0.6, 
                'detector_lr_scale': 0.0,
                'description': '-'
            },
            2: {
                'detector_trainable': True, 
                'patch_visibility': 0.4, 
                'detector_lr_scale': 0.001,
                'description': '-'
            },
            3: {
                'detector_trainable': True, 
                'patch_visibility': 0.8, 
                'detector_lr_scale': 0.005,
                'description': '-'
            }
        }
        
        return challenge_params.get(self.challenge_level, challenge_params[1])

# =============================================================================

# =============================================================================

class RepairEffectivenessValidation:
    """"""
    
    def __init__(self, stage1_detector):
        self.stage1_detector = stage1_detector
    
    def validate_and_penalize(self, orig_imgs, repaired_imgs, detector, targets, compute_loss):
        """（🚀 ）"""
        
        # 🚀 ：mini-batch
        batch_size = min(4, orig_imgs.shape[0])
        orig_imgs_mini = orig_imgs[:batch_size]
        repaired_imgs_mini = repaired_imgs[:batch_size]
        
        # targets
        targets_mini = targets[targets[:, 0] < batch_size]
        
        if len(targets_mini) == 0:
            return torch.tensor(0.0, device=orig_imgs.device), 0.0
        
        with torch.no_grad():
            # 1. （mini-batch）
            orig_patch_pred = self.stage1_detector(orig_imgs_mini)
            repaired_patch_pred = self.stage1_detector(repaired_imgs_mini)
            
            orig_detections = non_max_suppression(orig_patch_pred, conf_thres=0.05)
            repaired_detections = non_max_suppression(repaired_patch_pred, conf_thres=0.05)
            

            orig_count = sum(len(d) for d in orig_detections if d is not None)
            repaired_count = sum(len(d) for d in repaired_detections if d is not None)
            
            if orig_count > 0:
                reduction_rate = 1.0 - (repaired_count / orig_count)
            else:
                reduction_rate = 1.0  # ，
            

            del orig_patch_pred, repaired_patch_pred, orig_detections, repaired_detections
        
        # 2. （mini-batch）
        detector_pred = detector(repaired_imgs_mini)
        try:
            detector_loss, _ = compute_loss(detector_pred, targets_mini)
        except:
            # loss，0
            detector_loss = torch.tensor(0.0, device=orig_imgs.device)
        
        # 3. ：，（）
        effectiveness_penalty = 0.0
        
        if reduction_rate < 0.3 and detector_loss < 0.1:
            effectiveness_penalty = (0.3 - reduction_rate) * 5.0
            if effectiveness_penalty > 0.5:
                LOGGER.warning(f"⚠️ ，: {effectiveness_penalty:.4f}")
        
        total_loss = detector_loss + effectiveness_penalty
        

        del detector_pred
        
        return total_loss, reduction_rate

# =============================================================================

# =============================================================================

class RealTimeFeedbackSystem:
    """ - """
    
    def __init__(self):
        self.performance_history = []
        self.repair_strength_history = []
        self.adjustment_history = []
        
    def adapt_repair_parameters(self, repair_module, current_eval, previous_eval=None):
        """"""
        
        current_strength = repair_module.repair_strength.item()
        
        if previous_eval is None:
            # ，
            new_strength = current_strength
            adjustment_type = ""
            perf_change = 0.0
        else:

            perf_change = current_eval['comprehensive_score'] - previous_eval['comprehensive_score']
            
            if perf_change > 0.02:
                # ：
                new_strength = min(1.0, current_strength + 0.05)
                adjustment_type = "-"
            elif perf_change < -0.01:
                # ：
                new_strength = max(0.1, current_strength - 0.03)
                adjustment_type = "-"
            else:
                # ：
                new_strength = current_strength
                adjustment_type = "-"
        

        with torch.no_grad():
            repair_module.repair_strength.data = torch.tensor(
                new_strength, 
                device=repair_module.repair_strength.device
            )
        

        self.performance_history.append(current_eval['comprehensive_score'])
        self.repair_strength_history.append(new_strength)
        self.adjustment_history.append({
            'epoch': current_eval.get('epoch', 0),
            'old_strength': current_strength,
            'new_strength': new_strength,
            'adjustment_type': adjustment_type,
            'performance_change': perf_change
        })
        
        return {
            'old_strength': current_strength,
            'new_strength': new_strength,
            'adjustment_type': adjustment_type,
            'performance_change': perf_change
        }

# =============================================================================

# =============================================================================

class CompleteTwoStageDefenseSystem:
    """"""
    
    def __init__(self, opt, device, callbacks):
        self.opt = opt
        self.device = device
        self.callbacks = callbacks
        self.save_dir = Path(opt.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # opt
        self.imgsz = opt.imgsz
        self.detector_type = getattr(opt, 'detector_type', 'yolov5')
        self.cross_detector_mode = getattr(opt, 'cross_detector_mode', False)
        
        # ============  ============
        self.ablation_mode = getattr(opt, 'ablation_mode', False)
        self.ablation_config = {
            # 1: AFAL
            'disable_afal': getattr(opt, 'disable_afal', False),
            'afal_amplitude_only': getattr(opt, 'afal_amplitude_only', False),
            'afal_phase_only': getattr(opt, 'afal_phase_only', False),
            
            # 2 - 
            'disable_detection_loss': getattr(opt, 'disable_detection_loss', False),
            'disable_repair_loss': getattr(opt, 'disable_repair_loss', False),
            'disable_dependency_loss': getattr(opt, 'disable_dependency_loss', False),
            'disable_effectiveness_loss': getattr(opt, 'disable_effectiveness_loss', False),
            
            # 2 - 
            'freq_low_only': getattr(opt, 'freq_low_only', False),
            'freq_high_only': getattr(opt, 'freq_high_only', False),
            'spatial_only': getattr(opt, 'spatial_only', False),
            
            # 2 - 
            'disable_dynamic_strategy': getattr(opt, 'disable_dynamic_strategy', False),
            'disable_feedback_adjustment': getattr(opt, 'disable_feedback_adjustment', False),
            'disable_patch_visibility': getattr(opt, 'disable_patch_visibility', False),
        }
        
        # GPU
        self.multi_gpu = False
        self.gpu_ids = []
        if isinstance(device, str) and ',' in device:
            self.gpu_ids = [int(x.strip()) for x in device.split(',')]
            self.multi_gpu = len(self.gpu_ids) > 1
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
            if self.multi_gpu:
                LOGGER.info(f"🚀 GPU:  {len(self.gpu_ids)} GPU {self.gpu_ids}")
        else:
            self.device = device
            if torch.cuda.is_available():
                self.gpu_ids = [int(str(device).replace('cuda:', ''))] if 'cuda' in str(device) else [0]
        

        self.repair_module = None
        

        self.stage1_patch_detector = None
        self.person_detector = None
        

        self.training_history = {
            'stage1': {'patch_map': [], 'loss': [], 'precision': [], 'recall': []},
            'stage2': {'person_map': [], 'repair_quality': [], 'total_loss': []}
        }
        
        # 🔧 base（AFAL）
        self.is_base_mode = self.ablation_config.get('disable_afal', False)
        

        if self.ablation_mode:
            LOGGER.info(f"\n{'='*80}")
            LOGGER.info(f"🔬 ")
            LOGGER.info(f"{'='*80}")
            LOGGER.info(f"📋 :")
            
            if any(self.ablation_config.values()):
                # 🔧 base
                if self.is_base_mode:
                    LOGGER.info(f"\n  ⭐⭐⭐ BASE ⭐⭐⭐")
                    LOGGER.info(f"  YOLOv5s，")
                    LOGGER.info(f"  train.py")
                    LOGGER.info(f"")
                
                LOGGER.info(f"\n  ====== 1: AFAL ======")
                if self.ablation_config['disable_afal']:
                    LOGGER.info(f"    ❌  (λ_base = 0)")
                elif self.ablation_config['afal_amplitude_only']:
                    LOGGER.info(f"    🔵  (α_align > 0, β_align = 0)")
                elif self.ablation_config['afal_phase_only']:
                    LOGGER.info(f"    🟢  (α_align = 0, β_align > 0)")
                else:
                    LOGGER.info(f"    ✅ AFAL (+)")
                
                LOGGER.info(f"\n  ====== 2:  ======")
                if self.ablation_config['disable_detection_loss']:
                    LOGGER.info(f"    ❌ ")
                if self.ablation_config['disable_repair_loss']:
                    LOGGER.info(f"    ❌ ")
                if self.ablation_config['disable_dependency_loss']:
                    LOGGER.info(f"    ❌  (λ_dep = 0)")
                if self.ablation_config['disable_effectiveness_loss']:
                    LOGGER.info(f"    ❌ ")
                
                LOGGER.info(f"\n  ====== 2:  ======")
                if self.ablation_config['freq_low_only']:
                    LOGGER.info(f"    🔵 ")
                elif self.ablation_config['freq_high_only']:
                    LOGGER.info(f"    🟢 ")
                elif self.ablation_config['spatial_only']:
                    LOGGER.info(f"    🟡 ()")
                else:
                    LOGGER.info(f"    ✅ (+)")
                
                LOGGER.info(f"\n  ====== 2:  ======")
                if self.ablation_config['disable_dynamic_strategy']:
                    LOGGER.info(f"    ❌ ")
                if self.ablation_config['disable_feedback_adjustment']:
                    LOGGER.info(f"    ❌ ")
                if self.ablation_config['disable_patch_visibility']:
                    LOGGER.info(f"    ❌ ")
            else:
                LOGGER.info(f"  ⚠️ （）")
            
            ablation_tag = getattr(opt, 'ablation_tag', '')
            if ablation_tag:
                LOGGER.info(f"\n  🏷️  : {ablation_tag}")
            LOGGER.info(f"{'='*80}\n")
    
    def _create_detector(self, detector_type='yolov5', nc=80, weights_path='', stage='stage2'):
        """
        
        
        Args:
            detector_type:  ('yolov5', 'yolov11', 'yolov12')
            nc: 
            weights_path: 
            stage:  ('stage1'  'stage2')
        
        Returns:
            detector: 
        """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🔧 {stage}: {detector_type}")
        LOGGER.info(f"{'='*60}")
        
        if detector_type == 'yolov5':
            # YOLOv5 (models/yolo.pyModel)
            LOGGER.info(f"  YOLOv5 (cfg={self.opt.cfg})")
            detector = Model(self.opt.cfg, ch=3, nc=nc, anchors=None).to(self.device)
            
            # （）
            if weights_path and Path(weights_path).exists():
                LOGGER.info(f"  : {weights_path}")
                ckpt = torch.load(weights_path, map_location=self.device)
                state_dict = ckpt.get('model', ckpt).state_dict() if hasattr(ckpt.get('model', ckpt), 'state_dict') else ckpt.get('model', ckpt)
                detector.load_state_dict(state_dict, strict=False)
        
        elif detector_type in ['yolov11', 'yolov12']:
            # YOLOv11/v12 (Ultralytics)
            if not HAS_ULTRALYTICS:
                raise ImportError(f"Ultralytics{detector_type}: pip install ultralytics")
            
            # detector_type
            if weights_path and Path(weights_path).exists():
                LOGGER.info(f"  {detector_type}: {weights_path}")
                model = YOLO(weights_path)
                detector = model.model.to(self.device)
            else:

                default_weights = {
                    'yolov11': 'yolo11s.pt',
                    'yolov12': 'yolo12s.pt'
                }
                default_path = f"/path/to/project/detection_models/{default_weights[detector_type]}"
                
                if Path(default_path).exists():
                    LOGGER.info(f"  {detector_type}: {default_path}")
                    model = YOLO(default_path)
                    detector = model.model.to(self.device)
                else:
                    LOGGER.info(f"  {detector_type} (nc={nc})")

                    cfg_name = detector_type + 's.yaml'  #  yolo11s.yaml
                    cfg_path = Path('models') / cfg_name
                    
                    if cfg_path.exists():
                        detector = DetectionModel(str(cfg_path), ch=3, nc=nc).to(self.device)
                    else:
                        raise FileNotFoundError(f": {cfg_path}")
        
        else:
            raise ValueError(f": {detector_type}")
        
        # GPU
        if self.multi_gpu and torch.cuda.device_count() > 1:
            LOGGER.info(f"  GPU: {len(self.gpu_ids)}×GPU {self.gpu_ids}")
            detector = torch.nn.DataParallel(detector, device_ids=self.gpu_ids)
        

        try:
            model_for_check = detector.module if hasattr(detector, 'module') else detector
            param_count = sum(p.numel() for p in model_for_check.parameters())
            LOGGER.info(f"  : {param_count:,}")
            LOGGER.info(f"  : {nc}")
        except Exception as e:
            LOGGER.warning(f"  : {e}")
        
        LOGGER.info(f"{'='*60}\n")
        
        return detector
        
        LOGGER.info(f"\n🚀 : 1- | 2-+")
    
    def stage1_train_complete_patch_detector(self, train_loader, val_loader, epochs=100):
        """1: """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🔍 1:  ({epochs} epochs)")
        LOGGER.info(f"{'='*60}")
        
        # 🔧 data_dict（optnc）
        data_yaml = check_yaml(self.opt.data)
        with open(data_yaml) as f:
            data_dict = yaml.safe_load(f)
        
        # optncnames（）
        nc = getattr(self.opt, 'nc', data_dict.get('nc', 1))
        names = getattr(self.opt, 'names', data_dict.get('names', ['patch']))
        data_dict['nc'] = nc
        data_dict['names'] = names
        
        LOGGER.info(f"📋 : nc={data_dict['nc']}, names={data_dict['names']}")
        
        # （Stage1YOLOv5）
        nc = getattr(self.opt, 'nc', 1)
        LOGGER.info(f"   (nc={nc})")
        self.stage1_patch_detector = self._create_detector(
            detector_type='yolov5',  # Stage1YOLOv5
            nc=nc,
            weights_path='',
            stage='stage1'
        )
        

        try:
            model_for_check = self.stage1_patch_detector.module if hasattr(self.stage1_patch_detector, 'module') else self.stage1_patch_detector
            nc = model_for_check.model[-1].nc
        except:
            pass
        
        # （）
        self._validate_dataset(train_loader)
        
        # anchor
        LOGGER.info(f"  Anchor...")
        self._optimize_anchors_for_patches(train_loader, self.stage1_patch_detector)
        

        LOGGER.info(f"   (wavelet=db6, levels=3)")
        
        from models.frequency_guided_repair import FrequencyGuidedRepairNet, FrequencyConsistencyLoss
        

        self.frequency_repair = FrequencyGuidedRepairNet(
            wavelet='db6', 
            levels=3
        ).to(self.device)
        

        self.freq_consistency_loss = FrequencyConsistencyLoss(
            wavelet='db6',
            levels=3
        )
        
        # GPU
        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.frequency_repair = torch.nn.DataParallel(
                self.frequency_repair,
                device_ids=self.gpu_ids
            )
        
        # 🔧 AFAL (Adaptive Frequency Alignment Loss) 
        # opt
        spectrum_beta = getattr(opt, 'spectrum_beta', 1.0)
        afal_lambda_base = getattr(opt, 'afal_lambda_base', 10.0)
        afal_tau = getattr(opt, 'afal_tau', 0.8)
        afal_kappa = getattr(opt, 'afal_kappa', 10.0)
        afal_alpha = getattr(opt, 'afal_alpha', 0.4)
        
        LOGGER.info(f"  AFAL (α={afal_alpha if not self.ablation_config.get('disable_afal', False) else 0.0}, "
                   f"β={afal_lambda_base}, τ={afal_tau}, κ={afal_kappa}, spectrum_β={spectrum_beta})")
        
        # : AFAL
        if self.ablation_config.get('disable_afal', False):
            self.afal_dwt_forward = None
            self.afal_dwt_inverse = None
            self.wavelet_enhance = None
            self.afal_alpha = 0.0
            self.afal_beta = 0.0
            self.afal_tau = afal_tau
            self.afal_k_slope = afal_kappa
            self.spectrum_beta = 0.0
        else:
            self.afal_dwt_forward = None
            self.afal_dwt_inverse = None
            self.wavelet_enhance = None
            self.afal_alpha = afal_alpha
            self.afal_beta = afal_lambda_base  # λ_base 
            self.afal_tau = afal_tau
            self.afal_k_slope = afal_kappa  # Sigmoid
            self.spectrum_beta = spectrum_beta
        
        #  - smart_optimizerYOLOv5
        optimizer = smart_optimizer(
            model=self.stage1_patch_detector,
            name=self.opt.optimizer,
            lr=self.opt.lr0,
            momentum=0.937,
            decay=self.opt.weight_decay
        )
        

        from utils.general import one_cycle
        lf = one_cycle(1, 0.01, epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        LOGGER.info(f"  LR: one_cycle | : {self.opt.optimizer}")
        

        model_for_loss = self.stage1_patch_detector.module if hasattr(self.stage1_patch_detector, 'module') else self.stage1_patch_detector
        compute_loss = ComputeLoss(model_for_loss)
        ema = ModelEMA(model_for_loss) if self.opt.ema else None
        scaler = amp.GradScaler(enabled=True)
        LOGGER.info(f"  : FP16 | EMA: {self.opt.ema}")
        

        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
       
        
        # （train.py）
        best_patch_map = 0.0
        
        for epoch in range(epochs):
            # 🚀 epoch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            

            self.stage1_patch_detector.train()
            
            mloss = torch.zeros(4, device=self.device)  # AFAL
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=len(train_loader), 
                       bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            
            optimizer.zero_grad()
            
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + len(train_loader) * epoch
                
                #  ：
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                # contiguous()，detach
                imgs = imgs.contiguous().detach().requires_grad_(True)
                targets = targets.to(self.device)
                

                patch_targets = self._filter_patch_targets(targets)
                # 🔧 ：batch，train.py
                # if len(patch_targets) == 0:
                #     continue
                
                #  - 
                # ：
                # ：FP16，AFALFP32
                # ============== AFAL -  ==============
                afal_loss = torch.tensor(0.0, device=self.device)
                
                # 🔥 ：AFAL3batch
                compute_afal = (i % 3 == 0)
                
                # : AFAL
                if not self.ablation_config.get('disable_afal', False) and compute_afal:
                    # 🔥 AFAL：FP16，FP32
                    # 🚀🚀🚀 ：AFAL
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if i % 100 == 0:  # 100batchGC
                        import gc
                        gc.collect()
                    
                    # （）
                    if self.afal_dwt_forward is None:
                        try:
                            from pytorch_wavelets import DWTForward, DWTInverse
                            # 3
                            self.afal_dwt_forward = DWTForward(wave='coif1', J=3, mode='zero').to(self.device)
                            self.afal_dwt_inverse = DWTInverse(wave='coif1', mode='zero').to(self.device)
                            self.wavelet_enhance = WaveletEnhance(device=self.device)
                            LOGGER.info(f"✅ AFAL (J=3) : {self.device}")
                        except ImportError:
                            LOGGER.warning(f"⚠️ pytorch_wavelets，AFAL")
                            self.afal_dwt_forward = None
                            self.afal_dwt_inverse = None
                            self.wavelet_enhance = None
                    
                    if self.afal_dwt_forward is not None:
                            try:
                                # ✅ Full（A）:
                                # AFAL，
                                # Freq，
                                # ：AFAL() + Freq() → 
                                # 🔥 （pytorch_wavelets）
                                with torch.amp.autocast('cuda', enabled=False):
                                    imgs_branch = self.wavelet_enhance(imgs.float())
                                    
                                
                                # �🚀🚀 ：
                                imgsz = imgs.shape[-1]
                                dwt_size = 64  # 🔥 DWT，75%
                                imgs_resized = F.interpolate(
                                    imgs, 
                                    size=(dwt_size, dwt_size), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                                imgs_branch_resized = F.interpolate(
                                    imgs_branch, 
                                    size=(dwt_size, dwt_size), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                                
                                # ，try-catch
                                try:
                                    coeffs_origin = self.afal_dwt_forward(imgs_resized)
                                    coeffs_branch = self.afal_dwt_forward(imgs_branch_resized)
                                except RuntimeError as e:
                                    if "Expected all tensors to be on the same device" in str(e):

                                        from pytorch_wavelets import DWTForward, DWTInverse
                                        self.afal_dwt_forward = DWTForward(wave='coif1', J=4, mode='zero').to(self.device)
                                        self.afal_dwt_inverse = DWTInverse(wave='coif1', mode='zero').to(self.device)
                                        coeffs_origin = self.afal_dwt_forward(imgs_resized)
                                        coeffs_branch = self.afal_dwt_forward(imgs_branch_resized)
                                    else:
                                        raise e
                                
                                # 🔧 ：
                                clean_params = []
                                branch_params = []
                                

                                ll_origin = coeffs_origin[0].detach().clone().requires_grad_(True)
                                clean_params.append(ll_origin)
                                ll_branch = coeffs_branch[0].detach().clone().requires_grad_(True)
                                branch_params.append(ll_branch)
                                
                                # 🔧 ：batch
                                highs_origin = []
                                highs_branch = []
                                
                                # （dwt_size）
                                for level in range(len(coeffs_origin[1])):
                                    level_highs_origin = []
                                    level_highs_branch = []
                                    
                                    for j in range(3):
                                        hf_origin = coeffs_origin[1][level][:, :, j, :, :].detach().clone().requires_grad_(True)
                                        level_highs_origin.append(hf_origin)
                                        clean_params.append(hf_origin)
                                        
                                        hf_branch = coeffs_branch[1][level][:, :, j, :, :].detach().clone().requires_grad_(True)
                                        level_highs_branch.append(hf_branch)
                                        branch_params.append(hf_branch)
                                    
                                    highs_origin.append(torch.stack(level_highs_origin, dim=2))
                                    highs_branch.append(torch.stack(level_highs_branch, dim=2))
                                

                                for param in clean_params:
                                    if not param.requires_grad:
                                        param.requires_grad_(True)
                                
                                for param in branch_params:
                                    if not param.requires_grad:
                                        param.requires_grad_(True)
                                
                                # 🔧 AFAL：、、、
                                vulnerability_values = []
                                adaptive_lambdas = []
                                
                                # 🚀🚀🚀 ：AFAL
                                torch.cuda.empty_cache()
                                
                                try:
                                    # 🔥 AFAL - ，
                                    # 1. 
                                    imgs.requires_grad_(True)
                                    imgs_branch.requires_grad_(True)
                                    
                                    with torch.amp.autocast('cuda', enabled=True):
                                        pred_origin = self.stage1_patch_detector(imgs)
                                        loss_origin = compute_loss(pred_origin, patch_targets)[0]
                                        
                                        pred_enhanced = self.stage1_patch_detector(imgs_branch)
                                        loss_enhanced = compute_loss(pred_enhanced, patch_targets)[0]
                                    
                                    # 2. 
                                    grad_origin = torch.autograd.grad(
                                        loss_origin, imgs, 
                                        retain_graph=True, create_graph=False
                                    )[0]
                                    
                                    grad_enhanced = torch.autograd.grad(
                                        loss_enhanced, imgs_branch,
                                        retain_graph=False, create_graph=False
                                    )[0]
                                    
                                    # 3. （，）
                                    vulnerability = torch.norm(grad_enhanced) / (torch.norm(grad_origin) + 1e-8)
                                    
                                    # 4. lambda
                                    lambda_adaptive = self.afal_beta * torch.sigmoid(self.afal_k_slope * (vulnerability - self.afal_tau))
                                    

                                    del pred_origin, pred_enhanced, loss_origin, loss_enhanced
                                    del grad_origin, grad_enhanced
                                    torch.cuda.empty_cache()
                                    
                                    # 5.  - 
                                    # （）
                                    coeffs_origin = self.afal_dwt_forward(imgs)
                                    coeffs_enhanced = self.afal_dwt_forward(imgs_branch)
                                    
                                    # 6. （+）
                                    # ：lambda_adaptivebeta，
                                    align_loss = torch.tensor(0.0, device=self.device)
                                    

                                    ll_orig, ll_enh = coeffs_origin[0], coeffs_enhanced[0]
                                    mag_loss = F.mse_loss(torch.abs(ll_orig), torch.abs(ll_enh))
                                    phase_loss = 1 - F.cosine_similarity(ll_orig.flatten(1), ll_enh.flatten(1), dim=1).mean()
                                    align_loss += lambda_adaptive * (self.afal_alpha * mag_loss + 0.6 * phase_loss)
                                    
                                    # （：）
                                    for level in range(min(1, len(coeffs_origin[1]))):
                                        for band_idx in range(3):  # LH, HL, HH
                                            hf_orig = coeffs_origin[1][level][:, :, band_idx, :, :]
                                            hf_enh = coeffs_enhanced[1][level][:, :, band_idx, :, :]
                                            
                                            mag_loss_hf = F.mse_loss(torch.abs(hf_orig), torch.abs(hf_enh))
                                            phase_loss_hf = 1 - F.cosine_similarity(hf_orig.flatten(1), hf_enh.flatten(1), dim=1).mean()
                                            align_loss += lambda_adaptive * (self.afal_alpha * mag_loss_hf + 0.6 * phase_loss_hf)
                                    
                                    afal_loss = align_loss / 10.0
                                    

                                    if torch.isnan(afal_loss) or torch.isinf(afal_loss):
                                        LOGGER.warning(f"AFALNaN/Inf，0")
                                        afal_loss = torch.tensor(0.0, device=self.device)
                                    
                                    # 🚀 AFAL
                                    del coeffs_origin, coeffs_enhanced
                                    del ll_orig, ll_enh, hf_orig, hf_enh
                                    del mag_loss, phase_loss, mag_loss_hf, phase_loss_hf
                                    del ll_origin, ll_branch, highs_origin, highs_branch
                                    del vulnerability, lambda_adaptive
                                    torch.cuda.empty_cache()
                                        
                                except Exception as e:
                                    LOGGER.warning(f"AFAL: {e}")
                                    

                                    if 'coeffs_origin' in locals():
                                        del coeffs_origin
                                    if 'coeffs_enhanced' in locals():
                                        del coeffs_enhanced
                                    if 'll_origin' in locals():
                                        del ll_origin, ll_branch, highs_origin, highs_branch
                                    torch.cuda.empty_cache()
                                    
                                    afal_loss = torch.tensor(0.0, device=self.device)
                                
                                # 🔧 ：AFAL
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                    
                            except Exception as e:
                                LOGGER.warning(f"AFAL: {e}")
                                import traceback
                                LOGGER.warning(f": {traceback.format_exc()}")
                                

                                if 'coeffs_origin' in locals():
                                    del coeffs_origin
                                if 'coeffs_branch' in locals():
                                    del coeffs_branch
                                torch.cuda.empty_cache()
                                
                                afal_loss = torch.tensor(0.0, device=self.device)
                    
                # 🔥 FP16
                with torch.amp.autocast('cuda', enabled=True):
                    # （）
                    pred = self.stage1_patch_detector(imgs)
                    loss, loss_items = compute_loss(pred, patch_targets)
                    

                    del pred
                    torch.cuda.empty_cache()
                    

                    if torch.isnan(loss) or torch.isinf(loss):
                        LOGGER.warning(f"NaN/Inf: {loss.item()}")
                        loss = torch.tensor(0.1, device=self.device)
                    
                    # loss_items
                    for idx, item in enumerate(loss_items):
                        if torch.isnan(item) or torch.isinf(item):
                            LOGGER.warning(f"{idx}NaN/Inf: {item.item()}")
                            loss_items[idx] = torch.tensor(0.1, device=self.device)
                
                # 🔧 AFAL（autocast，FP32）
                total_loss = loss + self.afal_alpha * afal_loss
                

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    LOGGER.warning(f"NaN/Inf: {total_loss.item()}")
                    total_loss = torch.tensor(0.1, device=self.device)
                

                loss_items = torch.cat([loss_items, afal_loss.unsqueeze(0)])
                mloss = (mloss * i + loss_items) / (i + 1)
                

                pbar.set_postfix({'loss': f'{mloss[0]:.4f}', 'afal': f'{mloss[3]:.4f}'})
                

                scaler.scale(total_loss).backward()
                

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.stage1_patch_detector.parameters(), max_norm=1.0)
                
                if ni % self.opt.accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 🚀 ：
                    if ni % 10 == 0:
                        torch.cuda.empty_cache()
            
            # EMA
            if ema:
                ema.update(self.stage1_patch_detector)
            
            scheduler.step()
            
            # � GPU（epoch）
            torch.cuda.empty_cache()
            
            # �🔧 （train.py）
            self.callbacks.run('on_train_epoch_end', epoch=epoch)
            if ema:
                ema.update_attr(self.stage1_patch_detector, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            
            # train.py
            model_for_val = ema.ema if ema else self.stage1_patch_detector
            results, maps, _ = validate.run(
                data_dict,
                batch_size=self.opt.batch_size // len(self.gpu_ids) * 2 if self.multi_gpu else self.opt.batch_size * 2,
                imgsz=self.imgsz,
                half=False,  # train.py
                model=model_for_val,
                single_cls=False,
                dataloader=val_loader,
                save_dir=self.save_dir / 'stage1',
                plots=False,
                callbacks=self.callbacks,
                compute_loss=compute_loss
            )
            
            # （train.py）
            precision = results[0]  # P
            recall = results[1]     # R
            patch_map50 = results[2]  # mAP@0.5
            patch_map50_95 = results[3]  # mAP@0.5:0.95
            
            LOGGER.info(f"📊 1 - Epoch {epoch}:")
            LOGGER.info(f"   mAP@0.5: {patch_map50:.4f}")
            LOGGER.info(f"   mAP@0.5:0.95: {patch_map50_95:.4f}")
            
            # epochCSV
            self._save_stage1_results_csv(epoch, patch_map50, patch_map50_95, precision, recall, mloss)
            

            if patch_map50 > best_patch_map:
                best_patch_map = patch_map50
                self._save_complete_stage1_model(epoch, patch_map50, precision, recall, ema, mloss)
            

            self.training_history['stage1']['patch_map'].append(patch_map50)
            self.training_history['stage1']['loss'].append(mloss.mean().item())
            self.training_history['stage1']['precision'].append(precision)
            self.training_history['stage1']['recall'].append(recall)
        
        LOGGER.info(f"✅ 1 -  mAP@0.5: {best_patch_map:.4f}")
        return best_patch_map

    def convert_yolov11_to_yolov5_format(self, pred):
        """
        YOLOv11/v12YOLOv5 NMS
        
        YOLOv11: [B, 84, 8400] = 4(bbox) + 80(class_logits)  <-- objectness
        YOLOv5 NMS: [B, 8400, 85] = 4(bbox) + 1(objectness) + 80(class_probs)
        
        Args:
            pred: YOLOv11 [B, 84, N]  tuple([B, 84, N], ...)  YOLOv5 [B, N, 85]
        Returns:
             [B, N, 85] 
        """
        # 🔥 tuple (YOLOv11/v12evaltuple)
        if isinstance(pred, (tuple, list)):
            if len(pred) > 0:
                pred = pred[0]  # （）
            else:
                return pred
        
        if not isinstance(pred, torch.Tensor):
            return pred
        
        # YOLOv5 [B, N, 85]
        if pred.dim() == 3 and pred.shape[2] == 85:
            return pred
        
        # YOLOv11 [B, 84, N]
        if pred.dim() == 3 and pred.shape[1] == 84:
            B, C, N = pred.shape
            
            # 1.  [B, N, 84]
            pred = pred.permute(0, 2, 1)  # [B, N, 84]
            
            # 2. bboxclass logits
            bbox = pred[:, :, :4]  # [B, N, 4]
            class_logits = pred[:, :, 4:84]  # [B, N, 80]
            
            # 3. objectness：class logitsobjectness
            # ，YOLOv11anchor-free，objectness
            objectness = class_logits.max(dim=2, keepdim=True)[0].sigmoid()  # [B, N, 1]
            
            # 4. class logits
            class_probs = class_logits.sigmoid()  # [B, N, 80]
            
            # 5. YOLOv5
            pred_converted = torch.cat([bbox, objectness, class_probs], dim=2)  # [B, N, 85]
            
            return pred_converted
        
        # （list）
        return pred

    def lightweight_joint_training(self, train_loader, val_loader, epochs=80, stage1_weights_path=None):
        """
        🚀 ： + 
        
        ：
        1. ✅  -  + 
        2. ✅  - 
        3. ✅  - 
        """
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🚀 2: （+）")
        LOGGER.info(f"{'='*80}")
        LOGGER.info(f"：")
        LOGGER.info(f"  ✅  - ")
        LOGGER.info(f"  ✅  - ")
        LOGGER.info(f"  ✅ advpatch_strict - +")
        LOGGER.info(f"  ✅  - ")
        LOGGER.info(f"  ✅  - ")
        LOGGER.info(f"")
        LOGGER.info(f"：")
        LOGGER.info(f"  : {epochs}")
        LOGGER.info(f"  :  + ")
        LOGGER.info(f"  : advpatch_strict (+)")
        LOGGER.info(f"{'='*80}\n")
        
        # 1
        if stage1_weights_path:
            stage1_paths = [Path(stage1_weights_path)]
        else:
            stage1_paths = [
                self.save_dir / 'stage1' / 'weights' / 'best.pt',
                self.save_dir / 'stage1' / 'best.pt',
                self.save_dir / 'stage1' / 'last.pt',
            ]
        
        ckpt = None
        loaded_path = None
        
        for path in stage1_paths:
            if path.exists():
                try:
                    ckpt = torch.load(path, map_location=self.device)
                    loaded_path = path
                    LOGGER.info(f"  1: {path.name}")
                    break
                except Exception as e:
                    continue
        
        if ckpt is None:
            raise FileNotFoundError(f"1: {[str(p) for p in stage1_paths]}")
        
        if loaded_path:
            if self.stage1_patch_detector is None:
                self.stage1_patch_detector = Model(self.opt.cfg, ch=3, nc=1, anchors=None).to(self.device)
            
            if 'model' in ckpt:
                model_state = ckpt['model']
                
                # 🔥 1： nc=1
                expected_nc = 1  # 1patch
                if isinstance(model_state, dict):
                    #  state_dict  nc
                    for key in model_state.keys():
                        if 'model.24' in key or 'model.23' in key:  # YOLOv5 
                            loaded_nc = None
                            try:
                                #  nc
                                if '.m.' in key and '.weight' in key:
                                    shape = model_state[key].shape
                                    #  = (nc + 5) * num_anchors
                                    loaded_nc = (shape[0] // 3) - 5  # 3anchor
                                    if loaded_nc != expected_nc:
                                        LOGGER.warning(f"\n{'='*60}")
                                        LOGGER.warning(f"⚠️   nc ！")
                                        LOGGER.warning(f"    nc={loaded_nc}")
                                        LOGGER.warning(f"    nc={expected_nc}")
                                        LOGGER.warning(f"   1 nc=1")
                                        LOGGER.warning(f"    strict=False ")
                                        LOGGER.warning(f"{'='*60}\n")
                                    break
                            except:
                                pass
                    
                    # 🔥  strict=False ，
                    self.stage1_patch_detector.load_state_dict(model_state, strict=False)
                else:
                    self.stage1_patch_detector.load_state_dict(model_state.state_dict(), strict=False)
                LOGGER.info(f"✅ 1 (nc=1, patch)")
        

        self.stage1_patch_detector.eval()
        for param in self.stage1_patch_detector.parameters():
            param.requires_grad = False
        
        # 🚀 ：
        torch.cuda.empty_cache()
        
        LOGGER.info(f"🔒 1（）\n")
        
        # ============ 2.  ============
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"🔧 （1+2）")
        LOGGER.info(f"{'='*60}")
        
        # 🔥 1: 
        # opt
        repair_strength_init = getattr(self.opt, 'repair_strength_init', 0.3)
        
        self.repair_module = FrequencyGuidedRepairNet(
            wavelet='db6',
            levels=3,
            enable_frequency=True,
            repair_strength_init=repair_strength_init
        ).to(self.device)
        LOGGER.info(f"✅ （1）")
        LOGGER.info(f"   - : db6")
        LOGGER.info(f"   - : 3")
        LOGGER.info(f"   - : /")
        LOGGER.info(f"   - : ")
        LOGGER.info(f"   - : +")
        LOGGER.info(f"   - : {repair_strength_init}")
        LOGGER.info(f"   - : \n")
        
        # 🔥 2: 
        LOGGER.info(f"🔧 （2）...")
        self.freq_consistency_loss = FrequencyConsistencyLoss(
            wavelet='db6',
            levels=3,
            device=self.device,
            enable_frequency=True
        )
        LOGGER.info(f"✅ ")
        LOGGER.info(f"   - : MSE(LL_repaired, LL_clean)")
        LOGGER.info(f"   - : MSE(HF_repaired, HF_clean)")
        LOGGER.info(f"   - : \n")
        
        # ============  ============
        detector_type = getattr(self.opt, 'detector_type', 'yolov5')
        detector_weights = getattr(self.opt, 'detector_weights', '')
        cross_detector_mode = getattr(self.opt, 'cross_detector_mode', False)
        
        if cross_detector_mode:
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info(f"🔬 ")
            LOGGER.info(f"{'='*60}")
            LOGGER.info(f"  Stage1: YOLOv5 ()")
            LOGGER.info(f"  Stage2: {detector_type.upper()} ()")
            if detector_weights:
                LOGGER.info(f"  : {detector_weights}")
            LOGGER.info(f"{'='*60}\n")
        
        # （）
        LOGGER.info(f"🔧 Stage2: {detector_type.upper()}")
        self.person_detector = self._create_detector(
            detector_type=detector_type,
            nc=80,  # COCO80
            weights_path=detector_weights,
            stage='stage2'
        )
        
        # YOLOv5，COCO
        if detector_type == 'yolov5' and not detector_weights:
            self._load_pretrained_weights_for_detector(self.person_detector)
        
        LOGGER.info(f"✅ （nc=80，class=0）")
        
        # GPU ()
        if self.multi_gpu and torch.cuda.device_count() > 1:
            LOGGER.info(f"\n🚀 GPU:  {len(self.gpu_ids)} GPU {self.gpu_ids}")
            self.person_detector = torch.nn.DataParallel(
                self.person_detector, device_ids=self.gpu_ids
            )
            self.repair_module = torch.nn.DataParallel(
                self.repair_module, device_ids=self.gpu_ids
            )
            LOGGER.info(f"   GPU")
        
        # ============ 3.  ============
        model_detector = self.person_detector.module if hasattr(self.person_detector, 'module') else self.person_detector
        model_repair = self.repair_module.module if hasattr(self.repair_module, 'module') else self.repair_module
        
        # 🔑 ，
        optimizer = torch.optim.AdamW([
            {'params': model_repair.parameters(), 'lr': 1e-4},
            {'params': model_detector.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)
        
        LOGGER.info(f"✅ :")
        LOGGER.info(f"   🔧 : AdamW, lr=1e-4")
        LOGGER.info(f"   🔧 : AdamW, lr=5e-4")
        LOGGER.info(f"   💡 : ，")
        

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        
        #  - 
        if self.detector_type.lower() in ['yolov11', 'yolo11', 'yolov12', 'yolo12']:
            # YOLOv11/v12ultralytics
            from ultralytics.utils.loss import v8DetectionLoss
            from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
            
            # : model.args,SimpleNamespace
            if isinstance(model_detector.args, dict):
                print("  🔧 model.args,SimpleNamespace...")
                args_dict = model_detector.args.copy()
                # (DEFAULT_CFG)
                args_dict['box'] = getattr(DEFAULT_CFG, 'box', 7.5)
                args_dict['cls'] = getattr(DEFAULT_CFG, 'cls', 0.5)
                args_dict['dfl'] = getattr(DEFAULT_CFG, 'dfl', 1.5)
                
                # IterableSimpleNamespace
                model_detector.args = IterableSimpleNamespace(**args_dict)
                print(f"  ✅ : box={model_detector.args.box}, cls={model_detector.args.cls}, dfl={model_detector.args.dfl}")
            
            _compute_loss_raw = v8DetectionLoss(model_detector)
            print(f"✅ ultralytics v8DetectionLoss (anchor-free, {self.detector_type})")
            
            # ultralyticsYOLOv5
            def compute_loss(pred, targets):
                """
                ultralytics
                :
                    pred: list of [batch, channels, h, w] - YOLOv11
                    targets: [num_targets, 6] - [img_idx, class, x, y, w, h] (normalized)
                :
                    loss: 
                    loss_items: [lbox, lobj, lcls] (YOLOv5)
                """
                try:
                    # ：targets
                    if targets.dim() != 2:
                        print(f"❌ targets: dim={targets.dim()}, shape={targets.shape}")
                        return torch.tensor(0.0, device=targets.device, requires_grad=True), torch.zeros(3, device=targets.device)
                    
                    if targets.shape[1] != 6:
                        print(f"❌ targets: shape={targets.shape}, expected [N, 6]")
                        return torch.tensor(0.0, device=targets.device, requires_grad=True), torch.zeros(3, device=targets.device)
                    
                    # ultralyticsdict: {'batch_idx', 'cls', 'bboxes'}
                    # : [num_targets, 6] - [img_idx, class, x, y, w, h]
                    # ultralytics
                    batch_dict = {
                        'batch_idx': targets[:, 0].long(),  # [num_targets] - 
                        'cls': targets[:, 1:2],              # [num_targets, 1] - 
                        'bboxes': targets[:, 2:6],           # [num_targets, 4] - xywh
                    }
                    
                    # ultralytics
                    loss_tuple = _compute_loss_raw(pred, batch_dict)
                    
                    # YOLOv5
                    if isinstance(loss_tuple, tuple):
                        #  (loss, loss_items)
                        total_loss = loss_tuple[0]
                        loss_items = loss_tuple[1] if len(loss_tuple) > 1 else torch.zeros(3, device=targets.device)
                    elif isinstance(loss_tuple, dict):
                        #  {'loss': xxx, 'lbox': xxx, ...}
                        total_loss = sum(v for k, v in loss_tuple.items() if 'loss' in k.lower())
                        loss_items = torch.tensor([
                            loss_tuple.get('box_loss', 0),
                            loss_tuple.get('cls_loss', 0),
                            loss_tuple.get('dfl_loss', 0)
                        ], device=targets.device)
                    else:

                        total_loss = loss_tuple
                        loss_items = torch.zeros(3, device=targets.device)
                    
                    return total_loss, loss_items
                except Exception as e:

                    print(f"⚠️ ultralytics: {e}, ")
                    return torch.tensor(0.0, device=targets.device, requires_grad=True), torch.zeros(3, device=targets.device)
        else:
            # YOLOv5
            compute_loss = ComputeLoss(model_detector)
            print(f"✅ YOLOv5 ComputeLoss (anchor-based)")
        
        # ============ 4. （） ============
        best_person_map = 0.0
        best_comprehensive_score = 0.0
        

        dynamic_strategy = DynamicTrainingStrategy(total_epochs=epochs)
        feedback_system = RealTimeFeedbackSystem()
        

        repair_dependency_enforcer = RepairDependencyEnforcement(
            self.stage1_patch_detector, self.device, self.detector_type
        )
        progressive_challenge = ProgressiveChallengeIncrement()
        repair_effectiveness_validator = RepairEffectivenessValidation(
            self.stage1_patch_detector
        )
        
        previous_curriculum_eval = None
        current_adv_map = None
        current_repair_gain = 0.0
        
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🔥 （）")
        LOGGER.info(f"  ✅ ：，")
        LOGGER.info(f"  ✅ ：，")
        LOGGER.info(f"  ✅ ：")
        LOGGER.info(f"  ✅ ：mAP")
        LOGGER.info(f"  🔥 ： +  + ")
        LOGGER.info(f"  🚀 ：")
        LOGGER.info(f"{'='*80}\n")
        
        # 🚀 ：
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        LOGGER.info(f"🚀 ： PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        

        torch.cuda.empty_cache()
        LOGGER.info(f"🚀 CUDA\n")
        
        # ✅ Epoch -1: （）
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🔍 Epoch -1: （）")
        LOGGER.info(f"{'='*80}")
        
        baseline_eval = self.comprehensive_evaluation(val_loader)
        
        LOGGER.info(f"\n📊 :")
        LOGGER.info(f"  mAP (conf=0.25): {baseline_eval['adv_performance']:.4f}")
        LOGGER.info(f"  mAP (conf=0.25):   {baseline_eval['repaired_performance']:.4f}")
        
        if baseline_eval['adv_performance'] > 0.5:
            LOGGER.warning(f"\n⚠️ ：mAP({baseline_eval['adv_performance']:.4f})！")
            LOGGER.warning(f"   ")
            LOGGER.warning(f"   ：")
            LOGGER.warning(f"     1) ")
            LOGGER.warning(f"     2) ")
        else:
            LOGGER.info(f"\n✅ ！mAP={baseline_eval['adv_performance']:.4f} < 0.5")
        
        LOGGER.info(f"{'='*80}\n")
        
        for epoch in range(epochs):
            # 🔥 epoch
            strategy = dynamic_strategy.get_strategy(epoch, current_adv_map)
            
            # 🔥 （）
            if epoch > 0:  # epoch
                challenge_params = progressive_challenge.get_challenge_parameters(
                    epoch, current_repair_gain
                )

                if current_repair_gain > 0.08:
                    strategy['detector_trainable'] = challenge_params['detector_trainable']
                    strategy['patch_visibility'] = challenge_params['patch_visibility']
                    strategy['detector_lr_scale'] = challenge_params['detector_lr_scale']
                    LOGGER.info(f"  🎯 : {challenge_params['description']}")
            

            ablation_skip_repair = (
                self.ablation_config.get('disable_dynamic_strategy', False) and
                self.ablation_config.get('disable_feedback_adjustment', False) and
                self.ablation_config.get('disable_patch_visibility', False)
            )
            
            # 🔥 （，）
            if ablation_skip_repair:
                strategy['detector_trainable'] = True
                LOGGER.info(f"\n[Epoch {epoch}] {strategy['stage']} | :✓() | 🔥 :  ()")
            else:
                det_status = "✓" if strategy['detector_trainable'] else "✗"
                LOGGER.info(f"\n[Epoch {epoch}] {strategy['stage']} | :{det_status} :{strategy['repair_strength']:.2f} :{strategy['patch_visibility']:.2f}")
            

            for param in self.person_detector.parameters():
                param.requires_grad = strategy['detector_trainable']
            

            if strategy['detector_trainable'] and strategy.get('detector_lr_scale', 1.0) > 0:
                self.person_detector.train()
                base_detector_lr = optimizer.param_groups[1]['lr']
                scaled_detector_lr = base_detector_lr * strategy.get('detector_lr_scale', 1.0)
                for param_group in optimizer.param_groups[1:]:
                    param_group['lr'] = scaled_detector_lr
            else:
                self.person_detector.train()
            
            self.repair_module.train()
            

            epoch_losses = {
                'total': 0.0,
                'detection': 0.0,
                'enhanced_repair': 0.0
            }
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f'Dynamic Training Epoch {epoch}/{epochs-1}')
            
            for imgs, targets, paths, _ in pbar:
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device)
                batch_size = imgs.shape[0]
                
                # 🔥 （advpatch_strict）

                target_size = (imgs.shape[2], imgs.shape[3])  # (H, W)
                clean_imgs = self._load_clean_images(paths, target_size=target_size).to(self.device, non_blocking=True).float() / 255.0
                
                # 🔥 ：person targets
                # targets: [img_idx, class_id, x, y, w, h]
                if len(targets) == 0:
                    continue
                    
                # person (class_id == 0)
                person_mask = targets[:, 1] == 0
                person_targets = targets[person_mask]
                
                if len(person_targets) == 0:
                    continue
                
                # 🔥 ：img_idxbatch_sizetargets
                # DataParallel
                valid_idx_mask = person_targets[:, 0] < batch_size
                person_targets = person_targets[valid_idx_mask]
                
                # ：person_targets
                if person_targets.dim() != 2 or person_targets.shape[1] != 6:
                    if batch_count < 3:
                        LOGGER.warning(f"[Batch {batch_count}] ❌ person_targets: {person_targets.shape}")
                    continue
                
                if len(person_targets) == 0:
                    continue
                
                # ============ 🔥  ============
                
                # 1. 1（，）
                with torch.no_grad():
                    patch_pred = self.stage1_patch_detector(imgs)
                    full_patch_masks = self._create_complete_patch_masks(patch_pred, imgs.shape, conf_thres=0.05)
                
                # 2. 🔥 ：（）
                dynamic_patch_masks = self._apply_dynamic_patch_visibility(
                    imgs, full_patch_masks, strategy['patch_visibility'], fast_mode=True
                )
                
                # 3. 
                model_repair = self.repair_module.module if hasattr(self.repair_module, 'module') else self.repair_module
                with torch.no_grad():
                    model_repair.repair_strength.data = torch.tensor(
                        strategy['repair_strength'], 
                        device=self.device
                        )
                
                # 4. ✅ ：
                # 🔥 ：，
                ablation_skip_repair = (
                    self.ablation_config.get('disable_dynamic_strategy', False) and
                    self.ablation_config.get('disable_feedback_adjustment', False) and
                    self.ablation_config.get('disable_patch_visibility', False)
                )
                
                # 🔍 ：
                if batch_count < 5 and epoch < 3:
                    LOGGER.info(f"  [Batch {batch_count}] :")
                    LOGGER.info(f"    full_patch_masks.sum(): {full_patch_masks.sum().item():.0f}")
                    LOGGER.info(f"    dynamic_patch_masks.sum(): {dynamic_patch_masks.sum().item():.0f}")
                    LOGGER.info(f"    ablation_skip_repair: {ablation_skip_repair}")
                
                if ablation_skip_repair:
                    # ：，
                    repaired_imgs = imgs
                else:
                    # ✅ ：，！
                    # ， total_loss.requires_grad = False
                    repair_result = self.repair_module(imgs, dynamic_patch_masks)
                    if isinstance(repair_result, tuple):
                        repaired_imgs = repair_result[0]
                    else:
                        repaired_imgs = repair_result
                
                # 5. 🔥 （，）
                try:
                    if strategy['detector_trainable']:
                        # ：+
                        person_pred = self.person_detector(repaired_imgs)
                        detection_loss, loss_items = compute_loss(person_pred, person_targets)
                        # ✅ YOLOv11
                        if isinstance(detection_loss, torch.Tensor) and detection_loss.numel() > 1:
                            detection_loss = detection_loss.sum()
                    else:
                        # ：
                        # ❌ ：with torch.no_grad() - 
                        # ✅ ：，（requires_grad=False）
                        person_pred = self.person_detector(repaired_imgs)
                        detection_loss, loss_items = compute_loss(person_pred, person_targets)
                        # ✅ YOLOv11
                        if isinstance(detection_loss, torch.Tensor) and detection_loss.numel() > 1:
                            detection_loss = detection_loss.sum()
                except (IndexError, RuntimeError, ValueError, TypeError) as e:
                    # ，batch
                    if batch_count < 5:
                        LOGGER.warning(f"  [Batch {batch_count}] : {e}")
                        LOGGER.warning(f"    targets shape: {person_targets.shape}, batch_size: {batch_size}")
                        if len(person_targets) > 0:
                            LOGGER.warning(f"    targets sample: img_idx={person_targets[0, 0].item():.0f}, class={person_targets[0, 1].item():.0f}")

                        if 'person_pred' in locals():
                            for pi, pred_i in enumerate(person_pred):
                                LOGGER.warning(f"    pred[{pi}] shape: {pred_i.shape}")
                    continue
                
                # 6. （，🚀 30-40%）
                # 🔥 ：，0
                if ablation_skip_repair:
                    enhanced_repair_loss = detection_loss * 0.0
                else:
                    enhanced_repair_loss = self._compute_enhanced_repair_loss(
                        imgs, repaired_imgs, dynamic_patch_masks, clean_imgs=clean_imgs, lightweight=True
                    )
                    
                    # 🔍 ：
                    if batch_count < 5 and epoch < 3:
                        LOGGER.info(f"  [Batch {batch_count}] :")
                        LOGGER.info(f"    repaired_imgs.requires_grad: {repaired_imgs.requires_grad}")
                        LOGGER.info(f"    enhanced_repair_loss.requires_grad: {enhanced_repair_loss.requires_grad}")
                        LOGGER.info(f"    enhanced_repair_loss.grad_fn: {enhanced_repair_loss.grad_fn}")
                        LOGGER.info(f"    enhanced_repair_loss.value: {enhanced_repair_loss.item():.6f}")
                
                # 🔥 7. （🚀 ：，epoch）
                # 🔧 tensor，
                dependency_loss = detection_loss * 0.0
                # � ：，
                # �🚀 ：5batch20batch，epoch
                if (not ablation_skip_repair and
                    strategy['detector_trainable'] and 
                    batch_count % 20 == 0 and 
                    epoch >= 30 and 
                    epoch % 5 == 0):  # epoch 30, 35, 40...

                    try:
                        # 🚀 ：
                        torch.cuda.empty_cache()
                        
                        dependency_loss, orig_perf, repaired_perf = repair_dependency_enforcer.enforce_repair_dependency(
                            self.person_detector, self.repair_module, imgs, person_targets, epoch
                        )
                        if batch_count % 50 == 0:

                            dep_loss_val = dependency_loss.item() if dependency_loss.numel() == 1 else dependency_loss.sum().item()
                            LOGGER.info(f"  [Batch {batch_count}]  - : {orig_perf:.3f}, : {repaired_perf:.3f}, : {dep_loss_val:.4f}")
                        
                        # 🚀 ：
                        torch.cuda.empty_cache()
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if "out of memory" in str(e).lower():
                            LOGGER.warning(f"  [Batch {batch_count}] ")
                            torch.cuda.empty_cache()
                        dependency_loss = detection_loss * 0.0
                    except Exception as e:
                        if batch_count < 5:
                            LOGGER.warning(f"  [Batch {batch_count}] : {e}")
                        dependency_loss = detection_loss * 0.0
                
                # 🔥 8. （🚀 ：，epoch）
                effectiveness_penalty = detection_loss * 0.0
                reduction_rate = 0.0
                # � ：，
                # �🚀 ：10batch30batch，epoch
                if (not ablation_skip_repair and
                    strategy['detector_trainable'] and 
                    batch_count % 30 == 0 and 
                    epoch >= 35 and 
                    epoch % 5 == 0):  # epoch 35, 40, 45...

                    try:
                        # 🚀 ：
                        torch.cuda.empty_cache()
                        
                        validated_loss, reduction_rate = repair_effectiveness_validator.validate_and_penalize(
                            imgs, repaired_imgs, self.person_detector, person_targets, compute_loss
                        )
                        # （validated_loss，）
                        effectiveness_penalty = validated_loss - detection_loss
                        if batch_count % 50 == 0:
                            LOGGER.info(f"  [Batch {batch_count}]  - : {reduction_rate:.3f}, : {effectiveness_penalty.item():.4f}")
                        
                        # 🚀 ：
                        torch.cuda.empty_cache()
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if "out of memory" in str(e).lower():
                            LOGGER.warning(f"  [Batch {batch_count}] ")
                            torch.cuda.empty_cache()
                        effectiveness_penalty = detection_loss * 0.0
                    except Exception as e:
                        if batch_count < 5:
                            LOGGER.warning(f"  [Batch {batch_count}] : {e}")
                        effectiveness_penalty = detection_loss * 0.0
                
                # 🔍 ：（10batch）
                if batch_count < 10 and epoch < 3:

                    det_loss_val = detection_loss.item() if detection_loss.numel() == 1 else detection_loss.sum().item()
                    repair_loss_val = enhanced_repair_loss.item() if enhanced_repair_loss.numel() == 1 else enhanced_repair_loss.sum().item()
                    LOGGER.info(f"  [Batch {batch_count}] : {det_loss_val:.6f}, "
                               f": {repair_loss_val:.6f}, "
                               f": {dynamic_patch_masks.sum().item():.0f}")
                
                # 9. 🔥 （）
                if strategy['detector_trainable']:

                    det_scalar = detection_loss if detection_loss.numel() == 1 else detection_loss.sum()
                    repair_scalar = enhanced_repair_loss if enhanced_repair_loss.numel() == 1 else enhanced_repair_loss.sum()
                    dep_scalar = dependency_loss if dependency_loss.numel() == 1 else dependency_loss.sum()
                    eff_scalar = effectiveness_penalty if effectiveness_penalty.numel() == 1 else effectiveness_penalty.sum()
                    
                    total_loss = (
                        strategy['detection_weight'] * det_scalar +
                        strategy['repair_weight'] * repair_scalar +
                        0.1 * dep_scalar +  # 0.1
                        0.1 * eff_scalar  # 0.1
                    )
                else:
                    # ✅ ：（）
                    # ，
                    total_loss = enhanced_repair_loss if enhanced_repair_loss.numel() == 1 else enhanced_repair_loss.sum()
                    
                    # 🔧 ：，detached
                    # ，no grad
                
                # 🔧 ：total_loss
                if not total_loss.requires_grad:
                    LOGGER.warning(f"  [Batch {batch_count}] ：total_loss，batch")
                    continue
                
                # 8. 
                optimizer.zero_grad()
                total_loss.backward()
                
                # 9. （）
                if strategy['detector_trainable']:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=5.0)
                
                optimizer.step()
                
                # 🔥 ：batch
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
                
                # --------  -------- 

                det_loss_scalar = detection_loss.item() if detection_loss.numel() == 1 else detection_loss.sum().item()
                repair_loss_scalar = enhanced_repair_loss.item() if enhanced_repair_loss.numel() == 1 else enhanced_repair_loss.sum().item()
                
                epoch_losses['total'] += total_loss.item()
                epoch_losses['detection'] += det_loss_scalar
                epoch_losses['enhanced_repair'] += repair_loss_scalar
                batch_count += 1
                

                pbar.set_postfix({
                    'total': f"{total_loss.item():.3f}",
                    'det': f"{det_loss_scalar:.3f}",
                    'rep': f"{repair_loss_scalar:.3f}",
                    'vis': f"{strategy['patch_visibility']:.2f}",
                    'det_train': '✓' if strategy['detector_trainable'] else '✗'
                })
                
                # 🚀 ：
                if batch_count % 5 == 0:
                    torch.cuda.empty_cache()
                
                # 🚀 ：
                del repaired_imgs, person_pred, detection_loss, enhanced_repair_loss
                del dynamic_patch_masks, total_loss

                if 'dependency_loss' in locals():
                    dep_val = dependency_loss.item() if dependency_loss.numel() == 1 else dependency_loss.sum().item()
                    if dep_val > 0:
                        del dependency_loss
                if 'effectiveness_penalty' in locals():
                    eff_val = effectiveness_penalty.item() if effectiveness_penalty.numel() == 1 else effectiveness_penalty.sum().item()
                    if eff_val > 0:
                        del effectiveness_penalty
            

            scheduler.step()
            
            # Epoch
            if batch_count > 0:
                avg_total_loss = epoch_losses['total'] / batch_count
                avg_detection_loss = epoch_losses['detection'] / batch_count
                avg_repair_loss = epoch_losses['enhanced_repair'] / batch_count
            
            # ============  ============
            if epoch % 5 == 0:  # 5
                LOGGER.info(f"\n{'='*80}")
                LOGGER.info(f"🔍  - Epoch {epoch}/{epochs-1}")
                LOGGER.info(f"{'='*80}")
                

                eval_results = self.comprehensive_evaluation(val_loader)
                

                torch.cuda.empty_cache()
                
                # 🔥 mAP
                current_adv_map = eval_results['adv_performance']
                current_repair_gain = eval_results.get('repair_gain', 0.0)
                
                LOGGER.info(f"  ✅  - mAP: {current_adv_map:.4f}, : {current_repair_gain:.4f}")
                

                curriculum_eval = self.adaptive_curriculum_evaluation(epoch, epochs, eval_results)
                

                model_repair = self.repair_module.module if hasattr(self.repair_module, 'module') else self.repair_module
                feedback_result = feedback_system.adapt_repair_parameters(
                    model_repair, curriculum_eval, previous_curriculum_eval
                )
                
                # 🆕 （）
                if not hasattr(self, 'adaptive_repair_weights'):
                    self.adaptive_repair_weights = {
                        'patch_region': 2.0,
                        'perceptual': 0.5,
                        'freq_low': 0.5,
                        'repair_gain_penalty': 3.0
                    }
                

                repair_quality = eval_results.get('repair_quality', 0.0)  # SSIM, [0,1]
                repair_gain = eval_results.get('average_repair_gain', 0.0)
                target_quality = 0.85  # SSIM
                target_gain = 0.15
                
                quality_gap = target_quality - repair_quality
                

                if quality_gap > 0.15:  #  (SSIM < 0.70)
                    self.adaptive_repair_weights['patch_region'] = min(5.0, self.adaptive_repair_weights['patch_region'] * 1.3)
                    self.adaptive_repair_weights['perceptual'] = min(2.0, self.adaptive_repair_weights['perceptual'] * 1.3)
                    self.adaptive_repair_weights['freq_low'] = min(2.0, self.adaptive_repair_weights['freq_low'] * 1.3)
                    adjustment_msg = f"({repair_quality:.3f})，"
                elif quality_gap > 0.05:  #  (SSIM 0.70-0.80)
                    self.adaptive_repair_weights['patch_region'] = min(4.0, self.adaptive_repair_weights['patch_region'] * 1.1)
                    self.adaptive_repair_weights['perceptual'] = min(1.5, self.adaptive_repair_weights['perceptual'] * 1.1)
                    adjustment_msg = f"({repair_quality:.3f})，"
                elif quality_gap < -0.05:  #  (SSIM > 0.90)
                    self.adaptive_repair_weights['patch_region'] = max(1.0, self.adaptive_repair_weights['patch_region'] * 0.9)
                    self.adaptive_repair_weights['perceptual'] = max(0.3, self.adaptive_repair_weights['perceptual'] * 0.9)
                    adjustment_msg = f"({repair_quality:.3f})，"
                else:
                    adjustment_msg = f"({repair_quality:.3f})，"
                

                if repair_gain < 0.05:
                    self.adaptive_repair_weights['repair_gain_penalty'] = min(10.0, self.adaptive_repair_weights['repair_gain_penalty'] * 1.5)
                    self.adaptive_repair_weights['patch_region'] = min(5.0, self.adaptive_repair_weights['patch_region'] * 1.2)
                    adjustment_msg += f" | ({repair_gain:.3f})，"
                elif repair_gain < target_gain:
                    self.adaptive_repair_weights['repair_gain_penalty'] = min(8.0, self.adaptive_repair_weights['repair_gain_penalty'] * 1.2)
                    adjustment_msg += f" | ({repair_gain:.3f})，"
                elif repair_gain > target_gain + 0.05:
                    self.adaptive_repair_weights['repair_gain_penalty'] = max(2.0, self.adaptive_repair_weights['repair_gain_penalty'] * 0.95)
                    adjustment_msg += f" | ({repair_gain:.3f})，"
                
                LOGGER.info(f"🔧 : {adjustment_msg}")
                LOGGER.info(f"  : {self.adaptive_repair_weights['patch_region']:.2f}")
                LOGGER.info(f"  : {self.adaptive_repair_weights['perceptual']:.2f}")
                LOGGER.info(f"  : {self.adaptive_repair_weights['freq_low']:.2f}")
                LOGGER.info(f"  : {self.adaptive_repair_weights['repair_gain_penalty']:.2f}")
                

                LOGGER.info(f"\n[Epoch {epoch}] {curriculum_eval['stage']} | mAP:{eval_results['adv_performance']:.3f} mAP:{eval_results['repaired_performance']:.3f}(+{eval_results['average_repair_gain']:.3f}) | :{eval_results['repair_quality']:.3f} :{curriculum_eval['comprehensive_score']:.3f}")
                

                if curriculum_eval['comprehensive_score'] > best_comprehensive_score:
                    best_comprehensive_score = curriculum_eval['comprehensive_score']
                    best_person_map = max(best_person_map, eval_results['repaired_performance'])
                    
                    self._save_lightweight_joint_model(
                        epoch, 
                        eval_results['repaired_performance'], 
                        eval_results['repair_quality'],
                        comprehensive_eval=curriculum_eval
                    )
                    
                    LOGGER.info(f"  ✅  (score={best_comprehensive_score:.3f})")
                
                # CSV（ + ）
                self._save_comprehensive_csv(
                    epoch, 
                    eval_results, 
                    curriculum_eval, 
                    epoch_losses, 
                    batch_count,
                    feedback_result,
                    strategy
                )
                

                previous_curriculum_eval = curriculum_eval
        
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"✅ 2 | mAP:{best_person_map:.3f} :{best_comprehensive_score:.3f}")
        LOGGER.info(f"{'='*60}")
        
        return best_person_map
    
    def _compute_unified_loss(self, orig_imgs, repaired_imgs, person_pred, targets, patch_masks, compute_loss):
        """ - """
        
        # 1. （）
        detection_loss, _ = compute_loss(person_pred, targets)
        
        # 2. （）
        repair_loss = torch.tensor(0.0, device=self.device)
        if patch_masks is not None and patch_masks.sum() > 0:

            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
                
            repair_change = torch.abs(repaired_imgs - orig_imgs) * patch_masks_3c
            repair_loss = torch.mean(repair_change) * 0.1
        
        # 3. 
        consistency_loss = torch.tensor(0.0, device=self.device)
        if patch_masks is not None:
            non_patch_masks = 1 - patch_masks.repeat(1, 3, 1, 1)
            consistency_loss = F.l1_loss(
                orig_imgs * non_patch_masks, 
                repaired_imgs * non_patch_masks
            ) * 0.05
        
        # 4. 
        total_loss = detection_loss + repair_loss + consistency_loss
        

        det_loss_val = detection_loss.item() if detection_loss.numel() == 1 else detection_loss.sum().item()
        
        loss_dict = {
            'detection': det_loss_val,
            'repair': repair_loss.item(),
            'consistency': consistency_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _get_lightweight_curriculum(self, epoch, total_epochs):
        """"""
        progress = epoch / total_epochs
        
        if progress < 0.4:
            # ：，
            return {'repair_strength_scale': 0.5, 'stage': '-'}
        elif progress < 0.7:
            # ：
            return {'repair_strength_scale': 0.8, 'stage': '-'}  
        else:
            # ：，
            return {'repair_strength_scale': 1.0, 'stage': '-'}
    
    def comprehensive_evaluation(self, val_loader):
        """
         -  + 
        
        🎯 ：
        1. conf_thres=0.001: COCO mAP（）
        2. conf_thres=0.25: （）
        """
        self.repair_module.eval()
        self.person_detector.eval()
        self.stage1_patch_detector.eval()
        
        # ：(0.001) vs (0.25)
        adv_stats_standard = []      # conf=0.001
        repaired_stats_standard = []
        adv_stats_practical = []     # conf=0.25
        repaired_stats_practical = []
        repair_qualities = []
        
        # 🔍 
        diagnostic_stats = {
            'total_images': 0,
            'images_with_patches': 0,
            'images_with_persons': 0,
            'total_patches_detected': 0,
            'total_person_labels': 0,
            'images_repaired': 0,
            'images_clean': 0      # ✅ （）
        }
        

        LOGGER.info(f"  📊 （ {len(val_loader)} ）...")
        
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc='', 
                               bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                               ncols=100)
            
            batch_idx = 0
            for imgs, targets, paths, shapes in val_iterator:
                imgs = imgs.to(self.device).float() / 255.0
                targets = targets.to(self.device)
                nb, _, height, width = imgs.shape
                
                # ✅ ：batch
                if batch_idx == 0:
                    LOGGER.info(f"\n🔍 （1batch）:")
                    LOGGER.info(f"  Batch: {nb}")
                    LOGGER.info(f"  : {paths[0]}")
                    first_img_labels = targets[targets[:, 0] == 0, 1:]
                    LOGGER.info(f"  1: {len(first_img_labels)}")
                    if len(first_img_labels) > 0:
                        LOGGER.info(f"  1: {first_img_labels[:, 0].cpu().tolist()}")
                        LOGGER.info(f"  1（）: {first_img_labels[0].cpu().tolist()}")
                        if (first_img_labels[:, 0] >= 0.5).any():
                            LOGGER.warning(f"  ⚠️ （class >= 0.5）！")
                        else:
                            LOGGER.info(f"  ✅ （class < 0.5）")
                batch_idx += 1
                
                # targets
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
                
                # 1. （）- 
                adv_pred = self.person_detector(imgs)
                # 🔥 YOLOv11YOLOv5 NMS
                adv_pred = self.convert_yolov11_to_yolov5_format(adv_pred)
                
                #  (COCO mAP)
                adv_nms_standard = non_max_suppression(
                    adv_pred, conf_thres=0.001, iou_thres=0.6,
                    multi_label=True, max_det=300
                )
                

                adv_nms_practical = non_max_suppression(
                    adv_pred, conf_thres=0.25, iou_thres=0.6,
                    multi_label=True, max_det=300
                )
                
                # 🔍 ：
                diagnostic_stats['total_images'] += nb
                for si in range(nb):
                    labels = targets[targets[:, 0] == si, 1:]
                    person_labels = labels[labels[:, 0] < 0.5]
                    if len(person_labels) > 0:
                        diagnostic_stats['images_with_persons'] += 1
                        diagnostic_stats['total_person_labels'] += len(person_labels)
                
                # 2. （✅ ：0.05）
                patch_pred = self.stage1_patch_detector(imgs)
                patch_masks = self._create_complete_patch_masks(patch_pred, imgs.shape, conf_thres=0.05)
                
                # 🔍 ：（bug）
                patch_nms_for_diag = non_max_suppression(
                    patch_pred, conf_thres=0.05, iou_thres=0.6, max_det=300
                )
                for img_idx, det in enumerate(patch_nms_for_diag):
                    if det is not None and len(det) > 0:
                        diagnostic_stats['images_with_patches'] += 1
                        diagnostic_stats['total_patches_detected'] += len(det)
                
                # ✅ ：
                repaired_imgs = torch.zeros_like(imgs)
                for img_idx in range(nb):
                    img_mask = patch_masks[img_idx:img_idx+1]
                    if img_mask.sum() > 0:
                        # ，
                        # 🔥 （）
                        repair_result = self.repair_module(
                            imgs[img_idx:img_idx+1], img_mask
                        )
                        if isinstance(repair_result, tuple):
                            repaired_imgs[img_idx:img_idx+1] = repair_result[0]
                        else:
                            repaired_imgs[img_idx:img_idx+1] = repair_result
                        diagnostic_stats['images_repaired'] += 1
                    else:
                        # ，
                        repaired_imgs[img_idx:img_idx+1] = imgs[img_idx:img_idx+1]
                        diagnostic_stats['images_clean'] += 1
                

                repair_quality = self._compute_ssim_loss(imgs, repaired_imgs)
                repair_qualities.append(repair_quality.item())
                
                # 3.  - 
                repaired_pred = self.person_detector(repaired_imgs)
                # 🔥 YOLOv11YOLOv5 NMS
                repaired_pred = self.convert_yolov11_to_yolov5_format(repaired_pred)
                
                #  (COCO mAP)
                repaired_nms_standard = non_max_suppression(
                    repaired_pred, conf_thres=0.001, iou_thres=0.6,
                    multi_label=True, max_det=300
                )
                

                repaired_nms_practical = non_max_suppression(
                    repaired_pred, conf_thres=0.25, iou_thres=0.6,
                    multi_label=True, max_det=300
                )
                
                # 4. （ - ）
                for si, p in enumerate(adv_nms_standard):
                    labels = targets[targets[:, 0] == si, 1:]
                    person_labels = labels[labels[:, 0] < 0.5].clone() if len(labels) > 0 else torch.zeros(0, 5, device=self.device)
                    
                    if len(person_labels) == 0:
                        if len(p) > 0:
                            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                            correct = torch.zeros(len(p), iouv.numel(), dtype=torch.bool, device=self.device)
                            adv_stats_standard.append((correct.cpu(), p[:, 4].cpu(), p[:, 5].cpu(), []))
                        continue
                    
                    person_labels[:, 0] = 0
                    
                    if len(p) == 0:
                        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                        adv_stats_standard.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), 
                                        torch.Tensor(), torch.Tensor(), 
                                        person_labels[:, 0].cpu().tolist()))
                        continue
                    

                    shape_i = shapes[si][0]
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])
                    predn[:, 5] = 0
                    
                    tbox = xywh2xyxy(person_labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])
                    labelsn = torch.cat((person_labels[:, 0:1], tbox), 1)
                    
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                    correct = self._process_batch(predn, labelsn, iouv)
                    adv_stats_standard.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), person_labels[:, 0].cpu()))
                
                # 4.5. （ - ）
                for si, p in enumerate(adv_nms_practical):
                    labels = targets[targets[:, 0] == si, 1:]
                    person_labels = labels[labels[:, 0] < 0.5].clone() if len(labels) > 0 else torch.zeros(0, 5, device=self.device)
                    
                    if len(person_labels) == 0:
                        if len(p) > 0:
                            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                            correct = torch.zeros(len(p), iouv.numel(), dtype=torch.bool, device=self.device)
                            adv_stats_practical.append((correct.cpu(), p[:, 4].cpu(), p[:, 5].cpu(), []))
                        continue
                    
                    person_labels[:, 0] = 0
                    
                    if len(p) == 0:
                        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                        adv_stats_practical.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), 
                                        torch.Tensor(), torch.Tensor(), 
                                        person_labels[:, 0].cpu().tolist()))
                        continue
                    

                    shape_i = shapes[si][0]
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])
                    predn[:, 5] = 0
                    
                    tbox = xywh2xyxy(person_labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])
                    labelsn = torch.cat((person_labels[:, 0:1], tbox), 1)
                    
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                    correct = self._process_batch(predn, labelsn, iouv)
                    adv_stats_practical.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), person_labels[:, 0].cpu()))
                
                # 5. （ - ）
                for si, p in enumerate(repaired_nms_standard):
                    labels = targets[targets[:, 0] == si, 1:]
                    person_labels = labels[labels[:, 0] < 0.5].clone() if len(labels) > 0 else torch.zeros(0, 5, device=self.device)
                    
                    if len(person_labels) == 0:
                        if len(p) > 0:
                            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                            correct = torch.zeros(len(p), iouv.numel(), dtype=torch.bool, device=self.device)
                            repaired_stats_standard.append((correct.cpu(), p[:, 4].cpu(), p[:, 5].cpu(), []))
                        continue
                    
                    person_labels[:, 0] = 0
                    
                    if len(p) == 0:
                        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                        repaired_stats_standard.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), 
                                             torch.Tensor(), torch.Tensor(), 
                                             person_labels[:, 0].cpu().tolist()))
                        continue
                    

                    shape_i = shapes[si][0]
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])
                    predn[:, 5] = 0
                    
                    tbox = xywh2xyxy(person_labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])
                    labelsn = torch.cat((person_labels[:, 0:1], tbox), 1)
                    
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                    correct = self._process_batch(predn, labelsn, iouv)
                    repaired_stats_standard.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), person_labels[:, 0].cpu()))
                
                # 5.5. （ - ）
                for si, p in enumerate(repaired_nms_practical):
                    labels = targets[targets[:, 0] == si, 1:]
                    person_labels = labels[labels[:, 0] < 0.5].clone() if len(labels) > 0 else torch.zeros(0, 5, device=self.device)
                    
                    if len(person_labels) == 0:
                        if len(p) > 0:
                            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                            correct = torch.zeros(len(p), iouv.numel(), dtype=torch.bool, device=self.device)
                            repaired_stats_practical.append((correct.cpu(), p[:, 4].cpu(), p[:, 5].cpu(), []))
                        continue
                    
                    person_labels[:, 0] = 0
                    
                    if len(p) == 0:
                        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                        repaired_stats_practical.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), 
                                             torch.Tensor(), torch.Tensor(), 
                                             person_labels[:, 0].cpu().tolist()))
                        continue
                    

                    shape_i = shapes[si][0]
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])
                    predn[:, 5] = 0
                    
                    tbox = xywh2xyxy(person_labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])
                    labelsn = torch.cat((person_labels[:, 0:1], tbox), 1)
                    
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                    correct = self._process_batch(predn, labelsn, iouv)
                    repaired_stats_practical.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), person_labels[:, 0].cpu()))
        
        # mAP - 
        #  (conf=0.001, COCO)
        adv_map_standard = self._calculate_map_from_stats(adv_stats_standard) if adv_stats_standard else 0.0
        repaired_map_standard = self._calculate_map_from_stats(repaired_stats_standard) if repaired_stats_standard else 0.0
        
        #  (conf=0.25)
        adv_map_practical = self._calculate_map_from_stats(adv_stats_practical) if adv_stats_practical else 0.0
        repaired_map_practical = self._calculate_map_from_stats(repaired_stats_practical) if repaired_stats_practical else 0.0
        avg_repair_quality = 1.0 - (np.mean(repair_qualities) if repair_qualities else 0.0)
        
        #  - 
        repair_gain_standard = max(0.0, repaired_map_standard - adv_map_standard)
        repair_gain_practical = max(0.0, repaired_map_practical - adv_map_practical)
        
        #  - 
        robustness_score = (repaired_map_practical + repair_gain_practical) / 2.0
        
        # 🎯 （）
        adv_map = adv_map_practical
        repaired_map = repaired_map_practical
        repair_gain = repair_gain_practical
        
        # 🔍 （）
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🔍 （）")
        LOGGER.info(f"{'='*80}")
        

        LOGGER.info(f"\n📊 :")
        LOGGER.info(f"{'':<25} {'(0.001)':<18} {'(0.25)':<18} {''}")
        LOGGER.info(f"{'-'*80}")
        LOGGER.info(f"{'mAP':<25} {adv_map_standard:<18.4f} {adv_map_practical:<18.4f} {adv_map_practical-adv_map_standard:+.4f}")
        LOGGER.info(f"{'mAP':<25} {repaired_map_standard:<18.4f} {repaired_map_practical:<18.4f} {repaired_map_practical-repaired_map_standard:+.4f}")
        LOGGER.info(f"{'':<25} {repair_gain_standard:<18.4f} {repair_gain_practical:<18.4f} {repair_gain_practical-repair_gain_standard:+.4f}")
        LOGGER.info(f"{'-'*80}")
        LOGGER.info(f"💡 :")
        LOGGER.info(f"  • (0.001): COCO，")
        LOGGER.info(f"  • (0.25): ，")
        LOGGER.info(f"  • (0.25) ⭐")
        
        LOGGER.info(f"\n📊 :")
        LOGGER.info(f"  : {diagnostic_stats['total_images']}")
        LOGGER.info(f"  : {diagnostic_stats['images_with_persons']}")
        LOGGER.info(f"  : {diagnostic_stats['total_person_labels']}")
        LOGGER.info(f"")
        LOGGER.info(f"🔍 :")
        LOGGER.info(f"  : {diagnostic_stats['images_with_patches']}")
        LOGGER.info(f"  : {diagnostic_stats['total_patches_detected']}")
        
        patch_coverage = diagnostic_stats['images_with_patches'] / max(diagnostic_stats['total_images'], 1) * 100
        LOGGER.info(f"  : {patch_coverage:.1f}%")
        
        LOGGER.info(f"")
        LOGGER.info(f"🔧 :")
        LOGGER.info(f"  : {diagnostic_stats['images_repaired']}")
        LOGGER.info(f"  （）: {diagnostic_stats['images_clean']}")
        
        repair_ratio = diagnostic_stats['images_repaired'] / max(diagnostic_stats['total_images'], 1) * 100
        LOGGER.info(f"  : {repair_ratio:.1f}%")
        

        if abs(patch_coverage - repair_ratio) > 5.0:
            LOGGER.warning(f"\n⚠️ ：({patch_coverage:.1f}%)({repair_ratio:.1f}%)！")
            if repair_ratio > patch_coverage * 1.5:
                LOGGER.warning(f"   → （{0.05}）")
                LOGGER.warning(f"   → ：patch_masks")
            elif repair_ratio < patch_coverage * 0.8:
                LOGGER.warning(f"   → （）")
        
        if patch_coverage < 10:
            LOGGER.warning(f"\n⚠️ ：({patch_coverage:.1f}%)！")
        if adv_map > 0.5 and patch_coverage > 50:
            LOGGER.warning(f"\n⚠️ ：{patch_coverage:.1f}%，mAP({adv_map:.4f})")
            LOGGER.warning(f"   ：")
            LOGGER.warning(f"     1) ，")
            LOGGER.warning(f"     2) ，")
            LOGGER.warning(f"     3) ")
        if repair_ratio < patch_coverage:
            LOGGER.warning(f"\n⚠️ ：({repair_ratio:.1f}%)({patch_coverage:.1f}%)")
            LOGGER.warning(f"   （）")
        LOGGER.info(f"{'='*80}\n")
        
        return {
            # （0.25）
            'adv_performance': adv_map,
            'repaired_performance': repaired_map, 
            'average_repair_gain': repair_gain,
            'robustness_score': robustness_score,
            'repair_quality': avg_repair_quality,
            
            # （）
            'adv_performance_standard': adv_map_standard,
            'repaired_performance_standard': repaired_map_standard,
            'repair_gain_standard': repair_gain_standard,
            
            # （）
            'adv_performance_practical': adv_map_practical,
            'repaired_performance_practical': repaired_map_practical,
            'repair_gain_practical': repair_gain_practical,
            

            'diagnostic_stats': diagnostic_stats
        }
    
    def adaptive_curriculum_evaluation(self, epoch, total_epochs, eval_results):
        """"""
        
        progress = epoch / total_epochs
        

        if progress < 0.3:
            # ：
            weights = {
                'repair_gain': 0.6,
                'repaired_performance': 0.4,
                'robustness': 0.0
            }
            stage = "-"
            
        elif progress < 0.7:
            # ：
            weights = {
                'repair_gain': 0.4,
                'repaired_performance': 0.4,
                'robustness': 0.2
            }
            stage = "-"
            
        else:
            # ：
            weights = {
                'repair_gain': 0.3,
                'repaired_performance': 0.3, 
                'robustness': 0.4
            }
            stage = "-"
        

        comprehensive_score = (
            weights['repair_gain'] * eval_results['average_repair_gain'] +
            weights['repaired_performance'] * eval_results['repaired_performance'] +
            weights['robustness'] * eval_results.get('robustness_score', 0)
        )
        
        return {
            'comprehensive_score': comprehensive_score,
            'stage': stage,
            'weights': weights,
            'detailed_metrics': eval_results,
            'epoch': epoch
        }
    
    def _lightweight_validation(self, val_loader):
        """ - """
        self.repair_module.eval()
        self.person_detector.eval()
        self.stage1_patch_detector.eval()
        
        person_stats = []
        repair_qualities = []
        
        with torch.no_grad():
            for imgs, targets, paths, shapes in val_loader:
                imgs = imgs.to(self.device).float() / 255.0
                targets = targets.to(self.device)
                nb, _, height, width = imgs.shape
                
                # 1. 
                patch_pred = self.stage1_patch_detector(imgs)
                patch_masks = self._create_complete_patch_masks(patch_pred, imgs.shape)
                
                # 2. ✅ ：
                repaired_imgs = torch.zeros_like(imgs)
                for img_idx in range(nb):
                    img_mask = patch_masks[img_idx:img_idx+1]
                    if img_mask.sum() > 0:
                        # ，
                        # 🔥 （）
                        repair_result = self.repair_module(
                            imgs[img_idx:img_idx+1], img_mask
                        )
                        if isinstance(repair_result, tuple):
                            repaired_imgs[img_idx:img_idx+1] = repair_result[0]
                        else:
                            repaired_imgs[img_idx:img_idx+1] = repair_result
                    else:
                        # ，
                        repaired_imgs[img_idx:img_idx+1] = imgs[img_idx:img_idx+1]
                
                # 3. （SSIM）
                repair_quality = self._compute_ssim_loss(imgs, repaired_imgs)
                repair_qualities.append(repair_quality.item())
                
                # 4. 
                repaired_pred = self.person_detector(repaired_imgs)
                # 🔥 YOLOv11YOLOv5 NMS
                repaired_pred = self.convert_yolov11_to_yolov5_format(repaired_pred)
                
                # 🔧 ：targets
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
                
                # 5. NMS
                repaired_nms = non_max_suppression(
                    repaired_pred, conf_thres=0.001, iou_thres=0.6,
                    multi_label=True, max_det=300
                )
                
                # 6. 
                for si, p in enumerate(repaired_nms):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    
                    # 🔧 （class < 0.5，class=0）
                    person_labels = labels[labels[:, 0] < 0.5].clone() if nl else torch.zeros(0, 5, device=self.device)
                    
                    if len(person_labels) == 0:
                        if len(p) > 0:
                            # GT，FP
                            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                            correct = torch.zeros(len(p), iouv.numel(), dtype=torch.bool, device=self.device)
                            person_stats.append((correct.cpu(), p[:, 4].cpu(), p[:, 5].cpu(), []))
                        continue
                    
                    # 🔧 class=0（）
                    person_labels[:, 0] = 0
                    
                    if len(p) == 0:
                        # GT，FN
                        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                        person_stats.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), 
                                           torch.Tensor(), torch.Tensor(), 
                                           person_labels[:, 0].cpu().tolist()))
                        continue
                    
                    # 🔧 ：scale_boxes（1）
                    shape_i = shapes[si][0]  # shape
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])  # native-space pred
                    predn[:, 5] = 0  # class 0
                    

                    tbox = xywh2xyxy(person_labels[:, 1:5])  # ，
                    scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])  # resize -> 
                    labelsn = torch.cat((person_labels[:, 0:1], tbox), 1)
                    
                    # IoU
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
                    

                    correct = self._process_batch(predn, labelsn, iouv)
                    person_stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), person_labels[:, 0].cpu()))
        
        # mAP
        person_map = self._calculate_map_from_stats(person_stats) if person_stats else 0.0
        avg_repair_quality = 1.0 - (np.mean(repair_qualities) if repair_qualities else 0.0)
        
        return person_map, avg_repair_quality
    
    def _compute_ssim_loss(self, img1, img2):
        """"""
        try:
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu1 = F.avg_pool2d(img1, 3, 1, 1)
            mu2 = F.avg_pool2d(img2, 3, 1, 1)
            
            sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
            sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
            sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
            
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return 1 - ssim_map.mean()
        except:
            return torch.tensor(0.0, device=img1.device)
    
    def _calculate_map_from_stats(self, stats):
        """mAP@0.5"""
        if not stats:
            return 0.0
        
        try:
            stats_cpu = []
            for x in zip(*stats):
                x_cpu = [item.cpu().numpy() if torch.is_tensor(item) else item for item in x]
                stats_cpu.append(np.concatenate(x_cpu, 0))
            
            if len(stats_cpu) and stats_cpu[0].any():
                from utils.metrics import ap_per_class
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_cpu, plot=False, save_dir='.', names={0: 'person'})
                return float(ap[:, 0].mean())
        except Exception as e:
            LOGGER.warning(f"mAP: {e}")
        
        return 0.0
    
    # =========================================================================

    # =========================================================================
    
    def _filter_patch_targets(self, targets):
        """
        （1）
        
        🔧 ：single_cls=False，DataLoader
        targets: [img_idx_in_batch, class_id, x_center, y_center, width, height]
        """
        if targets.numel() == 0:
            return targets
        
        # 🔧 ：single_cls=False，class=0（）
        # ，
        return targets.clone()
        
        return patch_targets
    
    def _filter_person_targets(self, targets, batch_size=None):
        """
        
        
        targets: [img_idx_in_batch, class_id, x_center, y_center, width, height]
        - img_idx_in_batch: batch (0, 1, 2, ..., batch_size-1)
        - class_id: ID (0=person)
        """
        if targets.numel() == 0:
            return targets
        
        # person (class_id == 0)
        person_mask = targets[:, 1] == 0
        filtered_targets = targets[person_mask]
        
        if len(filtered_targets) == 0:
            return filtered_targets
        
        # 🔥 ：img_idx
        if batch_size is not None:
            # targets
            valid_mask = filtered_targets[:, 0] < batch_size
            filtered_targets = filtered_targets[valid_mask]
        
        return filtered_targets
    
    def _load_clean_images(self, paths, target_size=None):
        """
        🔥 （advpatch_strict）
        
        Args:
            paths: 
            target_size:  (H, W)， (640, 640)
            
        Returns:
            clean_imgs: tensor [B, 3, H, W]
        """
        import cv2
        
        if target_size is None:
            target_size = (640, 640)  # YOLOv5
        
        clean_imgs_list = []
        
        for path in paths:

            # /path/to/advpatch_strict/images/train/xxx.jpg 
            # -> /path/to/advpatch_strict/images/clean_train/xxx.jpg
            path = Path(path)
            
            if 'train' in path.parts:
                # ：train -> clean_train
                clean_path = str(path).replace('/train/', '/clean_train/')
            elif 'val' in path.parts:
                # ：val -> clean_val
                clean_path = str(path).replace('/val/', '/clean_val/')
            else:
                # ，
                clean_path = str(path)
            

            try:
                clean_img = cv2.imread(clean_path)
                if clean_img is None:
                    raise FileNotFoundError(f"Cannot load {clean_path}")
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                
                # 🔥 ：
                clean_img = cv2.resize(clean_img, (target_size[1], target_size[0]))  # (W, H)
                
                # tensor
                clean_img = torch.from_numpy(clean_img).permute(2, 0, 1)  # HWC -> CHW
                clean_imgs_list.append(clean_img)
            except Exception as e:
                # ，
                if len(clean_imgs_list) < 5:  # 5
                    LOGGER.warning(f"Failed to load clean image {clean_path}: {e}, using zeros")
                clean_imgs_list.append(torch.zeros(3, target_size[0], target_size[1]))
        
        # batch
        clean_imgs = torch.stack(clean_imgs_list, dim=0)
        return clean_imgs
    
    def _apply_dynamic_patch_visibility(self, imgs, patch_masks, visibility_ratio, fast_mode=True):
        """
        🔥 ：
        
        ，
        
        Args:
            imgs:  [B, 3, H, W]
            patch_masks:  [B, 1, H, W]
            visibility_ratio:  (0.0-1.0)
            fast_mode: （True）
        
        Returns:
            dynamic_masks:  [B, 1, H, W]
        """
        if visibility_ratio >= 0.99:
            return patch_masks  # ，
        
        if fast_mode:
            # 🚀 ：（10-15%）
            random_mask = torch.rand_like(patch_masks) < visibility_ratio
            dynamic_masks = patch_masks * random_mask.float()
            return dynamic_masks
        
        else:
            # ：
            batch_size = imgs.shape[0]
            dynamic_masks = torch.zeros_like(patch_masks)
            
            for i in range(batch_size):
                patch_area = patch_masks[i, 0] > 0.5
                
                if patch_area.sum() == 0:
                    continue
                

                patch_coords = torch.nonzero(patch_area, as_tuple=False)
                total_pixels = len(patch_coords)
                

                num_visible = int(total_pixels * visibility_ratio)
                
                if num_visible > 0:

                    visible_indices = torch.randperm(total_pixels, device=self.device)[:num_visible]
                    
                    for idx in visible_indices:
                        y, x = patch_coords[idx]
                        dynamic_masks[i, 0, y, x] = 1.0
            
            return dynamic_masks
    
    def _compute_enhanced_repair_loss(self, orig_imgs, repaired_imgs, patch_masks, clean_imgs=None, lightweight=True):
        """
        🔥 （ + ）
        
        Args:
            orig_imgs: 
            repaired_imgs: 
            patch_masks: 
            clean_imgs: （，）
            lightweight: 
        
        ，
        """
        
        # ============  ============
        if clean_imgs is not None:
            # mask3
            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
            
            non_patch_masks = 1 - patch_masks_3c
            
            # 🆕 （）
            with torch.no_grad():
                # 1.  (SSIM)
                patch_diff = torch.abs(repaired_imgs - clean_imgs) * patch_masks_3c
                patch_repair_error = patch_diff.mean()
                
                # 2. 
                non_patch_diff = torch.abs(repaired_imgs - orig_imgs) * non_patch_masks
                non_patch_preserve_error = non_patch_diff.mean()
                
                # 3. （，[0,1]）
                overall_quality = torch.abs(repaired_imgs - clean_imgs).mean()
                
                # 🔥 ：，
                # ：weight = base_weight * exp(quality * scale)
                quality_factor = torch.exp(overall_quality * 3.0).clamp(0.5, 3.0)  # [0.5, 3.0]
            
            # 🔧 1. ：
            # ：，
            base_patch_weight = getattr(self, 'adaptive_repair_weights', {}).get('patch_region', 2.0)
            patch_region_loss = F.l1_loss(
                repaired_imgs * patch_masks_3c,
                clean_imgs * patch_masks_3c
            ) * (base_patch_weight * quality_factor)  #  * 
            
            # 2. ：（）
            non_patch_loss = F.l1_loss(
                repaired_imgs * non_patch_masks,
                orig_imgs * non_patch_masks
            ) * 1.5
            
            # 3. ：
            # ：
            base_perceptual_weight = getattr(self, 'adaptive_repair_weights', {}).get('perceptual', 0.5)
            perceptual_loss = F.l1_loss(repaired_imgs, clean_imgs) * (base_perceptual_weight * quality_factor)
            
            # 🆕 4. （）
            # : 
            freq_consistency = torch.tensor(0.0, device=self.device)
            
            if not self.ablation_config.get('spatial_only', False):

                freq_loss_dict = self.freq_consistency_loss(
                    repaired_imgs, 
                    clean_imgs, 
                    patch_masks
                )
                
                #  - 
                base_freq_low_weight = getattr(self, 'adaptive_repair_weights', {}).get('freq_low', 0.5)
                

                if self.ablation_config.get('freq_low_only', False):

                    freq_consistency = freq_loss_dict['low_freq_loss'] * (base_freq_low_weight * quality_factor)
                elif self.ablation_config.get('freq_high_only', False):

                    freq_consistency = freq_loss_dict['high_freq_loss'] * 0.2
                else:
                    # (+)
                    freq_consistency = (
                        freq_loss_dict['low_freq_loss'] * (base_freq_low_weight * quality_factor) +
                        freq_loss_dict['high_freq_loss'] * 0.2
                    )
            
            # 🆕 5. （）

            before_repair_dist = torch.abs(orig_imgs - clean_imgs)
            after_repair_dist = torch.abs(repaired_imgs - clean_imgs)
            
            # （）
            repair_gain = (before_repair_dist - after_repair_dist) * patch_masks_3c
            
            # （）- 
            base_gain_penalty_weight = getattr(self, 'adaptive_repair_weights', {}).get('repair_gain_penalty', 3.0)
            negative_gain = torch.clamp(-repair_gain, min=0.0)
            repair_gain_loss = negative_gain.mean() * base_gain_penalty_weight
            
            # （）- 
            positive_gain = torch.clamp(repair_gain, min=0.0)
            repair_bonus = -positive_gain.mean() * 0.3  # （）
            

            total_loss = (
                patch_region_loss +      # （）
                non_patch_loss +
                perceptual_loss +        # （）
                freq_consistency +       # （）
                repair_gain_loss +
                repair_bonus
            )
            
            return total_loss

        # ============  ============
        # 🚀 ：
        if lightweight:
            # mask3
            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
            
            non_patch_masks = 1 - patch_masks_3c
            
            # 1. （）- 
            # 🔧 : 10.030.0 (3)
            consistency_loss = F.l1_loss(
                orig_imgs * non_patch_masks,
                repaired_imgs * non_patch_masks
            ) * 30.0
            
            # 2. 
            # 🔧 : 0.13.0 (30)
            if patch_masks.sum() > 0:
                # ：
                repaired_patch_region = repaired_imgs * patch_masks_3c
                smoothness_loss = F.mse_loss(
                    repaired_patch_region, 
                    torch.zeros_like(repaired_patch_region)
                ) * 3.0
            else:
                # ：
                # 🔧 : 5.015.0 (3)
                smoothness_loss = F.l1_loss(orig_imgs, repaired_imgs) * 15.0
            
            return consistency_loss + smoothness_loss
        
        # （）
        # mask3
        if patch_masks.shape[1] == 1:
            patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
        else:
            patch_masks_3c = patch_masks
        
        non_patch_masks = 1 - patch_masks_3c
        
        # 1. （）
        # 🔧 : 10.030.0 (3)
        consistency_loss = F.l1_loss(
            orig_imgs * non_patch_masks,
            repaired_imgs * non_patch_masks
        ) * 30.0
        
        # 2. （）
        # 🔧 : 0.13.0 (30)
        repaired_patch_region = repaired_imgs * patch_masks_3c
        tv_h = torch.abs(repaired_patch_region[:, :, 1:, :] - repaired_patch_region[:, :, :-1, :])
        tv_w = torch.abs(repaired_patch_region[:, :, :, 1:] - repaired_patch_region[:, :, :, :-1])
        tv_loss = (tv_h.mean() + tv_w.mean()) * 3.0
        
        # 3. 
        edge_loss = torch.tensor(0.0, device=self.device)
        try:
            patch_binary = (patch_masks > 0.5).float()
            edge_x = F.conv2d(patch_binary, torch.ones(1, 1, 1, 3, device=self.device), padding=(0, 1))
            edge_y = F.conv2d(patch_binary, torch.ones(1, 1, 3, 1, device=self.device), padding=(1, 0))
            edges = ((edge_x > 0) & (edge_x < 3)) | ((edge_y > 0) & (edge_y < 3))
            
            if edges.sum() > 0:
                edge_loss = F.mse_loss(
                    repaired_imgs * edges.repeat(1, 3, 1, 1),
                    orig_imgs * edges.repeat(1, 3, 1, 1)
                ) * 0.15
        except:
            pass
        
        # 4. 
        adversarial_loss = torch.tensor(0.0, device=self.device)
        try:
            with torch.no_grad():
                patch_pred_after = self.stage1_patch_detector(repaired_imgs)
            patch_pred_processed = non_max_suppression(
                patch_pred_after, conf_thres=0.05, iou_thres=0.6, max_det=10
            )
            num_detected = sum([len(p) for p in patch_pred_processed if p is not None])
            adversarial_loss = torch.tensor(num_detected * 0.01, device=self.device)
        except:
            pass
        
        total_repair_loss = (
            consistency_loss +
            tv_loss +
            edge_loss +
            adversarial_loss
        )
        
        return total_repair_loss
    
    def _create_complete_patch_masks(self, detections, img_shape, conf_thres=0.05):
        """（✅ 0.05，）"""
        batch_size, _, height, width = img_shape
        masks = torch.zeros((batch_size, 1, height, width), device=self.device)
        
        # NMS
        detections = non_max_suppression(
            detections, 
            conf_thres=conf_thres, 
            iou_thres=0.6,
            multi_label=True,
            max_det=300
        )
        
        if detections is None or len(detections) == 0:
            return masks
        
        try:
            for i in range(min(len(detections), batch_size)):
                det = detections[i]
                
                if det is None or not isinstance(det, torch.Tensor):
                    continue
                
                if det.numel() == 0:
                    continue
                
                # det2 [num_detections, 6]
                if det.dim() == 3:
                    det = det[0] if det.shape[0] > 0 else det.reshape(-1, det.shape[-1])
                elif det.dim() != 2:
                    continue
                

                if det.shape[-1] < 6:
                    continue
                
                # 1
                class_col = det[:, 5] if det.shape[1] > 5 else det[:, -1]
                conf_col = det[:, 4] if det.shape[1] > 4 else det[:, -2]
                
                class_mask = ((class_col == 0) | (class_col == 1))
                conf_mask = (conf_col > conf_thres)
                combined_mask = class_mask & conf_mask
                
                patch_detections = det[combined_mask]
                
                if patch_detections.numel() == 0:
                    continue
                

                for j in range(patch_detections.shape[0]):
                    detection = patch_detections[j]
                    x1, y1, x2, y2 = detection[:4].cpu().numpy().astype(int)
                    

                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))
                    
                    if x2 > x1 and y2 > y1:
                        masks[i, 0, y1:y2, x1:x2] = 1.0
        
        except Exception as e:
            LOGGER.warning(f": {e}")
        
        return masks
    
    def _validate_dataset(self, train_loader):
        """，"""
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"📊 1:")
        LOGGER.info(f"{'='*60}")
        
        total_images = 0
        total_patches = 0
        all_target_classes = set()
        
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            if i >= 10:
                break
            
            total_images += len(imgs)
            

            if len(targets) > 0:
                target_classes = targets[:, 1].cpu().numpy()  # class1
                all_target_classes.update(target_classes.tolist())
                total_patches += len(targets)
        
        LOGGER.info(f"  : {i+1}")
        LOGGER.info(f"  : {total_images}")
        LOGGER.info(f"  : {total_patches}")
        
        # 🔍 ：（ train.py ，）
        if all_target_classes:
            LOGGER.info(f"  ✅ : {sorted(all_target_classes)}")
            LOGGER.info(f"  ✅  ( patch_optimized.yaml )")
            LOGGER.info(f"  💡 : ， --single-cls ")
        
        if total_images > 0 and total_patches > 0:
            LOGGER.info(f"  : {total_patches/total_images:.1f}")
        elif total_patches == 0:
            LOGGER.error(f"  ❌ ：！")
            raise ValueError('DataLoader')
        else:
            LOGGER.warning(f"  ⚠️ ：！")
        
        LOGGER.info(f"{'='*60}")
        LOGGER.info("✅ 1 - \n")
    
    def _optimize_anchors_for_patches(self, train_loader, model):
        """anchor"""
        try:
            patch_wh = []
            img_size = self.opt.imgsz
            
            for i, (imgs, targets, paths, _) in enumerate(train_loader):
                if i >= 50:
                    break
                
                patch_targets = self._filter_patch_targets(targets)
                if len(patch_targets) == 0:
                    continue
                
                wh = patch_targets[:, 4:6]
                wh_pixel = wh * img_size
                patch_wh.append(wh_pixel)
            
            if not patch_wh:
                LOGGER.warning("  ⚠️ ，anchor")
                return
            
            patch_wh = torch.cat(patch_wh, 0).cpu().numpy()
            avg_w, avg_h = patch_wh.mean(0)
            
            detect_layer = model.model[-1]
            strides = detect_layer.stride.cpu().numpy()
            
            # anchor
            patch_anchors_pixel = [
                [[avg_w*0.6, avg_h*0.6], [avg_w*0.8, avg_h*1.0], [avg_w*1.0, avg_h*0.8]],
                [[avg_w*0.9, avg_h*0.9], [avg_w*1.1, avg_h*1.3], [avg_w*1.3, avg_h*1.1]],
                [[avg_w*1.2, avg_h*1.2], [avg_w*1.4, avg_h*1.6], [avg_w*1.6, avg_h*1.4]]
            ]
            
            new_anchors = []
            for i, stride in enumerate(strides):
                anchor_grid = torch.tensor(patch_anchors_pixel[i], dtype=torch.float32) / stride
                new_anchors.append(anchor_grid)
            
            new_anchors_tensor = torch.stack(new_anchors, dim=0).to(detect_layer.anchors.device)
            detect_layer.anchors.data = new_anchors_tensor
            
            detect_layer.anchor_grid = [torch.empty(0) for _ in range(detect_layer.nl)]
            detect_layer.grid = [torch.empty(0) for _ in range(detect_layer.nl)]
            
            LOGGER.info(f"  ✅ anchors")
            
        except Exception as e:
            LOGGER.warning(f"  ⚠️ Anchor: {e}, anchor")
    
    def _load_pretrained_weights_for_detector(self, detector):
        """"""
        try:
            possible_weights = [
                'yolov5s.pt',
                'weights/yolov5s.pt', 
                '../yolov5s.pt'
            ]
            
            pretrained_path = None
            for path in possible_weights:
                if os.path.exists(path):
                    pretrained_path = path
                    break
            
            if pretrained_path:
                LOGGER.info(f"🔄 : {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                
                if 'model' in checkpoint:
                    state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                model_state_dict = detector.state_dict()
                filtered_state_dict = {}
                
                for k, v in state_dict.items():
                    if k in model_state_dict and v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                
                detector.load_state_dict(filtered_state_dict, strict=False)
                LOGGER.info(f"✅  {len(filtered_state_dict)}/{len(model_state_dict)} ")
            else:
                LOGGER.warning(f"❌ ，")
                
        except Exception as e:
            LOGGER.warning(f"❌ : {e}")
    
    def _complete_validate_patch_detection(self, val_loader, model, verbose=True):
        """（，train.py）"""
        model.eval()

        
        stats = []
        seen = 0
        
        # 🔍 
        debug_stats = {
            'total_images': 0,
            'total_targets_all': 0,
            'total_targets_patch': 0,
            'total_predictions': 0,
            'images_with_patch': 0,
            'images_with_predictions': 0
        }
        

        val_iterator = enumerate(val_loader)
        if verbose:
            val_iterator = tqdm(val_iterator, total=len(val_loader), 
                              desc='1', 
                              bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in val_iterator:
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device)
                nb, _, height, width = imgs.shape
                

                pred = self.stage1_patch_detector(imgs)
                
                # 🔧 ：targets
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
                
                # NMS（🔧 ：train.py）
                pred = non_max_suppression(
                    pred, 
                    conf_thres=0.001,  # 🔧 train.py（COCO）
                    iou_thres=0.6,
                    labels=[], 
                    multi_label=True,
                    agnostic=False,
                    max_det=300  # 🔧 train.py
                )
                
                # 🔍 ：batch
                if batch_i == 0 and verbose:
                    for si, p in enumerate(pred):
                        if p is not None and len(p) > 0:
                            conf = p[:, 4]
                            LOGGER.info(f"\n🔍 batch{si}:")
                            LOGGER.info(f"  : {len(p)}")
                            LOGGER.info(f"   - : {conf.mean():.4f}, : {conf.median():.4f}")
                            LOGGER.info(f"   - : {conf.max():.4f}, : {conf.min():.4f}")
                            LOGGER.info(f"   >0.5: {(conf > 0.5).sum()}/{len(conf)}")
                            LOGGER.info(f"   >0.25: {(conf > 0.25).sum()}/{len(conf)}")
                            LOGGER.info(f"   >0.1: {(conf > 0.1).sum()}/{len(conf)}\n")
                            if si >= 2:  # 3
                                break
                

                for si, p in enumerate(pred):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    seen += 1
                    
                    # 🔍 
                    debug_stats['total_images'] += 1
                    debug_stats['total_targets_all'] += len(labels)
                    
                    # 🎯 1：
                    # 1. （class=1）
                    patch_label_mask = labels[:, 0] == 1
                    patch_labels = labels[patch_label_mask].clone()
                    
                    # 🔍 
                    debug_stats['total_targets_patch'] += len(patch_labels)
                    if len(patch_labels) > 0:
                        debug_stats['images_with_patch'] += 1
                    
                    # 2. class=1class=0（）
                    if len(patch_labels) > 0:
                        patch_labels[:, 0] = 0
                    
                    tcls = patch_labels[:, 0].cpu().tolist() if len(patch_labels) > 0 else []
                    
                    # 🔍 
                    if len(p) > 0:
                        debug_stats['total_predictions'] += len(p)
                        debug_stats['images_with_predictions'] += 1
                    
                    if len(p) == 0:
                        if len(patch_labels) > 0:
                            stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue
                    
                    # 🔧 ：scale_boxes（train.py）
                    shape_i = shapes[si][0]  # shape
                    predn = p.clone()
                    scale_boxes(imgs[si].shape[1:], predn[:, :4], shape_i, shapes[si][1])  # native-space pred
                    predn[:, 5] = 0  # 🔧 class 0
                    
                    # 3. class=0，IoU
                    if len(patch_labels) > 0:
                        # 🔧 ：patch_labels[:, 1:5]（resize）
                        # scale
                        tbox = xywh2xyxy(patch_labels[:, 1:5])  # xywh -> xyxy（resize）
                        scale_boxes(imgs[si].shape[1:], tbox, shape_i, shapes[si][1])  # resize -> 
                        
                        labelsn = torch.cat((patch_labels[:, 0:1], tbox), 1)
                        correct = self._process_batch(predn, labelsn, iouv=torch.linspace(0.5, 0.95, 10).to(self.device))
                        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))
                    else:
                        correct = torch.zeros(predn.shape[0], 10, dtype=torch.bool)
                        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), []))
        

        
        # mAP
        if stats:
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            
            if len(stats) and stats[0].any():
                from utils.metrics import ap_per_class
                names_dict = {0: 'patch'}
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=self.save_dir, names=names_dict)
                ap50, ap = ap[:, 0], ap.mean(1)
                map50 = ap50.mean()
                map50_95 = ap.mean()
                precision = p.mean()
                recall = r.mean()
                
                # 🔍 debug
                LOGGER.info(f"🔍 mAP:")
                LOGGER.info(f"  TP: {stats[0].sum():.0f}")
                LOGGER.info(f"  FP: {len(stats[0]) - stats[0].sum():.0f}")
                LOGGER.info(f"  Precision: {precision:.4f}")
                LOGGER.info(f"  Recall: {recall:.4f}")
                LOGGER.info(f"  mAP@0.5: {map50:.4f}")
                LOGGER.info(f"  mAP@0.5:0.95: {map50_95:.4f}\n")
                
                return map50, map50_95, precision, recall
        
        LOGGER.warning(f"⚠️ ：\n")
        return 0.0, 0.0, 0.0, 0.0
    
    def _process_batch(self, detections, labels, iouv):
        """
        mAP（YOLOv5）
        
        Arguments:
            detections (Tensor[N, 6]): x1, y1, x2, y2, conf, class
            labels (Tensor[M, 5]): class, x1, y1, x2, y2
            iouv (Tensor[10]): IoU
        Returns:
            correct (Tensor[N, 10]): 10IoU
        """
        device = iouv.device
        detections = detections.to(device)
        labels = labels.to(device)
        
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=device)
        
        if detections.shape[0] == 0 or labels.shape[0] == 0:
            return correct
        
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        
        # 🔧 ：IoU（YOLOv5）
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]  # IoU
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                
                correct[matches[:, 1].astype(int), i] = True
        
        return correct
    
    # =========================================================================
    # AFAL - train.py
    # =========================================================================
    
    def calculate_vulnerability(self, grad_adv, grad_clean):
        """ - train_Defense.py"""
        def frobenius_norm(x):
            x_flat = x.view(x.size(0), -1)
            return torch.norm(x_flat, p=2, dim=1)
        
        numerator = frobenius_norm(grad_adv)
        denominator = frobenius_norm(grad_clean) + 1e-8
        result = numerator / denominator

        if result.dim() > 0:
            result = result.mean()
        return result
    
    def adaptive_lambda(self, V, base_lambda=10.0, tau=0.8, k=10.0):
        """ - train_Defense.py"""
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=torch.float32)
        vulnerability_diff = V - tau
        scaled_diff = k * vulnerability_diff
        weight = torch.sigmoid(scaled_diff)

        result = base_lambda * weight
        if result.dim() > 0:
            result = result.mean()
        return result
    
    def complex_to_mag_phase(self, real, imag=None):
        """ - train_Defense.py"""
        if imag is None:
            mag = torch.abs(real)
            phase = torch.zeros_like(real)
            return mag, phase
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        return mag, phase

    def circular_distance(self, phase1, phase2):
        """ - train_Defense.py"""
        phase_diff = phase1 - phase2
        return 1 - torch.cos(phase_diff)

    def mag_phase_alignment(self, coeffs1, coeffs2, levels=3, base_amp=1.0, base_phase=0.5, lambda_weights=None):
        """- - train_Defense.py"""
        original_dtype = coeffs1[0].dtype
        ll1 = coeffs1[0].float()
        ll2 = coeffs2[0].float()
        highs1 = [h.float() for h in coeffs1[1]]
        highs2 = [h.float() for h in coeffs2[1]]
        align_loss = 0.0
        weight_idx = 0

        # 🔧 ：
        if ll1.shape != ll2.shape:
            LOGGER.warning(f": {ll1.shape} vs {ll2.shape}")
            min_h = min(ll1.shape[2], ll2.shape[2])
            min_w = min(ll1.shape[3], ll2.shape[3])
            ll1 = F.interpolate(ll1, size=(min_h, min_w), mode='bilinear', align_corners=False)
            ll2 = F.interpolate(ll2, size=(min_h, min_w), mode='bilinear', align_corners=False)


        if torch.isnan(ll1).any() or torch.isinf(ll1).any():
            LOGGER.warning("LL1NaN/Inf")
            return torch.tensor(0.0, dtype=original_dtype, device=ll1.device)
        if torch.isnan(ll2).any() or torch.isinf(ll2).any():
            LOGGER.warning("LL2NaN/Inf")
            return torch.tensor(0.0, dtype=original_dtype, device=ll2.device)

        ll_mag1, ll_phase1 = self.complex_to_mag_phase(ll1)
        ll_mag2, ll_phase2 = self.complex_to_mag_phase(ll2)
        lambda_w = lambda_weights[weight_idx] if lambda_weights and weight_idx < len(lambda_weights) else 1.0
        
        # lambda
        if torch.isnan(lambda_w) or torch.isinf(lambda_w):
            lambda_w = torch.tensor(1.0, device=ll1.device)
        
        amp_loss = F.mse_loss(ll_mag1, ll_mag2)
        phase_loss = self.circular_distance(ll_phase1, ll_phase2).mean()
        

        if torch.isnan(amp_loss) or torch.isinf(amp_loss):
            amp_loss = torch.tensor(0.0, device=ll1.device)
        if torch.isnan(phase_loss) or torch.isinf(phase_loss):
            phase_loss = torch.tensor(0.0, device=ll1.device)
            
        align_loss += lambda_w * (base_amp * amp_loss + base_phase * phase_loss)
        weight_idx += 1

        for level in range(min(levels, len(highs1), len(highs2))):
            hf1 = highs1[level]
            hf2 = highs2[level]
            
            # 🔧 ：
            if hf1.shape != hf2.shape:
                LOGGER.warning(f"{level}: {hf1.shape} vs {hf2.shape}")
                min_h = min(hf1.shape[3], hf2.shape[3])
                min_w = min(hf1.shape[4], hf2.shape[4])
                hf1 = F.interpolate(hf1, size=(min_h, min_w), mode='bilinear', align_corners=False)
                hf2 = F.interpolate(hf2, size=(min_h, min_w), mode='bilinear', align_corners=False)
            
            for i in range(min(3, hf1.shape[2], hf2.shape[2])):
                real1 = hf1[:, :, i, :, :]
                real2 = hf2[:, :, i, :, :]
                

                if torch.isnan(real1).any() or torch.isinf(real1).any():
                    continue
                if torch.isnan(real2).any() or torch.isinf(real2).any():
                    continue
                
                # 🔧 ：real1real2
                if real1.shape != real2.shape:
                    min_h = min(real1.shape[2], real2.shape[2])
                    min_w = min(real1.shape[3], real2.shape[3])
                    real1 = F.interpolate(real1, size=(min_h, min_w), mode='bilinear', align_corners=False)
                    real2 = F.interpolate(real2, size=(min_h, min_w), mode='bilinear', align_corners=False)
                    
                mag1, phase1 = self.complex_to_mag_phase(real1)
                mag2, phase2 = self.complex_to_mag_phase(real2)
                lambda_w = lambda_weights[weight_idx] if lambda_weights and weight_idx < len(lambda_weights) else 1.0
                
                # lambda
                if torch.isnan(lambda_w) or torch.isinf(lambda_w):
                    lambda_w = torch.tensor(1.0, device=real1.device)
                    
                amp_loss = F.mse_loss(mag1, mag2)
                phase_loss = self.circular_distance(phase1, phase2).mean()
                

                if torch.isnan(amp_loss) or torch.isinf(amp_loss):
                    amp_loss = torch.tensor(0.0, device=real1.device)
                if torch.isnan(phase_loss) or torch.isinf(phase_loss):
                    phase_loss = torch.tensor(0.0, device=real1.device)
                    
                align_loss += lambda_w * (base_amp * amp_loss + base_phase * phase_loss)
                weight_idx += 1
        
        align_loss /= (1 + 3 * levels)
        

        if torch.isnan(align_loss) or torch.isinf(align_loss):
            LOGGER.warning("NaN/Inf，0")
            align_loss = torch.tensor(0.0, dtype=original_dtype, device=align_loss.device)
            
        return align_loss.to(original_dtype)
    
    def _save_complete_stage1_model(self, epoch, patch_map, precision, recall, ema=None, mloss=None):
        """1"""
        save_dir = self.save_dir / 'stage1'
        save_dir.mkdir(exist_ok=True)
        
        best = save_dir / 'best.pt'
        last = save_dir / 'last.pt'
        
        model_for_save = ema.ema if ema else self.stage1_patch_detector
        if hasattr(model_for_save, 'module'):
            model_for_save = model_for_save.module
        
        ckpt = {
            'epoch': epoch,
            'best_fitness': patch_map,
            'model': deepcopy(model_for_save).half(),
            'ema': deepcopy(ema.ema).half() if ema else None,
            'updates': ema.updates if ema else 0,
            'opt': vars(self.opt),
            'date': datetime.now().isoformat(),
            'patch_map': patch_map,
            'precision': precision,
            'recall': recall,
            'training_history': self.training_history['stage1']
        }
        
        torch.save(ckpt, last)
        if patch_map > getattr(self, 'best_patch_map', 0.0):
            torch.save(ckpt, best)
            self.best_patch_map = patch_map
            LOGGER.info(f'✅ 1: {best} (patch_map: {float(patch_map):.4f})')
    
    def _save_stage1_results_csv(self, epoch, patch_map50, patch_map50_95, precision, recall, mloss):
        """1CSV"""
        save_dir = self.save_dir / 'stage1'
        save_dir.mkdir(exist_ok=True)
        csv_file = save_dir / 'results.csv'
        
        precision_val = precision[0] if isinstance(precision, np.ndarray) and len(precision) > 0 else precision
        recall_val = recall[0] if isinstance(recall, np.ndarray) and len(recall) > 0 else recall
        
        try:
            current_lr = self.stage1_patch_detector.optimizer.param_groups[0]['lr'] if hasattr(self.stage1_patch_detector, 'optimizer') else 0.0
            
            row_data = {
                'epoch': epoch,
                'train/box_loss': mloss[0].item() if len(mloss) > 0 else 0.0,
                'train/obj_loss': mloss[1].item() if len(mloss) > 1 else 0.0,
                'train/cls_loss': mloss[2].item() if len(mloss) > 2 else 0.0,
                'metrics/precision': precision_val,
                'metrics/recall': recall_val,
                'metrics/mAP_0.5': patch_map50,
                'metrics/mAP_0.5:0.95': patch_map50_95,
                'x/lr0': current_lr,
            }
            
            import csv
            file_exists = csv_file.exists()
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
                
        except Exception as e:
            LOGGER.warning(f"CSV: {e}")
    
    def _save_lightweight_joint_model(self, epoch, person_map, repair_quality, comprehensive_eval=None):
        """✅ ：（）"""
        save_dir = Path(self.save_dir) / 'stage2_lightweight' / 'weights'
        save_dir.mkdir(parents=True, exist_ok=True)
        

        detector = self.person_detector.module if hasattr(self.person_detector, 'module') else self.person_detector
        repair = self.repair_module.module if hasattr(self.repair_module, 'module') else self.repair_module
        
        ckpt = {
            'epoch': epoch,
            'person_detector': detector.state_dict(),
            'repair_module': repair.state_dict(),
            'person_map': person_map,
            'repair_quality': repair_quality,
            'repair_strength': repair.repair_strength.item(),
        }
        
        # ✅ （）
        if comprehensive_eval:
            ckpt['comprehensive_eval'] = comprehensive_eval
        
        torch.save(ckpt, save_dir / 'best_lightweight.pt')
        
        LOGGER.info(f"💾 : {save_dir / 'best_lightweight.pt'}")
        LOGGER.info(f"  ✅ :  + ")
        if comprehensive_eval:
            LOGGER.info(f"  📊 : {comprehensive_eval['comprehensive_score']:.4f}")
    
    def _save_lightweight_joint_csv(self, epoch, person_map, repair_quality, epoch_losses, batch_count):
        """CSV"""
        csv_file = Path(self.save_dir) / 'stage2_lightweight' / 'lightweight_results.csv'
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            row_data = {
                'epoch': epoch,
                'person_map': person_map,
                'repair_quality': repair_quality,
                'loss_total': epoch_losses['total'] / max(batch_count, 1),
                'loss_detection': epoch_losses['detection'] / max(batch_count, 1),
                'loss_repair': epoch_losses['repair'] / max(batch_count, 1),
            }
            
            import csv
            file_exists = csv_file.exists()
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
        except Exception as e:
            LOGGER.warning(f"CSV: {e}")
    
    def _save_comprehensive_csv(self, epoch, eval_results, curriculum_eval, epoch_losses, batch_count, feedback_result, strategy=None):
        """✅ CSV（ + ）"""
        csv_file = Path(self.save_dir) / 'stage2_lightweight' / 'comprehensive_results.csv'
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            row_data = {
                'epoch': epoch,
                'stage': curriculum_eval['stage'],
                'dynamic_stage': strategy['stage'] if strategy else 'N/A',
                'detector_trainable': strategy['detector_trainable'] if strategy else True,
                'detector_lr_scale': strategy.get('detector_lr_scale', 1.0) if strategy else 1.0,
                'patch_visibility': strategy['patch_visibility'] if strategy else 1.0,
                
                # （0.25）
                'adv_performance': eval_results['adv_performance'],
                'repaired_performance': eval_results['repaired_performance'],
                'repair_gain': eval_results['average_repair_gain'],
                
                # （）
                'adv_perf_std': eval_results['adv_performance_standard'],
                'repair_perf_std': eval_results['repaired_performance_standard'],
                'repair_gain_std': eval_results['repair_gain_standard'],
                

                'robustness_score': eval_results['robustness_score'],
                'repair_quality': eval_results['repair_quality'],
                'comprehensive_score': curriculum_eval['comprehensive_score'],
                'repair_strength': feedback_result['new_strength'],
                'performance_change': feedback_result['performance_change'],
                'adjustment_type': feedback_result['adjustment_type'],
                'loss_total': epoch_losses['total'] / max(batch_count, 1),
                'loss_detection': epoch_losses['detection'] / max(batch_count, 1),
                'loss_enhanced_repair': epoch_losses['enhanced_repair'] / max(batch_count, 1),
                'weight_repair_gain': curriculum_eval['weights']['repair_gain'],
                'weight_repaired_perf': curriculum_eval['weights']['repaired_performance'],
                'weight_robustness': curriculum_eval['weights']['robustness'],
            }
            
            import csv
            file_exists = csv_file.exists()
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
                
            LOGGER.info(f"📊 : {csv_file}")
        except Exception as e:
            LOGGER.warning(f"CSV: {e}")
    
    def save_complete_system(self):
        """"""
        complete_ckpt = {
            'stage1_patch_detector': self.stage1_patch_detector.state_dict(),
            'repair_module': self.repair_module.state_dict(),
            'person_detector': self.person_detector.state_dict(),
            'training_history': self.training_history,
            'complete_system_config': {
                'wavelet': 'db6',
                'levels': 3,
            },
            'opt': vars(self.opt)
        }
        torch.save(complete_ckpt, self.save_dir / 'complete_two_stage_system.pt')
        LOGGER.info(f"💾 ")


def parse_opt():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--data', type=str, default='data/patch_optimized.yaml', help='')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=416, help='train, val image size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    

    parser.add_argument('--stage1-epochs', type=int, default=100, help='1: ')
    parser.add_argument('--stage2-epochs', type=int, default=80, help='2: ')
    

    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate (SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--lrf', type=float, default=0.01, help='final OneCycleLR learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='optimizer weight decay 5e-4')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps')
    

    parser.add_argument('--wavelet-type', type=str, default='db1', 
                       choices=['db1', 'db2', 'db6', 'db8', 'haar', 'sym2', 'sym4', 'coif1'],
                        help='')
    parser.add_argument('--wavelet-levels', type=int, default=3, help='')
    

    parser.add_argument('--repair-strength', type=float, default=0.3, help=' (0.0-1.0)')
    

    parser.add_argument('--project', default='runs/complete_two_stage', help='')
    parser.add_argument('--name', default='exp', help='')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    

    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    

    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2], 
                       help=' (1=, 2=)')
    parser.add_argument('--stage1-weights', type=str, default='', 
                       help='1 (start-stage=2)')
    
    # ============  ============
    parser.add_argument('--detector-type', type=str, default='yolov5', 
                       choices=['yolov5', 'yolov11', 'yolov12'],
                       help=': yolov5, yolov11, yolov12')
    parser.add_argument('--detector-weights', type=str, default='', 
                       help=' (stage2)')
    parser.add_argument('--cross-detector-mode', action='store_true',
                       help=' (stage1yolov5, stage2)')
    
    # ============  ============
    parser.add_argument('--ablation-mode', action='store_true', 
                       help='')
    
    # ====== 1:  (AFAL) ======
    parser.add_argument('--disable-afal', action='store_true',
                       help=': AFAL (λ_base = 0)')
    parser.add_argument('--afal-amplitude-only', action='store_true',
                       help=':  (α_align > 0, β_align = 0)')
    parser.add_argument('--afal-phase-only', action='store_true',
                       help=':  (α_align = 0, β_align > 0)')
    
    # ====== 2:  ======
    parser.add_argument('--disable-detection-loss', action='store_true',
                       help=':  ()')
    parser.add_argument('--disable-repair-loss', action='store_true',
                       help=':  ()')
    parser.add_argument('--disable-dependency-loss', action='store_true',
                       help=':  (λ_dep = 0)')
    parser.add_argument('--disable-effectiveness-loss', action='store_true',
                       help=': ')
    
    # ============  ============
    #  β (Spectrum Enhancement Strength)
    parser.add_argument('--spectrum-beta', type=float, default=1.0,
                       help=' β ∈ {0, 0.5, 1.0, 1.5, 2.0} (: 1.0)')
    
    # AFAL λ_base (Alignment Weight)
    parser.add_argument('--afal-lambda-base', type=float, default=10.0,
                       help='AFAL λ_base (: 10.0)')
    
    # AFAL τ (Vulnerability Threshold)
    parser.add_argument('--afal-tau', type=float, default=0.8,
                       help='AFAL τ (: 0.8)')
    
    # AFAL Sigmoid κ (Sigmoid Slope)
    parser.add_argument('--afal-kappa', type=float, default=10.0,
                       help='AFALSigmoid κ (: 10.0)')
    
    #  s (Repair Strength)
    parser.add_argument('--repair-strength-init', type=float, default=0.3,
                       help=' s ∈ [0.1, 1.0] (: 0.3)')
    
    # AFAL α (Amplitude Alignment Weight)
    parser.add_argument('--afal-alpha', type=float, default=0.4,
                       help='AFAL α (: 0.4)')
    
    # ====== 2:  ======
    parser.add_argument('--freq-low-only', action='store_true',
                       help=': ')
    parser.add_argument('--freq-high-only', action='store_true',
                       help=': ')
    parser.add_argument('--spatial-only', action='store_true',
                       help=': ()')
    
    # ====== 2:  ======
    parser.add_argument('--disable-dynamic-strategy', action='store_true',
                       help=':  ()')
    parser.add_argument('--disable-feedback-adjustment', action='store_true',
                       help=': ')
    parser.add_argument('--disable-patch-visibility', action='store_true',
                       help=':  (100%%)')
    

    parser.add_argument('--ablation-tag', type=str, default='',
                       help=' ()')
    

    parser.add_argument('--ema', action='store_true', help='EMA')
    
    return parser.parse_args()


def main(opt):
    """"""

    if not HAS_WAVELETS:
        raise ImportError("pytorch_waveletspywt")
    

    device = select_device(opt.device, batch_size=opt.batch_size)
    

    init_seeds(42)
    

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    callbacks = Callbacks()
    

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"🔬  - ")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"📋 :")
    LOGGER.info(f"  : {opt.data}")
    LOGGER.info(f"  : {opt.cfg}")
    LOGGER.info(f"  : {device}")
    LOGGER.info(f"  : {opt.imgsz}px")
    LOGGER.info(f"  : {opt.batch_size}")
    
    LOGGER.info(f"\n🎯 :")
    LOGGER.info(f"  1 (): {opt.stage1_epochs} epochs")
    LOGGER.info(f"  2 (): {opt.stage2_epochs} epochs")
    
    LOGGER.info(f"\n🌊 :")
    LOGGER.info(f"  : {opt.wavelet_type}")
    LOGGER.info(f"  : {opt.wavelet_levels}")
    
    LOGGER.info(f"\n🔧 :")
    LOGGER.info(f"  : {opt.repair_strength}")
    
    LOGGER.info(f"\n💾 :")
    LOGGER.info(f"  : {opt.project}")
    LOGGER.info(f"  : {opt.name}")
    LOGGER.info(f"  : {opt.save_dir}")
    LOGGER.info(f"{'='*80}\n")
    

    with open(check_yaml(opt.data), encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    

    with open(check_yaml(opt.hyp), encoding='utf-8') as f:
        hyp = yaml.safe_load(f)
    
    #  single_cls 
    # 🔧 ：1（）
    # 1，
    #  train.py  --filter-class 1 
    
    # 🔥 ：single_cls，
    # 1：labels_patch（，class=1）
    # 2：labels_person_only（，class=0）
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"🔥 :")
    LOGGER.info(f"   1: labels_patch, class=1")
    LOGGER.info(f"   2: labels_person_only, class=0")
    LOGGER.info(f"   ✅ single_cls，nc=2（person=0, patch=1）")
    LOGGER.info(f"{'='*60}\n")
    
    # nc=2（：person=0, patch=1）
    nc = 2
    single_cls = False  # 🔥 single_cls
    names = {0: 'person', 1: 'patch'}
    
    #  nc  opt 
    opt.nc = nc
    opt.names = names
    

    training_system = CompleteTwoStageDefenseSystem(opt, device, callbacks)
    
    try:
        # 🔧 ========== ： ==========
        # 1：（class=1），（class=0）
        # class=1class=0（）
        filter_class = 1  # （）
        LOGGER.info(f'\n{"="*60}')
        LOGGER.info(f'🎯 :')
        LOGGER.info(f'  • nc (): {nc}')
        LOGGER.info(f'  • single_cls: {single_cls}')
        LOGGER.info(f'  • names: {names}')
        LOGGER.info(f'{"="*60}\n')
        

        if 'path' in data_dict:
            base_path = Path(data_dict['path'])
            if isinstance(data_dict.get('train'), list):
                train_path = [str(base_path / path) for path in data_dict['train']]
            else:
                train_path = str(base_path / data_dict['train']) if data_dict.get('train') else None
            
            val_data = data_dict.get('val', data_dict.get('test', data_dict['train']))
            if isinstance(val_data, list):
                val_path = [str(base_path / path) for path in val_data]
            else:
                val_path = str(base_path / val_data)
        else:
            train_path = data_dict['train']
            val_path = data_dict.get('val', data_dict.get('test', train_path))
        
        # 🔥 1：labels_patch（，class=1）
        base_path = Path(data_dict.get('path', ''))
        labels_patch_dir = base_path / 'labels_patch'
        
        LOGGER.info(f"📂 1：patch:")
        LOGGER.info(f"  : {train_path}")
        LOGGER.info(f"  : {val_path}")
        LOGGER.info(f"  📌 : {labels_patch_dir}")
        
        # 🔧 ：cache，labels_patch
        train_cache = labels_patch_dir / 'train.cache'
        val_cache = labels_patch_dir / 'val.cache'
        if train_cache.exists():
            train_cache.unlink()
            LOGGER.info(f"🗑️  cache: {train_cache}")
        if val_cache.exists():
            val_cache.unlink()
            LOGGER.info(f"🗑️  cache: {val_cache}")
        
        # data_dictlabels_patch
        original_path = data_dict.get('path', '')
        data_dict['path'] = str(base_path)
        
        # （ train.py ）
        # � 1：labels_patch（，class=1）
        # labels_patch
        base_dataset_path = Path(data_dict['path'])
        labels_patch_train = str(base_dataset_path / 'labels_patch' / 'train')
        labels_patch_val = str(base_dataset_path / 'labels_patch' / 'val')
        
        LOGGER.info(f"🔥 1:")
        LOGGER.info(f"  : {train_path}")
        LOGGER.info(f"  ✅ : {labels_patch_train}")
        LOGGER.info(f"  ✅ : {labels_patch_val}")
        LOGGER.info(f"  ✅ single_cls=False (class=1)")
        
        # �🚀 ：workerspin_memory
        train_loader, dataset = create_dataloader(
            train_path,  # 🔥 labels_patch
            opt.imgsz,
            opt.batch_size,
            stride=32,
            single_cls=False,  # 🔥 1single_cls，class=1
            hyp=hyp,
            cache=opt.cache,
            rect=opt.rect,
            rank=-1,
            workers=opt.workers,
            prefix=colorstr('train: '),
            shuffle=True
        )
        

        #  DataLoader（ train.py ）
        val_loader = create_dataloader(
            val_path,
            opt.imgsz,
            opt.batch_size,
            stride=32,
            single_cls=False,  # 🔥 1single_cls
            hyp=hyp,
            cache=opt.cache,
            rect=True,
            rank=-1,
            workers=opt.workers,
            prefix=colorstr('val: ')
        )[0]
        
        # 🔥 ：，labels_patch
        dataset.label_files = [lf.replace('/labels/', '/labels_patch/') for lf in dataset.label_files]
        val_loader.dataset.label_files = [lf.replace('/labels/', '/labels_patch/') for lf in val_loader.dataset.label_files]
        LOGGER.info(f"  ✅  labels  labels_patch")
        
        # 🔥 1：labels_patch（class=1）
        LOGGER.info(f"\n✅ 1:")
        LOGGER.info(f"  : {len(dataset)}")
        LOGGER.info(f"  : {len(val_loader.dataset)}")
        LOGGER.info(f"  ✅ labels_patch（class=1）")
        LOGGER.info(f"  ✅ : nc=2 (person=0, patch=1)\n")
        
        # 1: 
        if opt.start_stage <= 1:
            LOGGER.info(f"\n{'='*80}")
            LOGGER.info(f"🚀 1: （labels_patch）")
            LOGGER.info(f"{'='*80}")
            patch_map = training_system.stage1_train_complete_patch_detector(
                train_loader, val_loader, epochs=opt.stage1_epochs
            )
        else:
            LOGGER.info(f"\n⏭️ 1 ({opt.start_stage})")
            patch_map = 0.0
        
        # 🔥 2: advpatch_strict
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🔥 2: advpatch_strict")
        LOGGER.info(f"{'='*80}")
        
        # 🔥 ：advpatch_strict（）
        advpatch_dataset_path = Path('/path/to/project/dataset/adversarial_datasets_inria/advpatch_strict')
        train_path_stage2 = str(advpatch_dataset_path / 'images' / 'train')
        val_path_stage2 = str(advpatch_dataset_path / 'images' / 'val')
        labels_path_stage2 = str(advpatch_dataset_path / 'labels')
        
        LOGGER.info(f"🔥 2:")
        LOGGER.info(f"  ✅ : {train_path_stage2}")
        LOGGER.info(f"  ✅ : {val_path_stage2}")
        LOGGER.info(f"  ✅ : {train_path_stage2.replace('/train', '/clean_train')}")
        LOGGER.info(f"  ✅ : {val_path_stage2.replace('/val', '/clean_val')}")
        LOGGER.info(f"  ✅ : {labels_path_stage2}")
        LOGGER.info(f"  ✅ single_cls=False (class=0=person)")
        
        # 2dataloader（advpatch_strict）
        train_loader_stage2, dataset_stage2 = create_dataloader(
            train_path_stage2,  # advpatch_strict
            opt.imgsz,
            opt.batch_size,
            stride=32,
            single_cls=False,  # 🔥 2single_cls，class=0
            hyp=hyp,
            cache=opt.cache,
            rect=opt.rect,
            rank=-1,
            workers=opt.workers,
            prefix=colorstr('train: '),
            shuffle=True
        )
        
        val_loader_stage2 = create_dataloader(
            val_path_stage2,  # advpatch_strict
            opt.imgsz,
            opt.batch_size,
            stride=32,
            single_cls=False,  # 🔥 2single_cls
            hyp=hyp,
            cache=opt.cache,
            rect=True,
            rank=-1,
            workers=opt.workers,
            prefix=colorstr('val: ')
        )[0]
        
        # 🔥 ：，advpatch_strict/labels
        dataset_stage2.label_files = [
            str(Path(lf).parent.parent.parent / 'labels' / Path(lf).parent.name / Path(lf).name) 
            for lf in dataset_stage2.label_files
        ]
        val_loader_stage2.dataset.label_files = [
            str(Path(lf).parent.parent.parent / 'labels' / Path(lf).parent.name / Path(lf).name)
            for lf in val_loader_stage2.dataset.label_files
        ]
        LOGGER.info(f"  ✅  advpatch_strict/labels")
        
        LOGGER.info(f"\n✅ 2:")
        LOGGER.info(f"  : {len(dataset_stage2)}")
        LOGGER.info(f"  : {len(val_loader_stage2.dataset)}")
        LOGGER.info(f"  : {sum(len(l) for l in dataset_stage2.labels)}")
        LOGGER.info(f"  : {sum(len(l) for l in val_loader_stage2.dataset.labels)}")
        LOGGER.info(f"  ✅ +\n")
        
        # 2: 
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🚀 2: （+）")
        LOGGER.info(f"{'='*80}")
        
        person_map = training_system.lightweight_joint_training(
            train_loader_stage2, val_loader_stage2,  # 🔥 labels_person_onlydataloader
            epochs=opt.stage2_epochs,
            stage1_weights_path=opt.stage1_weights if opt.start_stage >= 2 else None
        )
        

        training_system.save_complete_system()
        

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"🎉 !")
        LOGGER.info(f"  ✅ 1 - : {patch_map:.4f}")
        LOGGER.info(f"  ✅ 2 - : {person_map:.4f}") 
        LOGGER.info(f"  💡 !")
        LOGGER.info(f"{'='*80}\n")
        
    except Exception as e:
        LOGGER.error(f": {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)