#!/usr/bin/env python3
"""


:
- YOLOv5 
- Faster R-CNN (cross_detector_full/RCNN)

:
- advPatch, DM-NAP, GNAP, LaVAN, T-SEA, diffpatch, adaptive_advpatch, advtexture
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse
import json
import time

# Defense
sys.path.insert(0, '/path/to/project')

from models.yolo import Model
from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class
from utils.torch_utils import select_device
from detector_adapters import FasterRCNNAdapter

# train1.py
try:
    from train1 import FrequencyGuidedRepairNet as Train1RepairNet
    TRAIN1_REPAIR_AVAILABLE = True
except ImportError:
    TRAIN1_REPAIR_AVAILABLE = False
    Train1RepairNet = None

# ====================  ====================
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELET_AVAILABLE = True
except ImportError:
    print("⚠️ pytorch_wavelets，")
    WAVELET_AVAILABLE = False
    
    class DWTForward(nn.Module):
        """(pytorch_wavelets)"""
        def __init__(self, J=4, wave='db6', mode='zero'):
            super().__init__()
            self.J = J
            self.wave = wave
            
        def forward(self, x):
            yl = F.avg_pool2d(x, 2)
            yh = [torch.randn(x.size(0), x.size(1), 3, 
                             x.size(2)//(2**(i+1)), x.size(3)//(2**(i+1)), 
                             device=x.device) for i in range(self.J)]
            return yl, yh
    
    class DWTInverse(nn.Module):
        """(pytorch_wavelets)"""
        def __init__(self, wave='db6', mode='zero'):
            super().__init__()
            self.wave = wave
            
        def forward(self, coeffs):
            yl, yh = coeffs
            return F.interpolate(yl, scale_factor=2**(len(yh)), mode='bilinear')

# ==================== exp373 ====================
class Exp373FrequencyRepairNet(nn.Module):
    """exp373+"""
    
    def __init__(self, wavelet='db6', levels=4, channels=3):
        super(Exp373FrequencyRepairNet, self).__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        self.channels = channels
        
        self.wavelet_transform = DWTForward(J=levels, wave=wavelet, mode='zero')
        self.wavelet_inverse = DWTInverse(wave=wavelet, mode='zero')
        
        #  ()
        # :  + 4(3) = channels + levels*3*channels = 3 + 4*3*3 = 3 + 36 = 39
        # checkpoint30,
        # : (3) + concat
        freq_input_channels = channels * (1 + levels * 3)  # 3 * (1 + 12) = 39,checkpoint30
        # checkpointfreq_encoder.0.weight: [64, 30, 3, 3],30
        freq_input_channels = 30
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(freq_input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        #  ()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        #  ()
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128*2 = 256 (+)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.repair_strength = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, patch_masks=None):
        """
        
        
        Args:
            x:  [B, C, H, W]
            patch_masks:  [B, 1, H, W] - 1
            
        Returns:
            repaired:  [B, C, H, W]
        """
        batch_size = x.size(0)
        
        # 1. 
        spatial_feat = self.spatial_encoder(x)  # [B, 128, H, W]
        
        # 2. 
        # 2.1 
        yl, yh = self.wavelet_transform(x)  # yl: [B, C, H/16, W/16], yh: list of [B, C, 3, H/2^i, W/2^i]
        
        # 2.2 
        yl_up = F.interpolate(yl, size=yh[0].shape[-2:], mode='bilinear', align_corners=False)
        
        # concat
        yh_concat = []
        target_size = yh[0].shape[-2:]
        for i, yh_level in enumerate(yh):
            # yh_level: [B, C, 3, H, W]
            B, C, dirs, H, W = yh_level.shape
            yh_flat = yh_level.view(B, C*dirs, H, W)  # [B, C*3, H, W]
            
            if (H, W) != target_size:
                yh_flat = F.interpolate(yh_flat, size=target_size, mode='bilinear', align_corners=False)
            yh_concat.append(yh_flat)
        
        yh_all = torch.cat(yh_concat, dim=1)  # [B, C*3*levels, H, W]
        
        freq_feat_input = torch.cat([yl_up, yh_all], dim=1)  # [B, C+C*3*levels, H, W]
        
        # 30 (checkpoint)
        if freq_feat_input.size(1) != 30:
            if not hasattr(self, 'freq_channel_adapter'):
                self.freq_channel_adapter = nn.Conv2d(
                    freq_feat_input.size(1), 30, 1, bias=False
                ).to(x.device)
                nn.init.constant_(self.freq_channel_adapter.weight, 
                                1.0 / freq_feat_input.size(1))
            freq_feat_input = self.freq_channel_adapter(freq_feat_input)
        
        # 2.3 
        freq_feat = self.freq_encoder(freq_feat_input)  # [B, 128, H, W]
        
        if freq_feat.shape[-2:] != spatial_feat.shape[-2:]:
            freq_feat = F.interpolate(freq_feat, size=spatial_feat.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        
        # 3. 
        fused_feat = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 256, H, W]
        repair_delta = self.fusion(fused_feat)  # [B, C, H, W]
        
        # 4. （）
        if patch_masks is not None and patch_masks.sum() > 0:
            # mask3
            if patch_masks.shape[1] == 1:
                patch_masks_3c = patch_masks.repeat(1, 3, 1, 1)
            else:
                patch_masks_3c = patch_masks
            
            # repair_delta
            if repair_delta.shape[-2:] != x.shape[-2:]:
                repair_delta = F.interpolate(repair_delta, size=x.shape[-2:], 
                                           mode='bilinear', align_corners=False)
            
            # mask
            if patch_masks_3c.shape[-2:] != x.shape[-2:]:
                patch_masks_3c = F.interpolate(patch_masks_3c, size=x.shape[-2:], 
                                              mode='nearest')
            
            # 🔥 : 
            alpha = torch.sigmoid(self.repair_strength)
            repaired = x + alpha * (repair_delta - 0.5) * 2.0 * patch_masks_3c
        else:
            # mask,
            repaired = x
            
        return torch.clamp(repaired, 0, 1)

# ====================  ====================
DATASET_CONFIGS = {
    'advPatch': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/advPatch',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'DM-NAP': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/DM-NAP',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'GNAP': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/GNAP',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'LaVAN': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/LaVAN',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'T-SEA': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/T-SEA',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'diffpatch': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/diffpatch',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'adaptive_advpatch': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/adaptive_advpatch',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    },
    'advtexture': {
        'root': '/path/to/project/dataset/adversarial_datasets_inria/advtexture',
        'adv_dir': 'images/val',
        'clean_dir': 'images/clean_val',
        'label_dir': 'labels/val'
    }
}

# ====================  ====================
DEFAULT_MODEL_PATH = "/path/to/project/runs/cross_detector_full/RCNN/complete_two_stage_system.pt"
DEFAULT_YOLO_CFG = "/path/to/project/models/yolov5s.yaml"
DEFAULT_OUTPUT_DIR = "/path/to/project/evaluation_multi_dataset"
DEFAULT_DEVICE = 'cuda:0'
IMG_SIZE = 640
CONF_THRES = 0.001  # COCO
IOU_THRES = 0.6

# ==================== YOLOv11/v12 ====================
def convert_yolov11_to_yolov5_format(pred):
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
        objectness = class_logits.max(dim=2, keepdim=True)[0].sigmoid()  # [B, N, 1]
        
        # 4. class logits
        class_probs = class_logits.sigmoid()  # [B, N, 80]
        
        # 5. YOLOv5
        pred_converted = torch.cat([bbox, objectness, class_probs], dim=2)  # [B, N, 85]
        
        return pred_converted
    
    return pred

# ====================  ====================
def load_complete_model(model_path, yolo_cfg, device, use_train1_repair=False):
    """
     - YOLOv5/v11/v12Faster R-CNN
    
    :
    1. checkpoint (stage1_patch_detector, repair_module, person_detector)
    2. YOLO (person_detector, stage1repair_module)
    
    Args:
        model_path: checkpoint
        yolo_cfg: YOLO  
        device: 
        use_train1_repair: train1.py（）
    """
    print(f"\n{'='*80}")
    print(f"🔍 ...")
    print(f"{'='*80}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"✓ Checkpoint")
    print(f"  : {list(checkpoint.keys())}")
    
    # YOLO
    is_complete_system = 'person_detector' in checkpoint or 'stage1_patch_detector' in checkpoint
    
    if is_complete_system:
        print(f"  : checkpoint")
        config = checkpoint.get('complete_system_config', {})
        wavelet = config.get('wavelet', 'db6')
        levels = config.get('levels', 4)
        channels = config.get('channels', 3)
        detector_type = config.get('detector_type', 'yolo')
        
        print(f"  : detector_type={detector_type}, wavelet={wavelet}, levels={levels}, channels={channels}")
        print(f"  ⚠️  : levelscheckpoint，")
        
        # 🔥 YOLOv11/v12
        is_yolov11 = False
        if 'person_detector' in checkpoint:
            person_detector_keys = list(checkpoint['person_detector'].keys())
            # Faster R-CNN
            if any('backbone.body' in k or 'rpn' in k or 'roi_heads' in k for k in person_detector_keys):
                actual_detector_type = 'faster_rcnn'
                print(f"  Faster R-CNN（checkpoint）")
            # YOLOv11/v12 (opt)
            elif (checkpoint.get('opt', {}).get('detector_type') in ['yolov11', 'yolov12'] or 
                  'model.10.m.0.attn.qkv.conv.weight' in person_detector_keys):
                actual_detector_type = checkpoint.get('opt', {}).get('detector_type', 'yolov11')
                is_yolov11 = True
                print(f"  {actual_detector_type} - ")
            else:
                actual_detector_type = detector_type
        else:
            actual_detector_type = detector_type
    else:
        print(f"  : YOLOv5")
        print(f"  ，person_detector")
        print(f"  ⚠️  stage1_patch_detector  repair_module ")
        
        wavelet = 'db6'
        levels = 4
        channels = 3
        detector_type = 'yolo'
        actual_detector_type = 'yolo'
        
        print(f"  : detector_type={detector_type}, wavelet={wavelet}, levels={levels}, channels={channels}")
    
    class TwoStageSystem(nn.Module):
        def __init__(self, yolo_cfg, device, wavelet, levels, channels, detector_type='yolo', use_train1_repair=False, is_yolov11=False):
            super().__init__()
            self.detector_type = detector_type
            self.use_train1_repair = use_train1_repair
            self.is_yolov11 = is_yolov11  # 🔥 YOLOv11/v12
            
            # 1. Stage1 (nc=1, ) - YOLO
            self.stage1_patch_detector = Model(yolo_cfg, ch=3, nc=1, anchors=None).to(device)
            
            # 2.  (train1.pyexp373)
            if use_train1_repair and TRAIN1_REPAIR_AVAILABLE:
                print(f"  train1.py（）")
                self.repair_module = Train1RepairNet(
                    wavelet=wavelet,
                    levels=levels,
                    enable_frequency=True,
                    repair_strength_init=0.3
                ).to(device)
            else:
                print(f"  exp373")
                self.repair_module = Exp373FrequencyRepairNet(
                    wavelet=wavelet, 
                    levels=levels, 
                    channels=channels
                ).to(device)
            
            # 3.  (YOLOFaster R-CNN)
            if detector_type == 'faster_rcnn':
                print(f"  Faster R-CNN...")
                self.person_detector_adapter = FasterRCNNAdapter(num_classes=80, device=device, backbone='resnet50')
                self.person_detector = self.person_detector_adapter.model
            elif is_yolov11:
                # YOLOv11/v12，
                print(f"  YOLOv11/v12 - ")
                self.person_detector = None  # load_state_dicts
                self.person_detector_adapter = None
            else:
                self.person_detector = Model(yolo_cfg, ch=3, nc=80, anchors=None).to(device)
                self.person_detector_adapter = None
            
        def load_state_dicts(self, ckpt, is_complete_system=True):
            """checkpointstate_dict
            
            Args:
                ckpt: checkpoint
                is_complete_system: checkpoint
            """
            if is_complete_system:
                # ：
                # YOLOv11，checkpointstate_dict，strict=False
                missing_s1, unexpected_s1 = self.stage1_patch_detector.load_state_dict(ckpt['stage1_patch_detector'], strict=False)
                if missing_s1:
                    print(f"  ⚠️  stage1_patch_detector: {len(missing_s1)} ")
                if unexpected_s1:
                    print(f"  ⚠️  stage1_patch_detector: {len(unexpected_s1)} ")
                
                #  - strict=Falselevels
                if not self.use_train1_repair:
                    missing_keys, unexpected_keys = self.repair_module.load_state_dict(ckpt['repair_module'], strict=False)
                    if missing_keys:
                        print(f"  ⚠️  repair_module: {len(missing_keys)} ")
                    if unexpected_keys:
                        print(f"  ⚠️  repair_module: {len(unexpected_keys)} ")
                else:
                    print(f"  ℹ️  train1，")
                
                # 🔥 YOLOv11/v12 - 
                if self.is_yolov11 or 'model.10.m.0.attn' in str(list(ckpt['person_detector'].keys())[:10]):
                    print(f"  YOLOv11/v12，...")
                    try:
                        # opt
                        detector_weights = ckpt['opt'].get('detector_weights', 'detection_models/yolo11s.pt')
                        print(f"     {detector_weights} ...")
                        
                        import torch
                        yolo_ckpt = torch.load(detector_weights, map_location='cpu')
                        
                        # model（）
                        if 'model' in yolo_ckpt and hasattr(yolo_ckpt['model'], 'yaml'):
                            base_model = yolo_ckpt['model'].float()  # 🔥 float32
                            print(f"    YOLOv11: {type(base_model).__name__}")
                            
                            missing, unexpected = base_model.load_state_dict(ckpt['person_detector'], strict=False)
                            
                            #  (person_detectorNonestage1)
                            target_device = next(self.stage1_patch_detector.parameters()).device
                            self.person_detector = base_model.to(target_device)
                            print(f"  ✓ person_detector (YOLOv11/v12, missing={len(missing)}, unexpected={len(unexpected)})")
                        else:
                            raise ValueError("model")
                    except Exception as e:
                        print(f"  ⚠️ : {e}")
                        print(f"  ！")
                        raise
                else:
                    missing_pd, unexpected_pd = self.person_detector.load_state_dict(ckpt['person_detector'], strict=False)
                    if missing_pd:
                        print(f"  ⚠️  person_detector: {len(missing_pd)} ")
                    if unexpected_pd:
                        print(f"  ⚠️  person_detector: {len(unexpected_pd)} ")
                    print(f"  ✓ person_detector ({self.detector_type})")
                
                print(f"✓ :")
                print(f"  ✓ stage1_patch_detector")
                if self.use_train1_repair:
                    print(f"  ✓ repair_module (train1)")
                else:
                    print(f"  ✓ repair_module (+)")
            else:
                # YOLO：person_detector
                # YOLOv5checkpoint'model''ema'
                if 'model' in ckpt:
                    state_dict = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
                elif 'ema' in ckpt and ckpt['ema']:
                    state_dict = ckpt['ema'].float().state_dict() if hasattr(ckpt['ema'], 'state_dict') else ckpt['ema']
                else:
                    state_dict = ckpt
                
                # person_detector
                self.person_detector.load_state_dict(state_dict, strict=False)
                
                print(f"✓ :")
                print(f"  ⚠️  stage1_patch_detector ()")
                if self.use_train1_repair:
                    print(f"  ⚠️  repair_module (train1，)")
                else:
                    print(f"  ⚠️  repair_module ()")
                print(f"  ✓ person_detector (YOLOv5)")
            
            # eval
            self.stage1_patch_detector.eval()
            self.repair_module.eval()
            self.person_detector.eval()
            
            return self
    
    training_system = TwoStageSystem(yolo_cfg, device, wavelet, levels, channels, actual_detector_type, use_train1_repair, is_yolov11 if is_complete_system else False)
    training_system.load_state_dicts(checkpoint, is_complete_system)
    
    print(f"✓ eval")
    if training_system.is_yolov11:
        print(f"  ℹ️  YOLOv11/v12： 84→85")
    
    return training_system

# ====================  ====================
def load_adversarial_dataset(dataset_name, dataset_config, img_size=640):
    """
    
    
    Args:
        dataset_name: 
        dataset_config: 
        img_size: 
    
    :
        list of dict: [{'adv_path', 'clean_path', 'label_path', 'labels'}, ...]
    """
    root = Path(dataset_config['root'])
    adv_dir = root / dataset_config['adv_dir']
    clean_dir = root / dataset_config['clean_dir']
    label_dir = root / dataset_config['label_dir']
    
    dataset = []
    
    if not adv_dir.exists():
        print(f"⚠️ : {adv_dir}")
        return dataset
    if not label_dir.exists():
        print(f"⚠️ : {label_dir}")
        return dataset
    
    adv_images = sorted(adv_dir.glob('*.jpg')) + sorted(adv_dir.glob('*.png'))
    
    print(f"\n{'='*80}")
    print(f"🔍  {dataset_name} ...")
    print(f"{'='*80}")
    print(f": {adv_dir}")
    print(f": {clean_dir}")
    print(f": {label_dir}")
    print(f": {len(adv_images)}")
    
    missing_clean = 0
    missing_label = 0
    
    for adv_path in tqdm(adv_images, desc=f"{dataset_name}"):
        clean_path = clean_dir / adv_path.name if clean_dir.exists() else None
        label_path = label_dir / (adv_path.stem + '.txt')
        
        if not label_path.exists():
            missing_label += 1
            continue
        
        # （）
        if clean_path and not clean_path.exists():
            missing_clean += 1
            clean_path = None
        
        #  (YOLO: class x_center y_center width height)
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    # person(class=0)
                    if int(cls) == 0:
                        labels.append([cls, x, y, w, h])
        
        if len(labels) > 0:
            dataset.append({
                'adv_path': str(adv_path),
                'clean_path': str(clean_path) if clean_path else None,
                'label_path': str(label_path),
                'labels': np.array(labels, dtype=np.float32)
            })
    
    print(f"✓ : {len(dataset)}")
    if missing_clean > 0:
        print(f"⚠️ : {missing_clean} ")
    if missing_label > 0:
        print(f"⚠️ : {missing_label} ")
    print(f"=" * 80)
    
    return dataset

# ====================  ====================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """"""
    shape = img.shape[:2]  #  [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    #  (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess_image(img_path, img_size=640):
    """"""
    img = cv2.imread(img_path)
    assert img is not None, f': {img_path}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img0 = img.copy()
    h0, w0 = img.shape[:2]
    
    # Letterbox
    img, ratio, pad = letterbox(img, img_size)
    h, w = img.shape[:2]
    
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img)
    
    return img, img0, (h0, w0), (h, w), ratio, pad

# ====================  ====================
def evaluate_dataset(model, dataset, device, output_dir, img_size=640, conf_thres=0.001, iou_thres=0.6):
    """
    LaVAN
    
    :
    1. 
    2.  ()
    3.  ()
    """
    print(f"\n{'='*80}")
    print(f"🚀 ()...")
    print(f"{'='*80}")
    
    all_adv_preds = []
    all_repaired_preds = []
    all_clean_preds = []
    all_labels = []
    
    stage1_times = []
    repair_times = []
    detection_times = []
    end_to_end_times = []
    
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc=""):
            # 1. 
            adv_img, adv_img0, (h0, w0), (h, w), ratio, pad = preprocess_image(sample['adv_path'], img_size)
            
            # tensor
            adv_tensor = torch.from_numpy(adv_img).unsqueeze(0).to(device)
            
            # （）
            if sample['clean_path'] is not None:
                clean_img, clean_img0, _, _, _, _ = preprocess_image(sample['clean_path'], img_size)
                clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).to(device)
            else:
                clean_tensor = None
            
            # 2.  ()
            labels = sample['labels'].copy()
            labels_pixel = labels.copy()
            labels_pixel[:, 1] = labels[:, 1] * w0  # x_center
            labels_pixel[:, 2] = labels[:, 2] * h0  # y_center
            labels_pixel[:, 3] = labels[:, 3] * w0  # width
            labels_pixel[:, 4] = labels[:, 4] * h0  # height
            
            # xyxy
            labels_xyxy = np.zeros_like(labels_pixel)
            labels_xyxy[:, 0] = labels_pixel[:, 0]  # class
            labels_xyxy[:, 1] = labels_pixel[:, 1] - labels_pixel[:, 3] / 2  # x1
            labels_xyxy[:, 2] = labels_pixel[:, 2] - labels_pixel[:, 4] / 2  # y1
            labels_xyxy[:, 3] = labels_pixel[:, 1] + labels_pixel[:, 3] / 2  # x2
            labels_xyxy[:, 4] = labels_pixel[:, 2] + labels_pixel[:, 4] / 2  # y2
            
            all_labels.append(labels_xyxy)
            
            # 3. 
            if model.detector_type == 'faster_rcnn':
                # Faster R-CNN
                adv_pred = model.person_detector([adv_tensor[0]])
                if len(adv_pred) > 0 and 'boxes' in adv_pred[0]:
                    boxes = adv_pred[0]['boxes']
                    scores = adv_pred[0]['scores']
                    labels = adv_pred[0]['labels']
                    
                    keep = scores > conf_thres
                    if keep.sum() > 0:
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]
                        
                        # boxes
                        boxes = scale_boxes((h, w), boxes, (h0, w0))
                        
                        # [x1,y1,x2,y2,conf,cls]
                        adv_det = torch.cat([
                            boxes,
                            scores.unsqueeze(1),
                            (labels - 1).float().unsqueeze(1)  # RCNN labels1
                        ], dim=1)
                        all_adv_preds.append(adv_det.cpu().numpy())
                    else:
                        all_adv_preds.append(np.zeros((0, 6)))
                else:
                    all_adv_preds.append(np.zeros((0, 6)))
            else:
                # YOLO
                adv_pred = model.person_detector(adv_tensor)
                if isinstance(adv_pred, tuple):
                    adv_pred = adv_pred[0]
                
                # 🔥 YOLOv11/v12，
                if model.is_yolov11:
                    adv_pred = convert_yolov11_to_yolov5_format(adv_pred)
                
                adv_nms = non_max_suppression(
                    adv_pred, 
                    conf_thres=conf_thres, 
                    iou_thres=iou_thres, 
                    multi_label=True,
                    max_det=300
                )
                
                if adv_nms[0] is not None and len(adv_nms[0]) > 0:
                    adv_det = adv_nms[0].clone()
                    adv_det[:, :4] = scale_boxes((h, w), adv_det[:, :4], (h0, w0))
                    all_adv_preds.append(adv_det.cpu().numpy())
                else:
                    all_adv_preds.append(np.zeros((0, 6)))
            
            # 4.  () - 

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            e2e_start = time.time()
            
            # 1: 
            stage1_start = time.time()
            patch_pred = model.stage1_patch_detector(adv_tensor)
            patch_masks = create_patch_masks(patch_pred, adv_tensor.shape, conf_thres=0.05)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            stage1_time = time.time() - stage1_start
            stage1_times.append(stage1_time)
            
            # 2:  (🔥 )
            repair_start = time.time()
            

            has_patch = patch_masks.sum() > 0
            
            if has_patch:
                # ，
                repaired_tensor = torch.zeros_like(adv_tensor)
                for img_idx in range(adv_tensor.shape[0]):
                    img_mask = patch_masks[img_idx:img_idx+1]
                    if img_mask.sum() > 0:
                        # ，
                        repair_result = model.repair_module(
                            adv_tensor[img_idx:img_idx+1],
                            img_mask
                        )
                        # （train1tuple，exp373tensor）
                        if isinstance(repair_result, tuple):
                            repaired_tensor[img_idx:img_idx+1] = repair_result[0]
                        else:
                            repaired_tensor[img_idx:img_idx+1] = repair_result
                    else:
                        # ，
                        repaired_tensor[img_idx:img_idx+1] = adv_tensor[img_idx:img_idx+1]
            else:
                # 🔥 ，，
                repaired_tensor = adv_tensor
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            repair_time = time.time() - repair_start
            repair_times.append(repair_time)
            
            # 3: 
            detection_start = time.time()
            if model.detector_type == 'faster_rcnn':
                repaired_pred = model.person_detector([repaired_tensor[0]])
                if len(repaired_pred) > 0 and 'boxes' in repaired_pred[0]:
                    boxes = repaired_pred[0]['boxes']
                    scores = repaired_pred[0]['scores']
                    labels = repaired_pred[0]['labels']
                    
                    keep = scores > conf_thres
                    if keep.sum() > 0:
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]
                        boxes = scale_boxes((h, w), boxes, (h0, w0))
                        repaired_det = torch.cat([
                            boxes,
                            scores.unsqueeze(1),
                            (labels - 1).float().unsqueeze(1)
                        ], dim=1)
                        all_repaired_preds.append(repaired_det.cpu().numpy())
                    else:
                        all_repaired_preds.append(np.zeros((0, 6)))
                else:
                    all_repaired_preds.append(np.zeros((0, 6)))
            else:
                repaired_pred = model.person_detector(repaired_tensor)
                if isinstance(repaired_pred, tuple):
                    repaired_pred = repaired_pred[0]
                
                # 🔥 YOLOv11/v12，
                if model.is_yolov11:
                    repaired_pred = convert_yolov11_to_yolov5_format(repaired_pred)
                
                repaired_nms = non_max_suppression(
                    repaired_pred, 
                    conf_thres=conf_thres, 
                    iou_thres=iou_thres, 
                    multi_label=True,
                    max_det=300
                )
                
                if repaired_nms[0] is not None and len(repaired_nms[0]) > 0:
                    repaired_det = repaired_nms[0].clone()
                    repaired_det[:, :4] = scale_boxes((h, w), repaired_det[:, :4], (h0, w0))
                    all_repaired_preds.append(repaired_det.cpu().numpy())
                else:
                    all_repaired_preds.append(np.zeros((0, 6)))
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            detection_time = time.time() - detection_start
            detection_times.append(detection_time)
            

            e2e_time = time.time() - e2e_start
            end_to_end_times.append(e2e_time)
            
            # 5.  () - 
            if sample['clean_path'] is not None and clean_tensor is not None:
                # （）
                clean_patch_pred = model.stage1_patch_detector(clean_tensor)
                clean_patch_masks = create_patch_masks(clean_patch_pred, clean_tensor.shape, conf_thres=0.05)
                

                has_clean_patch = clean_patch_masks.sum() > 0
                
                if has_clean_patch:
                    # （），
                    clean_processed_tensor = torch.zeros_like(clean_tensor)
                    for img_idx in range(clean_tensor.shape[0]):
                        img_mask = clean_patch_masks[img_idx:img_idx+1]
                        if img_mask.sum() > 0:
                            # ，
                            repair_result = model.repair_module(
                                clean_tensor[img_idx:img_idx+1],
                                img_mask
                            )
                            if isinstance(repair_result, tuple):
                                clean_processed_tensor[img_idx:img_idx+1] = repair_result[0]
                            else:
                                clean_processed_tensor[img_idx:img_idx+1] = repair_result
                        else:
                            # ，
                            clean_processed_tensor[img_idx:img_idx+1] = clean_tensor[img_idx:img_idx+1]
                else:
                    # 🔥 （），，
                    clean_processed_tensor = clean_tensor
                
                if model.detector_type == 'faster_rcnn':
                    clean_pred = model.person_detector([clean_processed_tensor[0]])
                    if len(clean_pred) > 0 and 'boxes' in clean_pred[0]:
                        boxes = clean_pred[0]['boxes']
                        scores = clean_pred[0]['scores']
                        labels = clean_pred[0]['labels']
                        
                        keep = scores > conf_thres
                        if keep.sum() > 0:
                            boxes = boxes[keep]
                            scores = scores[keep]
                            labels = labels[keep]
                            boxes = scale_boxes((h, w), boxes, (h0, w0))
                            clean_det = torch.cat([
                                boxes,
                                scores.unsqueeze(1),
                                (labels - 1).float().unsqueeze(1)
                            ], dim=1)
                            all_clean_preds.append(clean_det.cpu().numpy())
                        else:
                            all_clean_preds.append(np.zeros((0, 6)))
                    else:
                        all_clean_preds.append(np.zeros((0, 6)))
                else:
                    clean_pred = model.person_detector(clean_processed_tensor)
                    if isinstance(clean_pred, tuple):
                        clean_pred = clean_pred[0]
                    
                    # 🔥 YOLOv11/v12，
                    if model.is_yolov11:
                        clean_pred = convert_yolov11_to_yolov5_format(clean_pred)
                    
                    clean_nms = non_max_suppression(
                        clean_pred, 
                        conf_thres=conf_thres, 
                        iou_thres=iou_thres, 
                        multi_label=True,
                        max_det=300
                    )
                    
                    if clean_nms[0] is not None and len(clean_nms[0]) > 0:
                        clean_det = clean_nms[0].clone()
                        clean_det[:, :4] = scale_boxes((h, w), clean_det[:, :4], (h0, w0))
                        all_clean_preds.append(clean_det.cpu().numpy())
                    else:
                        all_clean_preds.append(np.zeros((0, 6)))
            else:
                # ，
                all_clean_preds.append(np.zeros((0, 6)))
    
    # 6. mAP
    print(f"\n{'='*80}")
    print(f"📊 mAP...")
    print(f"{'='*80}")
    
    results = {}
    
    # mAP
    adv_map = calculate_map(all_adv_preds, all_labels, output_dir, iou_thres=0.5)
    results['adversarial'] = adv_map
    print(f" mAP@0.5: {adv_map:.4f}")
    
    # mAP ()
    repaired_map = calculate_map(all_repaired_preds, all_labels, output_dir, iou_thres=0.5)
    results['repaired_frequency'] = repaired_map
    print(f"() mAP@0.5: {repaired_map:.4f}")
    
    # mAP
    clean_map = calculate_map(all_clean_preds, all_labels, output_dir, iou_thres=0.5)
    results['clean'] = clean_map
    print(f" mAP@0.5: {clean_map:.4f}")
    
    repair_gain = repaired_map - adv_map
    recovery_rate = (repaired_map - adv_map) / (clean_map - adv_map + 1e-6) * 100
    print(f"\n: {repair_gain:.4f} ({repair_gain*100:.2f}%)")
    print(f": {recovery_rate:.2f}%")
    
    # 7. 
    if len(end_to_end_times) > 0:
        stage1_ms = np.array(stage1_times) * 1000
        repair_ms = np.array(repair_times) * 1000
        detection_ms = np.array(detection_times) * 1000
        e2e_ms = np.array(end_to_end_times) * 1000
        
        print(f"\n⏱️  (ms):")
        print(f"{'':<20} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10}")
        print(f"{'-'*70}")
        
        components = [
            ("Stage1 ()", stage1_ms),
            ("Repair ()", repair_ms),
            ("Detection ()", detection_ms),
            ("End-to-End ()", e2e_ms)
        ]
        
        for name, times in components:
            print(f"{name:<20} {np.mean(times):<10.2f} {np.std(times):<10.2f} "
                  f"{np.min(times):<10.2f} {np.max(times):<10.2f} {np.median(times):<10.2f}")
        
        results['latency'] = {
            'stage1_ms': {
                'mean': float(np.mean(stage1_ms)),
                'std': float(np.std(stage1_ms)),
                'min': float(np.min(stage1_ms)),
                'max': float(np.max(stage1_ms)),
                'median': float(np.median(stage1_ms))
            },
            'repair_ms': {
                'mean': float(np.mean(repair_ms)),
                'std': float(np.std(repair_ms)),
                'min': float(np.min(repair_ms)),
                'max': float(np.max(repair_ms)),
                'median': float(np.median(repair_ms))
            },
            'detection_ms': {
                'mean': float(np.mean(detection_ms)),
                'std': float(np.std(detection_ms)),
                'min': float(np.min(detection_ms)),
                'max': float(np.max(detection_ms)),
                'median': float(np.median(detection_ms))
            },
            'end_to_end_ms': {
                'mean': float(np.mean(e2e_ms)),
                'std': float(np.std(e2e_ms)),
                'min': float(np.min(e2e_ms)),
                'max': float(np.max(e2e_ms)),
                'median': float(np.median(e2e_ms))
            }
        }
        
        print(f"\n💡 : {1000.0/np.mean(e2e_ms):.2f} FPS")
    
    print(f"=" * 80)
    
    return results

def create_patch_masks(patch_pred, img_shape, conf_thres=0.05):
    """"""
    _, _, h, w = img_shape
    
    if isinstance(patch_pred, tuple):
        patch_pred = patch_pred[0]
    
    patch_nms = non_max_suppression(
        patch_pred, 
        conf_thres=conf_thres, 
        iou_thres=0.6, 
        multi_label=True,
        max_det=300
    )
    
    masks = torch.zeros((1, 1, h, w), device=patch_pred.device)
    
    if patch_nms[0] is not None and len(patch_nms[0]) > 0:
        for det in patch_nms[0]:
            x1, y1, x2, y2 = det[:4].int()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            masks[0, 0, y1:y2, x1:x2] = 1.0
    
    return masks

def calculate_map(predictions, labels, output_dir, iou_thres=0.5):
    """mAP"""
    stats = []
    
    for pred, label in zip(predictions, labels):
        if len(pred) == 0:
            if len(label) > 0:
                pass
            continue
        
        pred_boxes = torch.from_numpy(pred[:, :4])
        pred_conf = torch.from_numpy(pred[:, 4])
        pred_cls = torch.from_numpy(pred[:, 5])
        
        if len(label) == 0:
            correct = torch.zeros((len(pred), 1), dtype=torch.bool)
            stats.append((correct, pred_conf, pred_cls, torch.zeros(0)))
            continue
        
        label_boxes = torch.from_numpy(label[:, 1:5])
        label_cls = torch.from_numpy(label[:, 0])
        
        iou_matrix = box_iou(pred_boxes, label_boxes)
        
        correct = torch.zeros((len(pred), 1), dtype=torch.bool)
        matched_gt = set()
        
        conf_order = torch.argsort(pred_conf, descending=True)
        
        for pred_idx in conf_order:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(label)):
                if gt_idx in matched_gt:
                    continue
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thres and best_gt_idx >= 0:
                if pred_cls[pred_idx] == label_cls[best_gt_idx]:
                    correct[pred_idx, 0] = True
                    matched_gt.add(best_gt_idx)
        
        stats.append((correct, pred_conf, pred_cls, label_cls))
    
    if len(stats) == 0:
        print("⚠️ ")
        return 0.0
    
    stats_np = [torch.cat(x, 0).cpu().numpy() if len(x) > 0 else np.array([]) for x in zip(*stats)]
    
    if len(stats_np[0]) == 0:
        print("⚠️ ")
        return 0.0
    
    try:
        result = ap_per_class(*stats_np, plot=False, save_dir=output_dir, names={0: 'person'})
        
        if len(result) == 7:
            tp, fp, p, r, f1, ap, unique_classes = result
        else:
            tp, fp, p, r, f1, ap = result
        
        if len(ap) > 0:
            if ap.ndim == 2:
                ap50 = ap[:, 0]
            else:
                ap50 = ap
            return ap50.mean()
        else:
            return 0.0
    except Exception as e:
        print(f"⚠️ AP: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def box_iou(boxes1, boxes2):
    """IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

# ====================  ====================
def parse_args():
    """"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help='checkpoint')
    parser.add_argument('--yolo-cfg', type=str, default=DEFAULT_YOLO_CFG,
                       help='YOLO')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                       help='')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='')
    parser.add_argument('--datasets', nargs='+', 
                       default=list(DATASET_CONFIGS.keys()),
                       choices=list(DATASET_CONFIGS.keys()),
                       help='')
    parser.add_argument('--conf-thres', type=float, default=CONF_THRES,
                       help='')
    parser.add_argument('--iou-thres', type=float, default=IOU_THRES,
                       help='IOU')
    parser.add_argument('--use-train1-repair', action='store_true',
                       help='train1.py')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 80)
    print("🎯 ")
    print("=" * 80)
    print(f": {args.model}")
    print(f"YOLO: {args.yolo_cfg}")
    print(f": {args.output_dir}")
    print(f": {args.device}")
    print(f": {', '.join(args.datasets)}")
    if args.use_train1_repair:
        print(f": train1.py ()")
    else:
        print(f": exp373 ()")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = select_device(args.device)
    model = load_complete_model(args.model, args.yolo_cfg, device, args.use_train1_repair)
    
    all_results = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"⚠️ : {dataset_name}")
            continue
        
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        dataset = load_adversarial_dataset(dataset_name, dataset_config, IMG_SIZE)
        
        if len(dataset) == 0:
            print(f"⚠️ {dataset_name} ，")
            continue
        
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        try:
            results = evaluate_dataset(
                model=model,
                dataset=dataset,
                device=device,
                output_dir=dataset_output_dir,
                img_size=IMG_SIZE,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres
            )
            all_results[dataset_name] = results
            
            result_file = os.path.join(dataset_output_dir, 'evaluation_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✓ {dataset_name} : {result_file}")
            
        except Exception as e:
            print(f"❌ {dataset_name} : {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}
    
    summary_file = os.path.join(args.output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"📊 ")
    print(f"{'='*80}")
    print(f"{'':<20} {'mAP':<12} {'mAP':<12} {'mAP':<12} {'':<12}")
    print(f"{'-'*80}")
    
    valid_datasets = []
    sum_adv_map = 0.0
    sum_rep_map = 0.0
    sum_clean_map = 0.0
    sum_gain = 0.0
    
    for dataset_name, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_name:<20} {'ERROR'}")
            continue
        
        adv_map = results.get('adversarial', 0.0)
        rep_map = results.get('repaired_frequency', 0.0)
        clean_map = results.get('clean', 0.0)
        gain = rep_map - adv_map
        
        print(f"{dataset_name:<20} {adv_map:<12.4f} {rep_map:<12.4f} {clean_map:<12.4f} {gain:<12.4f}")
        
        valid_datasets.append(dataset_name)
        sum_adv_map += adv_map
        sum_rep_map += rep_map
        sum_clean_map += clean_map
        sum_gain += gain
    
    if len(valid_datasets) > 0:
        print(f"{'-'*80}")
        avg_adv_map = sum_adv_map / len(valid_datasets)
        avg_rep_map = sum_rep_map / len(valid_datasets)
        avg_clean_map = sum_clean_map / len(valid_datasets)
        avg_gain = sum_gain / len(valid_datasets)
        
        print(f"{' (Average)':<20} {avg_adv_map:<12.4f} {avg_rep_map:<12.4f} {avg_clean_map:<12.4f} {avg_gain:<12.4f}")
        
        if avg_clean_map - avg_adv_map > 1e-6:
            recovery_rate = (avg_rep_map - avg_adv_map) / (avg_clean_map - avg_adv_map) * 100
            print(f"\n (Recovery Rate): {recovery_rate:.2f}%")
        
        all_results['_average'] = {
            'adversarial': avg_adv_map,
            'repaired_frequency': avg_rep_map,
            'clean': avg_clean_map,
            'gain': avg_gain,
            'num_datasets': len(valid_datasets),
            'datasets': valid_datasets
        }
        
        # JSON
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=4)
    
    print(f"{'='*80}")
    print(f"\n✓ : {summary_file}")
    print(f"\n🎉 !")
    print(f"{'='*80}")
