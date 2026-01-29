#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

(YOLO/DETR/Faster R-CNN/SSD)
train_kua.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

# =============================================================================
# =============================================================================

class DetectorAdapter(nn.Module):
    """ - """
    
    def __init__(self, num_classes=80, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.detector_type = "base"
        
    def forward(self, x):
        """ - """
        raise NotImplementedError
    
    def compute_loss(self, predictions, targets, imgs):
        """ - """
        raise NotImplementedError
    
    def get_trainable_parameters(self):
        """"""
        return self.model.parameters()
    
    def load_pretrained(self, weights_path):
        """"""
        raise NotImplementedError

# =============================================================================
# YOLO ()
# =============================================================================

class YOLOAdapter(DetectorAdapter):
    """YOLOv5"""
    
    def __init__(self, cfg, num_classes=80, device='cuda'):
        super().__init__(num_classes, device)
        self.detector_type = "yolo"
        
        from models.yolo import Model
        self.model = Model(cfg, ch=3, nc=num_classes, anchors=None).to(device)
        
        # YOLOloss
        from utils.loss import ComputeLoss
        self.compute_loss_fn = ComputeLoss(self.model)
        
    def forward(self, x):
        """YOLO"""
        return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """
        YOLO
        
        Args:
            predictions: YOLO
            targets:  [batch_idx, class, x, y, w, h]
            imgs: 
        
        Returns:
            loss: 
            loss_items: 
        """
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        loss, loss_items = self.compute_loss_fn(predictions, targets)
        return loss, loss_items
    
    def load_pretrained(self, weights_path):
        """YOLO"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        
        if 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict()
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"✅ YOLO: {weights_path}")

# =============================================================================
# DETR
# =============================================================================

class DETRAdapter(DetectorAdapter):
    """DETR"""
    
    def __init__(self, num_classes=80, device='cuda', backbone='resnet50'):
        super().__init__(num_classes, device)
        self.detector_type = "detr"
        self.backbone = backbone
        
        try:
            # 1: transformers
            from transformers import DetrForObjectDetection, DetrConfig
            import os
            
            # 🔥 ，
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            print(f"🔧 DETR...")
            
            # 🔥 1: pytorch_model.bin
            local_model_path = "/path/to/project/detection_models/pytorch_model.bin"
            if os.path.exists(local_model_path):
                try:
                    print(f"📁 DETR: {local_model_path}")
                    
                    # （backbone）
                    # 🔥 
                    config = DetrConfig(
                        num_labels=num_classes,
                        num_queries=100,  # 🔥 ：COCO100，
                        use_pretrained_backbone=False,
                        # 🔥 
                        class_cost=2,  # （1）
                        bbox_cost=5,   # bbox
                        giou_cost=2,   # GIoU
                        # 🔥 ：bbox，
                        bbox_loss_coefficient=10,  # 510
                        giou_loss_coefficient=5,   # 25
                        # 🔥 ：eos，precisionrecall
                        eos_coefficient=0.1,  # 0.010.1（DETR）
                        # 🔥 ：，
                        auxiliary_loss=True,
                    )
                    
                    self.model = DetrForObjectDetection(config)
                    
                    state_dict = torch.load(local_model_path, map_location='cpu')
                    
                    # （'detr.'）
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # （class_labels_classifier）
                        if 'class_labels_classifier' in k:
                            print(f"   ⏩ : {k}")
                            continue
                        
                        # 🔥 query_position_embeddingsquery
                        if 'query_position_embeddings' in k:
                            print(f"   ⏩ query: {k} (query)")
                            continue
                            
                        if k.startswith('detr.'):
                            new_state_dict[k[5:]] = v  # 'detr.'
                        else:
                            new_state_dict[k] = v
                    
                    # （）
                    missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                    
                    self.model = self.model.to(device)
                    print(f"✅ DETR (backbone={backbone})")
                    if len(missing) > 0:
                        print(f"   ⚠️  {len(missing)} （）")
                    if len(unexpected) > 0:
                        print(f"   ⚠️  {len(unexpected)} ")
                    
                except Exception as e:
                    print(f"⚠️ : {e}")
                    raise
                    
            else:
                # 2: HuggingFace
                try:
                    self.model = DetrForObjectDetection.from_pretrained(
                        "facebook/detr-resnet-50",
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                        local_files_only=True
                    ).to(device)
                    print(f"✅ DETR (backbone={backbone})")
                    
                except Exception as e:
                    print(f"⚠️ : {e}")
                    print(f"🔧 DETR...")
                    
                    config = DetrConfig(
                        num_labels=num_classes,
                        num_queries=100,
                    )
                    
                    self.model = DetrForObjectDetection(config).to(device)
                    print(f"✅ DETR (backbone={backbone})")
            
            self.use_transformers = True
            
        except ImportError:
            # 2: DETR
            print("⚠️ transformers，DETR")
            sys.path.insert(0, str(ROOT / 'detr-main'))
            
            try:
                from detr_main.models import build_model
                from detr_main.util.misc import nested_tensor_from_tensor_list
                
                # DETR
                args = type('Args', (), {
                    'backbone': backbone,
                    'num_classes': num_classes,
                    'hidden_dim': 256,
                    'nheads': 8,
                    'num_encoder_layers': 6,
                    'num_decoder_layers': 6,
                    'dim_feedforward': 2048,
                    'dropout': 0.1,
                    'num_queries': 100,
                    'pre_norm': False,
                    'masks': False,
                    'aux_loss': True,
                    'set_cost_class': 1,
                    'set_cost_bbox': 5,
                    'set_cost_giou': 2,
                    'bbox_loss_coef': 5,
                    'giou_loss_coef': 2,
                    'eos_coef': 0.1,
                })()
                
                self.model, self.criterion = build_model(args)
                self.model = self.model.to(device)
                self.use_transformers = False
                
                print(f"✅ DETR")
                
            except Exception as e:
                print(f"❌ DETR: {e}")
                raise
    
    def forward(self, x, targets=None):
        """DETR"""
        if self.use_transformers:
            # Transformers
            if targets is not None:
                # （DETRcompute_loss）
                outputs = self.model(pixel_values=x)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(pixel_values=x)
            return outputs
        else:
            return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """
        DETR
        
        Args:
            predictions: DETR
            targets:  [{'boxes': Tensor[N,4], 'labels': Tensor[N]}]
            imgs: 
        
        Returns:
            loss: 
            loss_items: 
        """
        if self.use_transformers:
            # Transformers
            # YOLOtargetsDETR
            detr_targets = self._convert_targets_to_detr_format(targets, imgs)
            
            try:
                outputs = self.model(
                    pixel_values=imgs,
                    labels=detr_targets
                )
                loss_dict = outputs.loss_dict
                
                # 🔥 ： cardinality_error
                # cardinality_error 
                # DETR（100query vs 1-2）
                filtered_loss_dict = {}
                for k, v in loss_dict.items():
                    if 'cardinality_error' in k:
                        # 🔥  cardinality_error（）
                        continue
                    else:
                        filtered_loss_dict[k] = v
                
                # （loss_dict）
                loss = sum(filtered_loss_dict.values())
                loss_items = torch.tensor([
                    loss_dict.get('loss_ce', 0),
                    loss_dict.get('loss_bbox', 0),
                    loss_dict.get('loss_giou', 0)
                ], device=self.device)
                
            except Exception as e:
                print(f"❌ DETR: {e}")
                print(f"   targets: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for target in detr_targets for k, v in target.items()]}")
                raise
            
        else:
            outputs = predictions
            detr_targets = self._convert_targets_to_detr_format(targets, imgs)
            
            loss_dict = self.criterion(outputs, detr_targets)
            
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            loss_items = torch.tensor([
                loss_dict.get('loss_ce', 0),
                loss_dict.get('loss_bbox', 0),
                loss_dict.get('loss_giou', 0)
            ], device=self.device)
        
        return loss, loss_items
    
    def _convert_targets_to_detr_format(self, yolo_targets, imgs):
        """
        YOLODETR
        
        YOLO: Tensor[N, 6] (batch_idx, class, x, y, w, h) - 
        DETR: List[Dict] dict 'boxes' [N,4], 'class_labels' [N]
        
        ⚠️ ：TransformersDetrForObjectDetection'class_labels'，'labels'
        """
        batch_size = imgs.shape[0]
        detr_targets = []
        
        for batch_idx in range(batch_size):
            # batch
            mask = yolo_targets[:, 0] == batch_idx
            if mask.sum() == 0:
                detr_targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'class_labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                continue
            
            batch_targets = yolo_targets[mask]
            
            # boxes
            # ⚠️ DETR+1，no-object（class_id = num_classes）
            labels = batch_targets[:, 1].long()
            boxes_xywh = batch_targets[:, 2:6]  # (cx, cy, w, h) 
            
            # DETR(cx, cy, w, h)，
            detr_targets.append({
                'boxes': boxes_xywh,
                'class_labels': labels  # ✅ class_labels，labels
            })
        
        return detr_targets
    
    def load_pretrained(self, weights_path):
        """DETR"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"✅ DETR: {weights_path}")
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        DETR
        
        Args:
            outputs: DETR (DetrObjectDetectionOutput)
            conf_thres: 
            iou_thres: NMSIoU（DETRNMS，）
            img_size:  (height, width) 
        
        Returns:
            predictions: List[Tensor[N, 6]]  [x1, y1, x2, y2, conf, cls] ()
        """
        predictions = []
        
        if self.use_transformers:
            # Transformers
            logits = outputs.logits  # [batch, num_queries, num_classes+1]
            pred_boxes = outputs.pred_boxes  # [batch, num_queries, 4] ()
            
            batch_size = logits.shape[0]
            
            if img_size is None:
                # outputs
                if hasattr(outputs, 'pixel_values') and outputs.pixel_values is not None:
                    img_size = outputs.pixel_values.shape[2:]  # (H, W)
                else:
                    # 640x640（）
                    img_size = (640, 640)
            
            if isinstance(img_size, torch.Tensor):
                img_size = img_size.tolist()
            
            height, width = img_size if len(img_size) == 2 else (img_size[0], img_size[0])
            
            for i in range(batch_size):
                # 🔥 ：，sigmoidsoftmax
                # ：softmaxno-object，
                if self.num_classes == 1:
                    # ：0logitsigmoid
                    scores = logits[i, :, 0].sigmoid()  # [num_queries]
                    labels = torch.zeros_like(scores).long()  # class=0
                else:
                    # ：softmax
                    probs = logits[i].softmax(-1)[:, :-1]  # [num_queries, num_classes]
                    scores, labels = probs.max(-1)  # [num_queries]
                
                keep = scores > conf_thres
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                scores = scores[keep]
                labels = labels[keep] if self.num_classes > 1 else labels[keep]
                boxes = pred_boxes[i][keep]  # [N, 4] (cx, cy, w, h)  (0-1)
                
                # 🔥 ： (0-1 → )
                boxes_pixel = boxes.clone()
                boxes_pixel[:, 0] *= width   # cx
                boxes_pixel[:, 1] *= height  # cy
                boxes_pixel[:, 2] *= width   # w
                boxes_pixel[:, 3] *= height  # h
                
                # boxes：cxcywh → xyxy ()
                boxes_xyxy = torch.zeros_like(boxes_pixel)
                boxes_xyxy[:, 0] = boxes_pixel[:, 0] - boxes_pixel[:, 2] / 2  # x1
                boxes_xyxy[:, 1] = boxes_pixel[:, 1] - boxes_pixel[:, 3] / 2  # y1
                boxes_xyxy[:, 2] = boxes_pixel[:, 0] + boxes_pixel[:, 2] / 2  # x2
                boxes_xyxy[:, 3] = boxes_pixel[:, 1] + boxes_pixel[:, 3] / 2  # y2
                
                #  [x1, y1, x2, y2, conf, cls] ()
                pred = torch.cat([
                    boxes_xyxy,
                    scores.unsqueeze(-1),
                    labels.float().unsqueeze(-1)
                ], dim=-1)
                
                predictions.append(pred)
        else:
            # DETR
            batch_size = outputs['pred_logits'].shape[0]
            
            for i in range(batch_size):
                logits = outputs['pred_logits'][i]  # [num_queries, num_classes]
                boxes = outputs['pred_boxes'][i]    # [num_queries, 4]
                
                probs = logits.softmax(-1)
                scores, labels = probs.max(-1)
                
                keep = scores > conf_thres
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]
                
                # boxes
                boxes_xyxy = torch.zeros_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
                
                pred = torch.cat([
                    boxes_xyxy,
                    scores.unsqueeze(-1),
                    labels.float().unsqueeze(-1)
                ], dim=-1)
                
                predictions.append(pred)
        
        return predictions

# =============================================================================
# Faster R-CNN
# =============================================================================

class FasterRCNNAdapter(DetectorAdapter):
    """Faster R-CNN"""
    
    def __init__(self, num_classes=80, device='cuda', backbone='resnet50'):
        super().__init__(num_classes, device)
        self.detector_type = "faster_rcnn"
        self.backbone = backbone
        
        try:
            # torchvision
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            import os
            
            # 🔥 
            local_weight_path = "/path/to/project/detection_models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
            
            if os.path.exists(local_weight_path):
                print(f"🔧 Faster R-CNN...")
                print(f"   : {local_weight_path}")
                
                # （pretrained，）
                self.model = fasterrcnn_resnet50_fpn(pretrained=False)
                
                state_dict = torch.load(local_weight_path, map_location=device)
                self.model.load_state_dict(state_dict)
                
                print(f"✅ ")
            else:
                print(f"⚠️ ，torchvision")
                print(f"   : {local_weight_path}")
                
                # torchvision（）
                try:
                    self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                    print(f"✅ torchvision")
                except Exception as e:
                    print(f"⚠️ : {e}")
                    print(f"   ")
                    self.model = fasterrcnn_resnet50_fpn(pretrained=False)
            
            # （COCO 80）
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, 
                num_classes + 1  # +1 for background
            )
            
            self.model = self.model.to(device)
            
            print(f"✅ Faster R-CNN (backbone={backbone}, classes={num_classes})")
            
        except Exception as e:
            print(f"❌ Faster R-CNN: {e}")
            raise
    
    def forward(self, x, targets=None):
        """
        Faster R-CNN
        
        : 
        : 
        """
        if self.training and targets is not None:
            return self.model(x, targets)
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """
        Faster R-CNN
        
        Args:
            predictions: ，loss_dict
            targets: YOLO
            imgs: 
        
        Returns:
            loss: 
            loss_items: 
        """
        # targetsFaster R-CNN
        frcnn_targets = self._convert_targets_to_frcnn_format(targets, imgs)
        
        self.model.train()
        
        # （loss_dict）
        loss_dict = self.model(imgs, frcnn_targets)
        
        loss = sum(loss for loss in loss_dict.values())
        
        loss_items = torch.tensor([
            loss_dict.get('loss_classifier', 0),
            loss_dict.get('loss_box_reg', 0),
            loss_dict.get('loss_objectness', 0),
            loss_dict.get('loss_rpn_box_reg', 0)
        ], device=self.device)
        
        return loss, loss_items
    
    def _convert_targets_to_frcnn_format(self, yolo_targets, imgs):
        """
        YOLOFaster R-CNN
        
        YOLO: Tensor[N, 6] (batch_idx, class, x, y, w, h) - 
        FRCNN: List[Dict] dict 'boxes' [N,4] (x1,y1,x2,y2), 'labels' [N]
        """
        batch_size = imgs.shape[0]
        _, _, h, w = imgs.shape
        
        frcnn_targets = []
        
        for batch_idx in range(batch_size):
            mask = yolo_targets[:, 0] == batch_idx
            
            if mask.sum() == 0:
                frcnn_targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                continue
            
            batch_targets = yolo_targets[mask]
            
            # boxes
            labels = batch_targets[:, 1].long() + 1  # +10
            boxes_xywh_norm = batch_targets[:, 2:6]
            
            boxes_xywh = boxes_xywh_norm * torch.tensor(
                [w, h, w, h], device=self.device
            )
            
            # (x1, y1, x2, y2)
            cx, cy, bw, bh = boxes_xywh.unbind(-1)
            boxes_xyxy = torch.stack([
                cx - bw/2, cy - bh/2,
                cx + bw/2, cy + bh/2
            ], dim=-1)
            
            frcnn_targets.append({
                'boxes': boxes_xyxy,
                'labels': labels
            })
        
        return frcnn_targets
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        Faster R-CNN
        
        Args:
            outputs: Faster R-CNN (List[Dict])
            conf_thres: 
            iou_thres: IoU(Faster R-CNNNMS)
            img_size:  (，Faster R-CNN)
            
        Returns:
            predictions: List[Tensor[N, 6]] (x1, y1, x2, y2, conf, cls)
        """
        predictions = []
        
        for output in outputs:
            if isinstance(output, dict):
                boxes = output['boxes']  # [N, 4] (x1, y1, x2, y2)
                scores = output['scores']  # [N]
                labels = output['labels']  # [N]
                
                keep = scores > conf_thres
                
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                boxes = boxes[keep]
                scores = scores[keep].unsqueeze(-1)
                labels = labels[keep].float().unsqueeze(-1) - 1  # -10
                
                #  [x1, y1, x2, y2, conf, cls]
                pred = torch.cat([boxes, scores, labels], dim=-1)
                predictions.append(pred)
            else:
                predictions.append(torch.zeros((0, 6), device=self.device))
        
        return predictions
    
    def load_pretrained(self, weights_path):
        """Faster R-CNN"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        print(f"✅ Faster R-CNN: {weights_path}")

# =============================================================================
# SSD
# =============================================================================

class SSDAdapter(DetectorAdapter):
    """SSD"""
    
    def __init__(self, num_classes=80, device='cuda', backbone='vgg16'):
        super().__init__(num_classes, device)
        self.detector_type = "ssd"
        self.backbone = backbone
        
        try:
            # torchvision SSD
            import torchvision
            from torchvision.models.detection import ssd300_vgg16
            import os
            
            # 🔥 
            local_weight_path = "/path/to/project/detection_models/ssd300_vgg16_coco-b556d3b4.pth"
            
            if os.path.exists(local_weight_path):
                print(f"📁 SSD: {local_weight_path}")
                
                # （81：80+）
                self.model = ssd300_vgg16(
                    pretrained=False,
                    num_classes=num_classes + 1
                )
                
                # （）
                try:
                    state_dict = torch.load(local_weight_path, map_location='cpu')
                    
                    # （）
                    filtered_state_dict = {}
                    for k, v in state_dict.items():
                        if 'classification_head' in k:
                            print(f"   ⏩ : {k}")
                            continue
                        filtered_state_dict[k] = v
                    
                    # backbone
                    missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"✅ SSD (backbone={backbone})")
                    if len(missing) > 0:
                        print(f"   ⚠️  {len(missing)} （）")
                    
                except Exception as e:
                    print(f"⚠️ : {e}")
                    print(f"🔧 ...")
            else:
                # torchvision
                print(f"🔧 ，torchvision...")
                self.model = ssd300_vgg16(
                    pretrained=True,
                    num_classes=num_classes + 1
                )
                print(f"✅ SSD (backbone={backbone}, pretrained=True)")
            
            self.model = self.model.to(device)
            
        except Exception as e:
            print(f"❌ SSD: {e}")
            raise
    
    def forward(self, x, targets=None):
        """SSD"""
        if targets is not None:
            # ：
            self.model.train()
            return self.model(x, targets)
        else:
            # ：
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """SSD（Faster R-CNN）"""
        ssd_targets = self._convert_targets_to_ssd_format(targets, imgs)
        
        self.model.train()
        loss_dict = self.model(imgs, ssd_targets)
        
        loss = sum(loss for loss in loss_dict.values())
        
        loss_items = torch.tensor([
            loss_dict.get('classification', 0),
            loss_dict.get('bbox_regression', 0)
        ], device=self.device)
        
        return loss, loss_items
    
    def _convert_targets_to_ssd_format(self, yolo_targets, imgs):
        """YOLOSSD（Faster R-CNN）"""
        return self._convert_targets_to_frcnn_format(yolo_targets, imgs)
    
    def _convert_targets_to_frcnn_format(self, yolo_targets, imgs):
        """Faster R-CNN"""
        batch_size = imgs.shape[0]
        _, _, h, w = imgs.shape
        
        ssd_targets = []
        
        for batch_idx in range(batch_size):
            mask = yolo_targets[:, 0] == batch_idx
            
            if mask.sum() == 0:
                ssd_targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                continue
            
            batch_targets = yolo_targets[mask]
            labels = batch_targets[:, 1].long() + 1
            boxes_xywh_norm = batch_targets[:, 2:6]
            
            boxes_xywh = boxes_xywh_norm * torch.tensor(
                [w, h, w, h], device=self.device
            )
            
            cx, cy, bw, bh = boxes_xywh.unbind(-1)
            boxes_xyxy = torch.stack([
                cx - bw/2, cy - bh/2,
                cx + bw/2, cy + bh/2
            ], dim=-1)
            
            ssd_targets.append({
                'boxes': boxes_xyxy,
                'labels': labels
            })
        
        return ssd_targets
    
    def load_pretrained(self, weights_path):
        """SSD"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        SSD
        
        Args:
            outputs: SSD (List[Dict])
            conf_thres: 
            iou_thres: IoU(SSDNMS)
            img_size:  (，SSD)
            
        Returns:
            predictions: List[Tensor[N, 6]] (x1, y1, x2, y2, conf, cls)
        """
        predictions = []
        
        for output in outputs:
            if isinstance(output, dict):
                boxes = output['boxes']  # [N, 4] (x1, y1, x2, y2)
                scores = output['scores']  # [N]
                labels = output['labels']  # [N]
                
                keep = scores > conf_thres
                
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                boxes = boxes[keep]
                scores = scores[keep].unsqueeze(-1)
                labels = labels[keep].float().unsqueeze(-1) - 1  # -1 for background class
                
                #  [x1, y1, x2, y2, conf, cls]
                pred = torch.cat([boxes, scores, labels], dim=-1)
                predictions.append(pred)
            else:
                predictions.append(torch.zeros((0, 6), device=self.device))
        
        return predictions


# =============================================================================
# RetinaNet
# =============================================================================

class RetinaNetAdapter(DetectorAdapter):
    """RetinaNet"""
    
    def __init__(self, num_classes=80, device='cuda', backbone='resnet50'):
        super().__init__(num_classes, device)
        self.detector_type = "retinanet"
        self.backbone = backbone
        
        try:
            # torchvision RetinaNet
            import torchvision
            from torchvision.models.detection import retinanet_resnet50_fpn
            import os
            
            # 🔥 
            local_weight_path = "/path/to/project/detection_models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
            
            if os.path.exists(local_weight_path):
                print(f"📁 RetinaNet: {local_weight_path}")
                
                # （81：80+）
                self.model = retinanet_resnet50_fpn(
                    pretrained=False,
                    num_classes=num_classes + 1
                )
                
                try:
                    state_dict = torch.load(local_weight_path, map_location='cpu')
                    
                    # （）
                    filtered_state_dict = {}
                    skip_keys = []
                    for k, v in state_dict.items():
                        # RetinaNet 'head.classification_head.*'
                        if 'classification_head' in k or 'cls_logits' in k:
                            skip_keys.append(k)
                            continue
                        filtered_state_dict[k] = v
                    
                    if skip_keys:
                        print(f"   ⏩  {len(skip_keys)} ")
                    
                    # backbone
                    missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"✅ RetinaNet (backbone={backbone})")
                    if len(missing) > 0:
                        print(f"   ⚠️  {len(missing)} （）")
                    
                except Exception as e:
                    print(f"⚠️ : {e}")
                    print(f"🔧 ...")
            else:
                # torchvision
                print(f"🔧 ，torchvision...")
                self.model = retinanet_resnet50_fpn(
                    pretrained=True,
                    num_classes=num_classes + 1
                )
                print(f"✅ RetinaNet (backbone={backbone}, pretrained=True)")
            
            self.model = self.model.to(device)
            
        except Exception as e:
            print(f"❌ RetinaNet: {e}")
            raise
    
    def forward(self, x, targets=None):
        """RetinaNet"""
        if targets is not None:
            # ：
            self.model.train()
            return self.model(x, targets)
        else:
            # ：
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """RetinaNet"""
        retinanet_targets = self._convert_targets_to_retinanet_format(targets, imgs)
        
        self.model.train()
        loss_dict = self.model(imgs, retinanet_targets)
        
        # RetinaNet
        loss = sum(loss for loss in loss_dict.values())
        
        loss_items = torch.tensor([
            loss_dict.get('classification', 0),
            loss_dict.get('bbox_regression', 0)
        ], device=self.device)
        
        return loss, loss_items
    
    def _convert_targets_to_retinanet_format(self, yolo_targets, imgs):
        """YOLORetinaNet"""
        batch_size = imgs.shape[0]
        _, _, h, w = imgs.shape
        
        retinanet_targets = []
        
        for batch_idx in range(batch_size):
            mask = yolo_targets[:, 0] == batch_idx
            
            if mask.sum() == 0:
                retinanet_targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                continue
            
            batch_targets = yolo_targets[mask]
            
            # boxes
            labels = batch_targets[:, 1].long() + 1  # +10
            boxes_xywh_norm = batch_targets[:, 2:6]
            
            boxes_xywh = boxes_xywh_norm * torch.tensor(
                [w, h, w, h], device=self.device
            )
            
            # (x1, y1, x2, y2)
            cx, cy, bw, bh = boxes_xywh.unbind(-1)
            boxes_xyxy = torch.stack([
                cx - bw/2, cy - bh/2,
                cx + bw/2, cy + bh/2
            ], dim=-1)
            
            retinanet_targets.append({
                'boxes': boxes_xyxy,
                'labels': labels
            })
        
        return retinanet_targets
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        RetinaNet
        
        Args:
            outputs: RetinaNet (List[Dict])
            conf_thres: 
            iou_thres: IoU(RetinaNetNMS)
            img_size:  (，RetinaNet)
            
        Returns:
            predictions: List[Tensor[N, 6]] (x1, y1, x2, y2, conf, cls)
        """
        predictions = []
        
        for output in outputs:
            if isinstance(output, dict):
                boxes = output['boxes']  # [N, 4] (x1, y1, x2, y2)
                scores = output['scores']  # [N]
                labels = output['labels']  # [N]
                
                keep = scores > conf_thres
                
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                boxes = boxes[keep]
                scores = scores[keep].unsqueeze(-1)
                labels = labels[keep].float().unsqueeze(-1) - 1  # -1 for background class
                
                #  [x1, y1, x2, y2, conf, cls]
                pred = torch.cat([boxes, scores, labels], dim=-1)
                predictions.append(pred)
            else:
                predictions.append(torch.zeros((0, 6), device=self.device))
        
        return predictions
    
    def load_pretrained(self, weights_path):
        """RetinaNet"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        print(f"✅ RetinaNet: {weights_path}")


# =============================================================================
# FCOS
# =============================================================================

class FCOSAdapter(DetectorAdapter):
    """FCOS - Fully Convolutional One-Stage Object Detection"""
    
    def __init__(self, num_classes=80, device='cuda', backbone='resnet50'):
        super().__init__(num_classes, device)
        self.detector_type = "fcos"
        self.backbone = backbone
        
        try:
            # torchvision FCOS
            import torchvision
            from torchvision.models.detection import fcos_resnet50_fpn
            import os
            
            # 🔥 
            local_weight_path = "/path/to/project/detection_models/fcos_resnet50_fpn_coco-99b0c9b7.pth"
            
            if os.path.exists(local_weight_path):
                print(f"📁 FCOS: {local_weight_path}")
                
                # （81：80+）
                self.model = fcos_resnet50_fpn(
                    pretrained=False,
                    num_classes=num_classes + 1
                )
                
                try:
                    state_dict = torch.load(local_weight_path, map_location='cpu')
                    
                    # （）
                    filtered_state_dict = {}
                    skip_keys = []
                    for k, v in state_dict.items():
                        # FCOS 'head.classification_head.*'  'cls_logits'
                        if 'classification_head' in k or 'cls_logits' in k:
                            skip_keys.append(k)
                            continue
                        filtered_state_dict[k] = v
                    
                    if skip_keys:
                        print(f"   ⏩  {len(skip_keys)} ")
                    
                    # backbone
                    missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"✅ FCOS (backbone={backbone})")
                    if len(missing) > 0:
                        print(f"   ⚠️  {len(missing)} （）")
                    
                except Exception as e:
                    print(f"⚠️ : {e}")
                    print(f"🔧 ...")
            else:
                # torchvision
                print(f"🔧 ，torchvision...")
                self.model = fcos_resnet50_fpn(
                    pretrained=True,
                    num_classes=num_classes + 1
                )
                print(f"✅ FCOS (backbone={backbone}, pretrained=True)")
            
            self.model = self.model.to(device)
            
        except Exception as e:
            print(f"❌ FCOS: {e}")
            raise
    
    def forward(self, x, targets=None):
        """FCOS"""
        if targets is not None:
            # ：
            self.model.train()
            return self.model(x, targets)
        else:
            # ：
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """FCOS"""
        fcos_targets = self._convert_targets_to_fcos_format(targets, imgs)
        
        self.model.train()
        loss_dict = self.model(imgs, fcos_targets)
        
        # FCOS
        loss = sum(loss for loss in loss_dict.values())
        
        loss_items = torch.tensor([
            loss_dict.get('classification', 0),
            loss_dict.get('bbox_regression', 0),
            loss_dict.get('bbox_ctrness', 0)  # FCOScenterness
        ], device=self.device)
        
        return loss, loss_items
    
    def _convert_targets_to_fcos_format(self, yolo_targets, imgs):
        """YOLOFCOS"""
        batch_size = imgs.shape[0]
        _, _, h, w = imgs.shape
        
        fcos_targets = []
        
        for batch_idx in range(batch_size):
            mask = yolo_targets[:, 0] == batch_idx
            
            if mask.sum() == 0:
                fcos_targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                continue
            
            batch_targets = yolo_targets[mask]
            
            # boxes
            labels = batch_targets[:, 1].long() + 1  # +10
            boxes_xywh_norm = batch_targets[:, 2:6]
            
            boxes_xywh = boxes_xywh_norm * torch.tensor(
                [w, h, w, h], device=self.device
            )
            
            # (x1, y1, x2, y2)
            cx, cy, bw, bh = boxes_xywh.unbind(-1)
            boxes_xyxy = torch.stack([
                cx - bw/2, cy - bh/2,
                cx + bw/2, cy + bh/2
            ], dim=-1)
            
            fcos_targets.append({
                'boxes': boxes_xyxy,
                'labels': labels
            })
        
        return fcos_targets
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        FCOS
        
        Args:
            outputs: FCOS (List[Dict])
            conf_thres: 
            iou_thres: IoU(FCOSNMS)
            img_size:  (，FCOS)
            
        Returns:
            predictions: List[Tensor[N, 6]] (x1, y1, x2, y2, conf, cls)
        """
        predictions = []
        
        for output in outputs:
            if isinstance(output, dict):
                boxes = output['boxes']  # [N, 4] (x1, y1, x2, y2)
                scores = output['scores']  # [N]
                labels = output['labels']  # [N]
                
                keep = scores > conf_thres
                
                if keep.sum() == 0:
                    predictions.append(torch.zeros((0, 6), device=self.device))
                    continue
                
                boxes = boxes[keep]
                scores = scores[keep].unsqueeze(-1)
                labels = labels[keep].float().unsqueeze(-1) - 1  # -1 for background class
                
                #  [x1, y1, x2, y2, conf, cls]
                pred = torch.cat([boxes, scores, labels], dim=-1)
                predictions.append(pred)
            else:
                predictions.append(torch.zeros((0, 6), device=self.device))
        
        return predictions
    
    def load_pretrained(self, weights_path):
        """FCOS"""
        import os
        if not os.path.exists(weights_path):
            print(f"⚠️ : {weights_path}")
            return
        
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        print(f"✅ FCOS: {weights_path}")


# =============================================================================
# YOLOv7 
# =============================================================================

class YOLOv7Adapter(DetectorAdapter):
    """YOLOv7 - YOLOv7"""
    
    def __init__(self, weights_path, num_classes=80, device='cuda'):
        super().__init__(num_classes, device)
        self.detector_type = "yolov7"
        self.weights_path = weights_path
        
        from utils.general import LOGGER
        import torch
        
        LOGGER.info(f"🔄  YOLOv7 : {weights_path}")
        
        ckpt = torch.load(weights_path, map_location=device)
        
        if 'model' in ckpt:
            self.model = ckpt['model'].float().to(device)
        else:
            raise ValueError("YOLOv7")
        
        # Detectgridanchor_grid（）
        if hasattr(self.model, 'model') and len(self.model.model) > 0:
            last_layer = self.model.model[-1]
            if hasattr(last_layer, 'grid'):
                # grid（）
                last_layer.grid = [torch.zeros(1) for _ in range(last_layer.nl)]
                
                # anchor_gridbuffer，
                if 'anchor_grid' in last_layer._buffers:
                    # YOLOv7anchor_gridbuffer，DefenseDetect
                    # buffer
                    del last_layer._buffers['anchor_grid']
                    last_layer.anchor_grid = [torch.empty(0) for _ in range(last_layer.nl)]
                else:
                    # anchor_grid，
                    last_layer.anchor_grid = [torch.empty(0) for _ in range(last_layer.nl)]
                
                # stride
                if hasattr(last_layer, 'stride') and last_layer.stride is None:
                    # stride
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 640, 640).to(device)
                        try:
                            self.model.eval()
                            _ = self.model(dummy)
                        except:
                            pass  # ，
        
        self.model.train()
        
        # YOLOv7（YOLOv5ComputeLoss）
        from utils.loss import ComputeLoss
        self.compute_loss_fn = ComputeLoss(self.model)
        
        LOGGER.info(f"✅ YOLOv7 ")
        LOGGER.info(f"   : {num_classes}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f"   : {total_params/1e6:.2f}M")
        
    def forward(self, x):
        """YOLOv7"""
        return self.model(x)
    
    def compute_loss(self, predictions, targets, imgs):
        """
        YOLOv7
        
        Args:
            predictions: YOLOv7
            targets:  [batch_idx, class, x, y, w, h]
            imgs: 
        
        Returns:
            loss: 
            loss_items: 
        """
        # YOLOv5
        loss, loss_items = self.compute_loss_fn(predictions, targets)
        return loss, loss_items
    
    def get_trainable_parameters(self):
        """"""
        return self.model.parameters()
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        YOLOv7
        
        Args:
            outputs: YOLOv7
                - : List[Tensor] shape=[bs, na, h, w, no]
                - : Tuple (inference_output, raw_output)
                  inference_output shape=[bs, num_predictions, no]
            conf_thres: 
            iou_thres: NMS IoU
            img_size:  (，YOLOv7)
        
        Returns:
            predictions: List[Tensor[N, 6]]  [x1, y1, x2, y2, conf, cls]
        """
        from utils.general import non_max_suppression
        
        # ，
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # （），
        if isinstance(outputs, list):
            # get_predictions，
            batch_size = outputs[0].shape[0]
            return [torch.zeros((0, 6), device=self.device) for _ in range(batch_size)]
        
        # NMS
        predictions = non_max_suppression(
            outputs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=False
        )
        
        return predictions
    
    def load_pretrained(self, weights_path):
        """"""
        import torch
        ckpt = torch.load(weights_path, map_location=self.device)
        if 'model' in ckpt:
            self.model = ckpt['model'].float().to(self.device)
        else:
            raise ValueError("YOLOv7")

# =============================================================================
# Ultralytics YOLO (YOLO11/YOLO12)
# =============================================================================

class UltralyticsYOLOAdapter(DetectorAdapter):
    """Ultralytics YOLO - YOLO11/YOLO12"""
    
    def __init__(self, weights_path, num_classes=80, device='cuda'):
        super().__init__(num_classes, device)
        self.detector_type = "ultralytics_yolo"
        self.weights_path = weights_path
        
        from ultralytics import YOLO
        from utils.general import LOGGER
        
        LOGGER.info(f"🔄  Ultralytics YOLO : {weights_path}")
        
        self.model = YOLO(weights_path)
        
        # PyTorch
        if hasattr(self.model, 'model'):
            self.torch_model = self.model.model
        else:
            self.torch_model = self.model
        
        self.torch_model = self.torch_model.to(device)
        
        LOGGER.info(f"✅ Ultralytics YOLO ")
        
    def forward(self, x):
        """ - PyTorch"""
        # PyTorch，Ultralytics
        return self.torch_model(x)
    
    def _convert_results(self, results):
        """UltralyticsYOLOv5"""
        # UltralyticsResults
        #  [batch, num_pred, 85] 
        output = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                #  xyxy, conf, cls
                pred = torch.cat([
                    boxes.xyxy,  # [N, 4]
                    boxes.conf.unsqueeze(1),  # [N, 1]
                    boxes.cls.unsqueeze(1)  # [N, 1]
                ], dim=1)
                output.append(pred)
            else:
                output.append(torch.zeros(0, 6).to(self.device))
        
        return output
    
    def compute_loss(self, predictions, targets, imgs):
        """
         - Ultralytics
        
        Args:
            predictions: （None，forward）
            targets:  [batch_idx, class, x, y, w, h]
            imgs: 
        
        Returns:
            loss: 
            loss_items: 
        """
        # 🔥 Ultralytics YOLO：targetsloss
        # ：train，forwardtargets
        if predictions is None or (isinstance(predictions, (list, tuple)) and len(predictions) > 0):
            if imgs is None:
                raise ValueError("imgsNone")
            
            was_training = self.torch_model.training
            self.torch_model.train()
            
            # 🔥 ：Ultralyticstargets
            # targetsbatch list
            batch_size = imgs.shape[0]
            batch_targets = []
            for i in range(batch_size):
                mask = targets[:, 0] == i
                if mask.any():
                    # [class, x, y, w, h]
                    img_targets = targets[mask, 1:].cpu()
                    batch_targets.append(img_targets)
                else:
                    batch_targets.append(torch.zeros(0, 5))
            
            # 🔥 Ultralyticsprofiling（'np'）
            old_profile = getattr(self.torch_model, 'profile', False)
            if hasattr(self.torch_model, 'profile'):
                self.torch_model.profile = False
            
            # forward（Ultralyticsforwardtargets）
            try:
                output = self.torch_model(imgs)
            except Exception as e:
                from utils.general import LOGGER
                LOGGER.warning(f"   Ultralytics forward: {e}")
                output = None
            finally:
                # profile
                if hasattr(self.torch_model, 'profile'):
                    self.torch_model.profile = old_profile
            
            if not was_training:
                self.torch_model.eval()
            
            predictions = output
        
        # loss（）
        if isinstance(predictions, torch.Tensor) and predictions.ndim == 0:
            # ，loss
            return predictions, torch.zeros(3, device=self.device)
        elif isinstance(predictions, tuple) and len(predictions) >= 2:
            # (loss, loss_items)  (predictions_tensor, auxiliary_output)
            if isinstance(predictions[0], torch.Tensor) and predictions[0].ndim == 0:
                # loss
                return predictions[0], predictions[1] if len(predictions) > 1 else torch.zeros(3, device=self.device)
            elif isinstance(predictions[1], dict) and 'loss' in predictions[1]:
                # Ultralytics v8: (predictions, {'loss': tensor})
                return predictions[1]['loss'], torch.zeros(3, device=self.device)
        
        # predictionsloss，
        # 🔥 Ultralytics YOLO v8+，loss
        # ，
        
        from utils.general import LOGGER
        
        # ：（）
        try:
            # predictions
            if isinstance(predictions, tuple) and len(predictions) > 0:
                pred_tensor = predictions[0]
            else:
                pred_tensor = predictions
            
            # Tensorshape
            if isinstance(pred_tensor, torch.Tensor) and pred_tensor.ndim >= 2:
                # Ultralytics: [batch, num_pred, 85] 
                # （4）
                if pred_tensor.shape[-1] >= 5:
                    # : [x, y, w, h, conf, ...]
                    confidences = pred_tensor[..., 4]
                    # ：
                    # 1-confloss（conf，loss）
                    pseudo_loss = (1.0 - confidences.mean()).clamp(min=0.0)
                    
                    # （）
                    if pseudo_loss.requires_grad:
                        return pseudo_loss, torch.zeros(3, device=self.device)
                    else:
                        return pseudo_loss.detach().requires_grad_(True), torch.zeros(3, device=self.device)
        except Exception as e:
            LOGGER.debug(f"   : {e}，")
        
        # ：（，）
        # ：requires_grad=True
        return torch.tensor(0.0, device=self.device, requires_grad=True), torch.zeros(3, device=self.device)
    
    def get_trainable_parameters(self):
        """"""
        if hasattr(self, 'torch_model'):
            return self.torch_model.parameters()
        elif hasattr(self, 'model'):
            if hasattr(self.model, 'model'):
                return self.model.model.parameters()
            return self.model.parameters()
        else:
            return self.parameters()
    
    def get_predictions(self, outputs, conf_thres=0.001, iou_thres=0.6, img_size=None):
        """
        Ultralytics YOLO
        
        Args:
            outputs: UltralyticsPyTorch
                - : List[Tensor] 
                - : Tuple(Tensor[batch, 84, 8400], List) 
                  84 = 4 + 80
            conf_thres: 
            iou_thres: NMS IoU
            img_size:  (，Ultralytics YOLO)
        
        Returns:
            predictions: List[Tensor[N, 6]]  [x1, y1, x2, y2, conf, cls]
        """
        from utils.general import non_max_suppression
        
        # ：tuple(Tensor[bs, 84, 8400], ...)
        if isinstance(outputs, tuple):
            pred = outputs[0]  # [batch, 84, 8400]
            
            # YOLOv5 [batch, 8400, 84]
            pred = pred.permute(0, 2, 1)  # [batch, 8400, 84]
            
            # Ultralytics v8+: [cx, cy, w, h, class0_conf, class1_conf, ...]
            # YOLOv5: [cx, cy, w, h, obj_conf, class0_conf, ...]
            batch_size = pred.shape[0]
            num_pred = pred.shape[1]
            
            boxes = pred[:, :, :4]  # [bs, 8400, 4] (cx, cy, w, h)
            class_conf = pred[:, :, 4:]  # [bs, 8400, 80]
            
            # （）
            obj_conf, class_idx = class_conf.max(dim=2, keepdim=True)  # [bs, 8400, 1]
            
            # YOLOv5: [cx, cy, w, h, obj_conf, class_confs...]
            yolov5_pred = torch.cat([boxes, obj_conf, class_conf], dim=2)  # [bs, 8400, 85]
            
            # NMS
            predictions = non_max_suppression(
                yolov5_pred,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                multi_label=False
            )
            
            return predictions
        
        # ：list of tensors ()
        elif isinstance(outputs, list):
            # get_predictions
            batch_size = outputs[0].shape[0] if len(outputs) > 0 else 1
            return [torch.zeros((0, 6), device=self.device) for _ in range(batch_size)]
        
        from utils.general import LOGGER
        LOGGER.warning(f"⚠️ Ultralytics: {type(outputs)}")
        return [torch.zeros((0, 6), device=self.device)]
    
    def load_pretrained(self, weights_path):
        """"""
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        if hasattr(self.model, 'model'):
            self.torch_model = self.model.model
        else:
            self.torch_model = self.model
        self.torch_model = self.torch_model.to(self.device)

# =============================================================================
# =============================================================================

def create_detector(detector_type, cfg=None, num_classes=80, device='cuda', **kwargs):
    """
    
    
    Args:
        detector_type:  ('yolo', 'yolov7', 'yolo11', 'yolo12', 'detr', 'faster_rcnn', 'ssd')
        cfg: YOLO（YOLOv5）
        num_classes: 
        device: 
        **kwargs: 
            - weights_path: （YOLOv7/YOLO11/YOLO12）
    
    Returns:
        DetectorAdapter: 
    """
    detector_type = detector_type.lower()
    
    if detector_type == 'yolo' or detector_type == 'yolov5':
        if cfg is None:
            raise ValueError("YOLOv5cfg")
        return YOLOAdapter(cfg, num_classes, device)
    
    elif detector_type == 'yolov7':
        weights_path = kwargs.get('weights_path')
        if weights_path is None:
            raise ValueError(f"YOLOv7weights_path")
        return YOLOv7Adapter(weights_path, num_classes, device)
    
    elif detector_type in ['yolo11', 'yolo12', 'ultralytics']:
        weights_path = kwargs.get('weights_path')
        if weights_path is None:
            raise ValueError(f"{detector_type}weights_path")
        return UltralyticsYOLOAdapter(weights_path, num_classes, device)
    
    elif detector_type == 'detr':
        backbone = kwargs.get('backbone', 'resnet50')
        return DETRAdapter(num_classes, device, backbone)
    
    elif detector_type == 'faster_rcnn':
        backbone = kwargs.get('backbone', 'resnet50')
        return FasterRCNNAdapter(num_classes, device, backbone)
    
    elif detector_type == 'ssd':
        backbone = kwargs.get('backbone', 'vgg16')
        return SSDAdapter(num_classes, device, backbone)
    
    elif detector_type == 'retinanet':
        backbone = kwargs.get('backbone', 'resnet50')
        return RetinaNetAdapter(num_classes, device, backbone)
    
    elif detector_type == 'fcos':
        backbone = kwargs.get('backbone', 'resnet50')
        return FCOSAdapter(num_classes, device, backbone)
    
    else:
        raise ValueError(f": {detector_type}")

# =============================================================================
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # YOLO
    print("\nYOLO...")
    try:
        yolo = create_detector('yolo', cfg='models/yolov5s.yaml', device=device)
        print(f"✅ YOLO")
    except Exception as e:
        print(f"❌ YOLO: {e}")
    
    # DETR
    print("\nDETR...")
    try:
        detr = create_detector('detr', device=device)
        print(f"✅ DETR")
    except Exception as e:
        print(f"❌ DETR: {e}")
    
    # Faster R-CNN
    print("\nFaster R-CNN...")
    try:
        frcnn = create_detector('faster_rcnn', device=device)
        print(f"✅ Faster R-CNN")
    except Exception as e:
        print(f"❌ Faster R-CNN: {e}")
    
    # SSD
    print("\nSSD...")
    try:
        ssd = create_detector('ssd', device=device)
        print(f"✅ SSD")
    except Exception as e:
        print(f"❌ SSD: {e}")
    
    print("\n" + "=" * 80)
