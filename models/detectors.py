"""Person-detector backends used by the patch defense."""
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from torchvision.models import detection as tvd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d


@contextmanager
def loss_mode(detector):
    states = [(m,m.training) for m in detector.modules()]
    detector.train()
    for m in detector.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            m.eval()  # no running-stat changes during frozen/attack scoring
    try:
        yield
    finally:
        for m,state in states:
            m.training=state


class TinyBackbone(nn.Module):
    """SMOKE ONLY. Never selected by formal configs or reported as paper architecture."""
    out_channels=16
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,8,3,2,1),nn.ReLU(),nn.Conv2d(8,16,3,2,1),nn.ReLU())
    def forward(self,x):
        return self.net(x)


def freeze_batchnorm_structure(module,eps=0.):
    """Match torchvision's public COCO-v1 backbone, which uses FrozenBN."""
    for name,child in list(module.named_children()):
        if isinstance(child,nn.BatchNorm2d):
            frozen=FrozenBatchNorm2d(child.num_features,eps=eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight);frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean);frozen.running_var.copy_(child.running_var)
            setattr(module,name,frozen)
        else:freeze_batchnorm_structure(child,eps)


def build_detector(kind='fasterrcnn', size=416, initialization='random', public_weights=None, tiny=False,roi_backend='native'):
    if roi_backend not in ['native','torch']:raise ValueError(roi_backend)
    if tiny:
        result=FasterRCNN(TinyBackbone(),num_classes=2,min_size=size,max_size=size,
            rpn_anchor_generator=AnchorGenerator(((8,16,32),),((.5,1.,2.),)),
            box_roi_pool=MultiScaleRoIAlign(['0'],3,2),rpn_pre_nms_top_n_train=40,
            rpn_post_nms_top_n_train=20,rpn_pre_nms_top_n_test=40,rpn_post_nms_top_n_test=20,
            box_batch_size_per_image=16,rpn_batch_size_per_image=32,box_score_thresh=.001,
            box_nms_thresh=.6,box_detections_per_img=100)
        if roi_backend=='torch':
            from .roi_backend import enable_higher_order_roi_align
            result=enable_higher_order_roi_align(result)
        return DetectionAdapter(result)
    if kind not in ['fasterrcnn','retinanet','fcos','ssd']:
        raise NotImplementedError(f'{kind}: exact YOLO instantiation is not implemented; no substitution')
    factories={'fasterrcnn':tvd.fasterrcnn_resnet50_fpn,'retinanet':tvd.retinanet_resnet50_fpn,
               'fcos':tvd.fcos_resnet50_fpn,'ssd':tvd.ssd300_vgg16}
    kwargs=dict(weights=None,weights_backbone=None)
    if kind!='ssd':kwargs.update(min_size=size,max_size=size)
    if initialization=='random':kwargs['num_classes']=2
    elif initialization not in ['public_coco','architecture_only_coco']:raise ValueError(initialization)
    detector=factories[kind](**kwargs)
    if initialization=='architecture_only_coco':
        # Restore architecture/buffers only; every weight is subsequently loaded
        # strictly from the exported trained bundle. No external weight access.
        if kind in ['fasterrcnn','retinanet','fcos']:freeze_batchnorm_structure(detector)
    if initialization=='public_coco':
        if public_weights is None:raise ValueError('Public weight file and hash provenance required')
        path=Path(public_weights).resolve()
        root=Path(__file__).resolve().parents[1]
        if not path.is_relative_to(root/'checkpoints'/'public'):
            raise ValueError('Only verified project-local public weights are accepted')
        provenance=root/'checkpoints/public/provenance.json'
        import json,hashlib
        records=json.loads(provenance.read_text())
        match=next((r for r in records if r['file']==str(path.relative_to(root))),None)
        if match is None or hashlib.sha256(path.read_bytes()).hexdigest()!=match['sha256']:
            raise ValueError('Public weight integrity/provenance mismatch')
        if kind in ['fasterrcnn','retinanet','fcos']:freeze_batchnorm_structure(detector)
        detector.load_state_dict(torch.load(path,map_location='cpu'),strict=True)
        # Keep original COCO class head. Targets use label1, predictions filter label1.
    if kind=='fasterrcnn':
        detector.roi_heads.score_thresh=.001
        detector.roi_heads.nms_thresh=.6
        detector.roi_heads.detections_per_img=100
    else:
        detector.score_thresh=.001
        detector.nms_thresh=.6
        detector.detections_per_img=100
    if roi_backend=='torch':
        if kind!='fasterrcnn':raise ValueError('ROI backend only applies to Faster R-CNN')
        from .roi_backend import enable_higher_order_roi_align
        detector=enable_higher_order_roi_align(detector)
    return DetectionAdapter(detector)


class DetectionAdapter(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector=detector

    def loss(self, x, targets):
        with loss_mode(self.detector):
            components=self.detector(list(x),targets)
        return sum(components.values()),components

    def predict(self,x):
        states=[(m,m.training) for m in self.detector.modules()]
        self.detector.eval()
        try:
            outputs=self.detector(list(x))
            return [{k:v[p['labels']==1] for k,v in p.items()} for p in outputs]
        finally:
            for m,state in states:m.training=state

    def soft_objectness(self,x):
        """Actual RPN logits; A12 mean anchors/levels then interpolate to input."""
        if not hasattr(self.detector,'rpn'):
            raise NotImplementedError('PDF RPN BPDA surrogate requires Faster R-CNN')
        images,_=self.detector.transform(list(x),None)
        features=self.detector.backbone(images.tensors)
        if isinstance(features,torch.Tensor):features={'0':features}
        logits,_=self.detector.rpn.head(list(features.values()))
        soft=[nn.functional.interpolate(logit.sigmoid().mean(1,keepdim=True),
                   size=images.tensors.shape[-2:],mode='bilinear',align_corners=False) for logit in logits]
        padded=torch.stack(soft).mean(0)
        # Remove transform's stride padding before returning source-grid mask.
        out=[]
        for i,(h,w) in enumerate(images.image_sizes):
            out.append(nn.functional.interpolate(padded[i:i+1,:,:h,:w],size=x.shape[-2:],
                        mode='bilinear',align_corners=False))
        return torch.cat(out)
