"""Use torchvision's own mathematically equivalent torch ROIAlign for grad-grad.

No author implementation is copied. Native/pure-torch parity is tested. This
changes the operator backend, not the detector architecture or paper objective.
"""
from contextlib import contextmanager
import importlib
from torch import nn
from torchvision.ops import poolers


def torch_roi_align(input,boxes,output_size,spatial_scale=1.,sampling_ratio=-1,aligned=False):
    module=importlib.import_module('torchvision.ops.roi_align')
    if not hasattr(module,'_roi_align'):raise RuntimeError('Installed torchvision lacks pure-torch ROIAlign')
    if isinstance(output_size,int):output_size=(output_size,output_size)
    if isinstance(boxes,(list,tuple)):
        from torchvision.ops._utils import convert_boxes_to_roi_format
        boxes=convert_boxes_to_roi_format(boxes)
    return module._roi_align(input,boxes,spatial_scale,output_size[0],output_size[1],sampling_ratio,aligned)


@contextmanager
def use_torch_roi_align():
    # Each trainer is a separate process, with sequential model forwards.
    # Restore the binding immediately after forward; its graph retains torch ops.
    original=poolers.roi_align
    poolers.roi_align=torch_roi_align
    try:yield
    finally:poolers.roi_align=original


class HigherOrderMultiScaleRoIAlign(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,features,boxes,image_shapes):
        with use_torch_roi_align():return self.base(features,boxes,image_shapes)


def enable_higher_order_roi_align(detector):
    detector.roi_heads.box_roi_pool=HigherOrderMultiScaleRoIAlign(detector.roi_heads.box_roi_pool)
    return detector
