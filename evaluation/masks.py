"""Actual Table XIX perturbations, with zero-fill translation and no wrap."""
import torch
import torch.nn.functional as F

CONDITIONS=['gt_box_proxy','predicted','empty','eroded_3','eroded_9','eroded_17',
            'dilated_3','dilated_9','dilated_17','shift_left_8','shift_left_16','shift_right_8','shift_right_16']

def perturb_mask(mask,condition,gt_mask=None):
    if condition=='gt_box_proxy':
        if gt_mask is None:raise ValueError('GT proxy explicitly requires GT mask')
        return gt_mask.clone()
    if condition=='predicted':return mask.clone()
    if condition=='empty':return torch.zeros_like(mask)
    if condition.startswith(('eroded_','dilated_')):
        k=int(condition.split('_')[1])
        if condition.startswith('dilated_'):return F.max_pool2d(mask,k,1,k//2)
        # Zero-padded exterior for actual binary morphology.
        return -F.max_pool2d(F.pad(-mask,(k//2,)*4,value=0),k,1)
    side,n=condition.split('_')[1:];n=int(n)
    result=torch.zeros_like(mask)
    if n>=mask.shape[-1]:return result
    if side=='left':result[...,:-n]=mask[...,n:]
    elif side=='right':result[...,n:]=mask[...,:-n]
    else:raise ValueError(condition)
    return result
