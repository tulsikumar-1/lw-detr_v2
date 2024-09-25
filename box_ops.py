# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:17:58 2024

@author: Administrator
"""

# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area



def box_cxcywh_to_xyxy(x, eps=1e-10, max_val=1.0):
    # Unbind the center coordinates, width, and height
    x_c, y_c, w, h = x.unbind(-1)
    
    # Clamp width and height to avoid negative or zero values
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)
    x_c = x_c.clamp(min=eps)
    y_c = y_c.clamp(min=eps)
    
    # Compute xmin, ymin, xmax, ymax
    x_min = (x_c - 0.5 * w).clamp(min=eps, max=max_val)
    y_min = (y_c - 0.5 * h).clamp(min=eps, max=max_val)
    x_max = (x_c + 0.5 * w).clamp(min=eps, max=max_val)
    y_max = (y_c + 0.5 * h).clamp(min=eps, max=max_val)
    
    # Ensure xmin <= xmax and ymin <= ymax
    x_min = torch.min(x_min, x_max)
    y_min = torch.min(y_min, y_max)
    x_max = torch.max(x_min + eps, x_max)  # Ensure xmax >= xmin
    y_max = torch.max(y_min + eps, y_max)  # Ensure ymax >= ymin
    
    # Return the box in (xmin, ymin, xmax, ymax) format
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)



def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    union = union.clamp(min=1e-10)

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    area = area.clamp(min=1e-10)
    return( iou - (area - union) / area).clamp(min=-1,max=1)




def ciou(boxes1, boxes2):
    # Ensure boxes are in the correct format (x_min, y_min, x_max, y_max)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"Invalid boxes1: {boxes1}"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"Invalid boxes2: {boxes2}"
    
    # Continue with your IoU and CIoU calculations
    iou, union = box_iou(boxes1, boxes2)
    iou = iou.clamp(min=0.0, max=1.0)  # Ensure IoU stays between 0 and 1

    center1 = (boxes1[:, None, :2] + boxes1[:, None, 2:]) / 2  # [N, M, 2]
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # [M, 2]

    center_dist = ((center1 - center2) ** 2).sum(dim=-1)  # [N, M]
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    enclosing_diag = ((rb - lt) ** 2).sum(dim=-1).clamp(min=1e-10)  # [N, M]

    wh1 = boxes1[:, None, 2:] - boxes1[:, None, :2]  # width and height of boxes1
    wh2 = boxes2[:, 2:] - boxes2[:, :2]  # width and height of boxes2

    v = (4 / (torch.pi ** 2)) * ((torch.atan(wh1[:, :, 0] / wh1[:, :, 1].clamp(min=1e-10)) -
                                  torch.atan(wh2[:, 0] / wh2[:, 1].clamp(min=1e-10))) ** 2)  # Aspect ratio term

    alpha = v / (1 - iou + v).clamp(min=1e-10)
    penalty = (center_dist / enclosing_diag) - alpha * v
    ciou = iou - penalty

    return ciou.clamp(min=-1,max=1)  # Prevent NaN by clamping CIoU





def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)