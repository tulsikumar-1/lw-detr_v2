# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou,ciou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 0, cost_bbox: float = 0, cost_giou: float = 0,cost_ciou: float = 0, focal_alpha: float = 0.25, use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_ciou = cost_ciou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_ciou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        out_bbox = torch.clamp(out_bbox, min=0)
        # Compute the giou cost betwen boxes
        if (out_bbox < 0).any():
          raise ValueError("Predicted boxes contain negative values")
        
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou
        
        ciou_loss = ciou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_ciou = -ciou_loss

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 3.0
        

                # Safe computation of focal loss
        epsilon = 1e-8
        out_prob = torch.clamp(out_prob, epsilon, 1 - epsilon)  # Avoid log(0)
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-out_prob.log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if not torch.isfinite(cost_bbox).all():
            raise ValueError("cost_bbox contains non-finite values (NaN or Inf)")
        if not torch.isfinite(cost_class).all():
            raise ValueError("cost_class contains non-finite values (NaN or Inf)")
        if not torch.isfinite(cost_giou).all():
            raise ValueError("cost_giou contains non-finite values (NaN or Inf)")
        if not torch.isfinite(cost_ciou).all():
            raise ValueError("cost_ciou contains non-finite values (NaN or Inf)")

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou+ self.cost_ciou * cost_ciou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

