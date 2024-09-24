# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:38:20 2024

@author: Administrator
"""

# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from misc import NestedTensor

from ViT import ViT

from projector import MultiScaleProjector




class Backbone(nn.Module):
   
    def __init__(self,
                 
                 vit_encoder_num_layers: int = 6,
                 pretrained_encoder: str=False,
                 window_block_indexes: list=[0,2,4],
                 drop_path=0.1,
                 out_channels=256,
                 out_feature_indexes: list=[1,3,5],
                 projector_scale: list= ['P4'],
                 embed_dim:int =192
                 ):
        super(Backbone,self).__init__()
        
        img_size, num_heads = 1024, 12

        depth = vit_encoder_num_layers
        
        
        self.encoder = ViT(  # Single-scale ViT encoder
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_path_rate=drop_path,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=window_block_indexes,
                use_act_checkpoint=True,  # use checkpoint to save memory
                use_abs_pos=True,
                out_feature_indexes=out_feature_indexes,
                use_cae=True)
 


        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, \
            "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(
            P3=2.0,
            P4=1.0,
            P5=0.5,
            P6=0.25
        )
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
        )



    def forward(self, tensor_list: NestedTensor):
        """
        """
        # (H, W, B, C)
        feats_encoder = self.encoder(tensor_list.tensors)
        feats = self.projector(feats_encoder)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out,feats_encoder

    def get_named_param_lr_pairs(self, vit_encoder_num_layers,lr_encoder,lr_vit_layer_decay,weight_decay,lr_component_decay, prefix:str = "backbone.0"):
        try:
            num_layers = vit_encoder_num_layers
            backbone_key = 'backbone.0.encoder'
            named_param_lr_pairs = {}
            for n, p in self.named_parameters():
                n = prefix + "." + n
                if backbone_key in n and p.requires_grad:
                    lr = lr_encoder * get_vit_lr_decay_rate(
                        n, lr_decay_rate=lr_vit_layer_decay, 
                        num_layers=num_layers) * lr_component_decay ** 2
                    wd = weight_decay * get_vit_weight_decay_rate(n)
                    named_param_lr_pairs[n] = {
                        "params": p,
                        "lr": lr,
                        "weight_decay": wd
                    }

        except:
            raise NotImplementedError
        return named_param_lr_pairs


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate

