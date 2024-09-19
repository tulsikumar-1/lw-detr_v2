# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:51:06 2024

@author: Administrator
"""

from typing import Dict, List

import torch
from torch import nn

from  misc import NestedTensor
from position_encoding import build_position_encoding
from Backbone import Backbone


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False


    def forward(self, tensor_list: NestedTensor):
        """
        """
        x,feats = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos,feats



def build_backbone(vit_encoder_num_layers: int=6 ,
                 pretrained_encoder: str=False,
                 window_block_indexes: list=[0,2,4],
                 drop_path: float = 0.1,
                 out_channels=256,
                 out_feature_indexes: list=[1 ,3 ,5],
                 projector_scale: list= ['P4','P5'],
                 hidden_dim: int =256,
                 embed_dim: int =192,
                 encoding_type: str ='sine'):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, encoding_type)

        
    backbone = Backbone(
                 vit_encoder_num_layers=vit_encoder_num_layers,
                 pretrained_encoder= pretrained_encoder,
                 window_block_indexes=window_block_indexes,
                 drop_path=0.1,
                 out_channels=out_channels,
                 out_feature_indexes=[-1],
                 projector_scale=projector_scale,
                 embed_dim= embed_dim,
        )

    model = Joiner(backbone, position_embedding)
    return model