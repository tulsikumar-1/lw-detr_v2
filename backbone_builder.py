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
                 out_channels:int=256,
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
                 drop_path=drop_path,
                 out_channels=out_channels,
                 out_feature_indexes=out_feature_indexes,
                 projector_scale=projector_scale,
                 embed_dim= embed_dim,
        )

    model = Joiner(backbone, position_embedding)
    return model


def get_param_dict( lr,vit_encoder_num_layers,lr_encoder,lr_vit_layer_decay,weight_decay,lr_component_decay, model_without_ddp: nn.Module):
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = model_without_ddp.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs( vit_encoder_num_layers,lr_encoder,lr_vit_layer_decay,weight_decay,lr_component_decay, prefix= "backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = 'transformer.decoder'
    decoder_params = [
        p
        for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad
    ]

    decoder_param_lr_pairs = [
        {"params": param, "lr": lr * lr_component_decay} 
        for param in decoder_params
    ]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters() if (
            n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [
        {"params": param, "lr": lr} 
        for param in other_params
    ]
    
    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
    )

    return final_param_dicts