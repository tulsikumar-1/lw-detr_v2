# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:17:55 2024

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedChannelReducer(nn.Module):
    def __init__(self, in_channels: int = 384, out_channels: int = 192):
        super(FixedChannelReducer, self).__init__()
        # Define a 1x1 convolutional layer
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights to fixed values (e.g., all ones or other fixed values)
        with torch.no_grad():
            self.conv1x1.weight.fill_(1.0 / in_channels)  # Example fixed initialization
        
        # Ensure the weights are not trainable
        for param in self.conv1x1.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.conv1x1(x)

class CosineSimilarityLoss(nn.Module):
    def __init__(self, in_channels: int = 384, out_channels: int = 192):
        super(CosineSimilarityLoss, self).__init__()
        self.fixed_channel_reducer = FixedChannelReducer(in_channels, out_channels)

    def forward(self, teacher_features, student_features):
        # Project teacher's feature map to match the number of channels of the student's feature map
        projected_teacher_features = self.fixed_channel_reducer(teacher_features)

        # Flatten the features for cosine similarity computation
        projected_teacher_features = projected_teacher_features.view(projected_teacher_features.size(0), projected_teacher_features.size(1), -1).mean(dim=-1)
        student_features = student_features.view(student_features.size(0), student_features.size(1), -1).mean(dim=-1)

        # Cosine similarity loss (want to minimize 1 - cosine similarity)
        cosine_sim = F.cosine_similarity(projected_teacher_features, student_features, dim=1)
        cosine_loss = 1 - cosine_sim.mean()

        return cosine_loss