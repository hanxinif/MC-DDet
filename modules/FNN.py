# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group > 0:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=group)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if group > 0:
            self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=group)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.group = group

    def forward(self, x):
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        return x

class FNN(nn.Module):

    def __init__(self, dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=2, drop=0., group=2, drop_path=0.):
        super(FNN, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop, group=group)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norml = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.norml(x)

        return x.view(B, H, W,C).permute(0, 3, 1, 2).contiguous()