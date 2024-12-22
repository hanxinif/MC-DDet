# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif


from torch import nn

from .sea_attention import Sea_Attention
from .LiteMLA import LiteMLA
from .cot_attention import CoTAttention
from .MSFblock import MSFblock

class SCMF(nn.Module):
    def __init__(self, in_channels):

        super().__init__()

        self.sea = Sea_Attention(dim=in_channels, key_dim=64, num_heads=4)
        self.mla = LiteMLA(in_channels=in_channels, out_channels=in_channels)
        self.cot = CoTAttention(in_channels)
        self.msf = MSFblock(in_channels)


    def forward(self, x):

        x0 = self.sea(x)
        x1 = self.cot(x)
        x3 = self.mla(x)
        x = self.msf([x0, x1, x3])

        return x



