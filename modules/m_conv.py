# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif

from .network_blocks import BaseConv
import torch.nn as nn

class m_g_conv(nn.Module):
    def __init__(self, in_channels, out_channels, c_head_nums=4, h_head_nums=2, w_head_nums=2, kernel_size=3, groups=1, act='silu'):
        super(m_g_conv, self).__init__()

        conv = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv.append(
                BaseConv(in_channels=int(in_channels / c_head_nums), out_channels=out_channels,
                         kernel_size=kernel_size, groups=groups, act=act),
                Sea_Attention(dim=int(inchannels / c_head_nums), key_dim=int((inchannels / c_head_nums) / 4),
                              num_heads=4))
        self.cot_heads = cot_heads

        self.conv = BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, act=act)

