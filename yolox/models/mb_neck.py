# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif

import torch
from torch import nn

from ...modules.sea_attention import Sea_Attention
from .network_blocks import BaseConv

class MHBNeck(nn.Module):
    def __init__(self, inchannels, c_head_nums=4, h_head_nums=2, w_head_nums=2, act="silu"):
        super(MHBNeck, self).__init__()

        self.c_head_nums = c_head_nums
        self.h_head_nums = h_head_nums
        self.w_head_nums = w_head_nums

        groups = int((inchannels / c_head_nums) / 2)

        conv1 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv1.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3, stride=1, groups=groups,
                         act=act))
        self.conv1 = conv1

        conv2 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv2.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv2 = conv2

        conv3 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv3.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv3 = conv3

        conv4 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv4.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv4 = conv4

        conv5 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv5.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv5 = conv5
        from .attention import se_block

        se1 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            se1.append(
                se_block(int(inchannels / c_head_nums)))
        self.se1 = se1

        conv6 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv6.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv6 = conv6

        conv7 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv7.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv7 = conv7

        conv8 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv8.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv8 = conv8

        conv9 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv9.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv9 = conv9

        from .attention import cbam_block
        cbam1 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            cbam1.append(
                cbam_block(int(inchannels / c_head_nums)))
        self.cbam1 = cbam1

        conv10 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv10.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv10 = conv10

        conv11 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv11.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv11 = conv11

        conv12 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv12.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv12 = conv12

        conv13 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv13.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv13 = conv13

        se2 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            se2.append(
                se_block(int(inchannels / c_head_nums)))
        self.se2 = se2


        conv14 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv14.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv14 = conv14

        conv15 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv15.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv15 = conv15

        conv16 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv16.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv16 = conv16

        conv17 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv17.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv17 = conv17

        cbam2 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            cbam2.append(
                cbam_block(int(inchannels / c_head_nums)))
        self.cbam2 = cbam2

        conv18 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv18.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv18 = conv18

        conv19 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv19.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv19 = conv19

        conv20 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv20.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv20 = conv20

        conv21 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv21.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv21 = conv21

        se3 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            se3.append(
                cbam_block(int(inchannels / c_head_nums)))
        self.se3 = se3

        conv22 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv22.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv22 = conv22

        conv23 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            conv23.append(
                BaseConv(in_channels=int(inchannels / c_head_nums), out_channels=int(inchannels / c_head_nums), ksize=3,
                         stride=1, groups=groups,
                         act=act))
        self.conv23 = conv23

        cbam3 = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            cbam3.append(
                cbam_block(int(inchannels / c_head_nums)))
        self.cbam3 = cbam3

        cot_heads = nn.ModuleList([])
        for i in range(c_head_nums * h_head_nums * w_head_nums):
            cot_heads.append(
                Sea_Attention(dim=int(inchannels / c_head_nums), key_dim=int((inchannels / c_head_nums) / 4),
                              num_heads=4))
        self.cot_heads = cot_heads

    def forward(self, x):

        b, c, h, w = x.size()

        # 构建每个维度的切片步长
        c_length = int(c / self.c_head_nums)
        h_length = int(h / self.h_head_nums)
        w_length = int(w / self.w_head_nums)

        out = [[[None for _ in range(self.w_head_nums)] for _ in range(self.h_head_nums)] for _ in range(self.c_head_nums)]
        residual = [[[None for _ in range(self.w_head_nums)] for _ in range(self.h_head_nums)] for _ in
               range(self.c_head_nums)]

        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] =x[:, c_length * i: c_length * (i + 1), h_length * j: h_length * (j + 1), w_length * k: w_length * (k + 1)]


        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):

                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv1[count](residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):

                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.conv2[count](out[i][j][k]) + residual[i][j][k]

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):

                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv3[count](out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):



                    # 切片顺序：w->h->c

                    residual[i][j][k] = out[i][j][k] = self.se1[count](self.conv4[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv5[count](out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.conv6[count](out[i][j][k]) + residual[i][j][k]

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv7[count](out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.cbam1[count](
                        self.conv8[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv9[count](out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.conv10[count](out[i][j][k]) + residual[i][j][k]

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv11[count](out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.se2[count](
                        self.conv12[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv13[count](
                        out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.conv14[count](
                        out[i][j][k]) + residual[i][j][k]

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv15[count](
                        out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.cbam2[count](
                        self.conv16[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv17[count](
                        out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.conv18[count](
                        out[i][j][k]) + residual[i][j][k]

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv19[count](
                        out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.se3[count](
                        self.conv20[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    out[i][j][k] = self.conv21[count](
                        out[i][j][k])

                    count += 1

        count = 0
        for i in range(0, self.c_head_nums):
            for j in range(0, self.h_head_nums):
                for k in range(0, self.w_head_nums):
                    # 切片顺序：w->h->c
                    residual[i][j][k] = out[i][j][k] = self.cbam3[count](
                        self.conv22[count](out[i][j][k]) + residual[i][j][k])

                    count += 1

        for i in range(self.c_head_nums):
            for j in range(self.h_head_nums):
                out[i][j] = torch.cat(out[i][j], dim=3)

            out[i] = torch.cat(out[i], dim=2)

        out = torch.cat(out, dim=1)

        return out
