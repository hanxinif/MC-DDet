# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif

import torch.nn as nn
import torch

'''
使用说明：使用前先构建多尺度特征，然后根据几个尺度来修改模块代码
'''

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        # self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))

    def forward(self, x):
        [y0, y1, y2] = x
        # y1 = x['1']
        # y2 = x['2']
        # y3 = x['3']

        # 全局池化并1x1卷积，融合所有通道的信息，
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        # y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight, y1_weight, y2_weight], 2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        # y3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        # x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2
        return self.project(x_att)


if __name__ == '__main__':
    # input = {}
    x0 = torch.randn(1, 256, 40, 40)
    x1 = torch.randn(1, 256, 40, 40)
    # x2 = torch.randn(1, 256, 40, 40)
    # input['3'] = torch.randn(1, 64, 20, 20)
    input = [x0, x1]
    block = MSFblock(in_channels=256)
    output = block(input)
    print(output.shape)


