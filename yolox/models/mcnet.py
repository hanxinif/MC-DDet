#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, SPPBottleneck
from .attention import se_block, cbam_block
from .gcnet import ContextBlock2d
from modules.sea_attention import Sea_Attention
from modules.FNN import FNN


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)  
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0) 
        f_out1 = self.upsample(fpn_out1) 
        f_out1 = torch.cat([f_out1, x2], 1) 
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2) 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) 
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1) 
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        out_channels=[64,128,256],
        in_channels=[256, 512, 1024],
        depthwise=False,
        backbone=None,
        act="silu",
        low_fea=False,
    ):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        self.in_features = in_features
        self.in_channels = in_channels
        self.low_fea=low_fea
        Conv = DWConv if depthwise else BaseConv

        self.SPP = SPPBottleneck(int(out_channels[2]), int(out_channels[2]))

        self.CSP_out0 = CSPLayer(int(out_channels[2]), int(in_channels[2] * width), shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(out_channels[1]+in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(out_channels[0]+in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        if self.low_fea:
            low_feature=out_features["dark2"]
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        out0=self.SPP(x0) 
        out0=self.CSP_out0(out0)  

        out1 = self.lateral_conv0(out0)  
        out1 = self.upsample(out1)  
        out1 = torch.cat([out1, x1], 1)  
        out1 = self.C3_p4(out1)  

        out2 = self.reduce_conv1(out1)  
        out2 = self.upsample(out2)
        out2 = torch.cat([out2, x2], 1) 
        out2 = self.C3_p3(out2) 

        outputs = (out2, out1, out0)
        return outputs


class YOLOFPN_SE(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        out_channels=[64,128,256],
        in_channels=[256, 512, 1024],
        depthwise=False,
        backbone=None,
        act="silu",
        low_fea=False,
    ):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        self.in_features = in_features
        self.in_channels = in_channels
        self.low_fea=low_fea
        Conv = DWConv if depthwise else BaseConv

        self.SPP = SPPBottleneck(int(out_channels[2]), int(out_channels[2]))

        self.CSP_out0 = CSPLayer(int(out_channels[2]), int(in_channels[2] * width), shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(out_channels[1]+in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(out_channels[0]+in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # SE-Block
        self.se1 = se_block(int(out_channels[2]))
        self.se2 = se_block(int(out_channels[1]))
        self.se3 = se_block(int(out_channels[0]))


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        if self.low_fea:
            low_feature=out_features["dark2"]
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        out0=self.SPP(x0) 
        out0 = self.se1(out0)
        out0=self.CSP_out0(out0)

        out1 = self.lateral_conv0(out0) 
        out1 = self.upsample(out1) 
        x1 = self.se2(x1)
        out1 = torch.cat([out1, x1], 1) 
        out1 = self.C3_p4(out1)  

        out2 = self.reduce_conv1(out1)  
        out2 = self.upsample(out2)  
        x2 = self.se3(x2)
        out2 = torch.cat([out2, x2], 1) 
        out2 = self.C3_p3(out2)

        outputs = (out2, out1, out0)
        return outputs


class MCNet(nn.Module):
    """
    Channel-modulated FPN (CM-FPN) in paper
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        out_channels=[64,128,256],
        in_channels=[256, 512, 1024],
        depthwise=False,
        backbone=None,
        act="silu",
        low_fea=False,
    ):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        self.in_features = in_features
        self.in_channels = in_channels
        self.low_fea=low_fea
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") # 没有权重更新
        self.downsample = nn.AvgPool2d(2, stride=2) # 没有权重更新

        groups = 32
        conv_channels = 128

        self.conv1 = BaseConv(in_channels=480, out_channels=conv_channels, ksize=3, stride=1, groups=32,
                              act=act)

        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                BaseConv(in_channels=conv_channels, out_channels=conv_channels, ksize=3, stride=1, groups=groups,
                         act=act)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                BaseConv(in_channels=conv_channels, out_channels=conv_channels, ksize=3, stride=1, groups=groups,
                         act=act)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                BaseConv(in_channels=conv_channels, out_channels=conv_channels, ksize=3, stride=1, groups=groups,
                         act=act)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups),
                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                          groups=groups)
            )

        ])

        self.atts = nn.ModuleList([se_block(conv_channels),
                                   cbam_block(conv_channels),
                                   se_block(conv_channels),
                                   cbam_block(conv_channels),
                                   ])

        self.group_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                      groups=groups),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                      groups=groups),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1,
                      groups=groups),
            BaseConv(in_channels=conv_channels, out_channels=conv_channels, ksize=3, stride=1, groups=groups,
                     act=act)
        )

        self.se = se_block(conv_channels)
        self.sea = Sea_Attention(dim=conv_channels, key_dim=64, num_heads=4)
        self.ffn = FNN(dim=conv_channels)

    def forward(self, input):

        #  backbone
        out_features = self.backbone(input)

        if self.low_fea:
            low_feature=out_features["dark2"]
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        out0 = self.upsample(x0)
        out2 = self.downsample(x2)
        out = torch.cat([out2, x1, out0], 1)

        residual = out = self.conv1(out)

        for i in range(4):
            residual = out = self.atts[i](self.group_convs[i](out) + residual)

        out = self.se(self.group_conv2(out) + residual)

        out = self.sea(out)
        out = self.ffn(out)

        out0 = self.downsample(out)
        out1 = out
        out2 = self.upsample(out)

        return [out2, out1, out0]