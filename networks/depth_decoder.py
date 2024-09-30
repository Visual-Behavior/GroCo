# Code is adapted from Monodepth2:
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
import torch.nn.functional as F


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc,
        scales=range(4),
        use_skips=True,
    ):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1

        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("outconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            # self.convs[("outconv", s)].conv.bias.data.fill_(1.0)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        self.ground_attn = LightNeck(self.num_ch_dec, out_ch=1)
        self.ground_attn.convfinal.bias.data.fill_(-3)

        self.scale_param = nn.Parameter(torch.tensor(3.5).view(1, 1, 1, 1))

        self.act = nn.Softplus()

    def forward(self, input_features):
        outputs = {}

        # decoder
        x = input_features[-1]
        features = []
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            features.append(x)

            if i in self.scales:
                outputs["depth"] = self.act(self.convs[("outconv", i)](x)) * torch.exp(self.scale_param).to(x.device)

        attn = self.ground_attn(features)
        outputs["ground_attn"] = torch.sigmoid(attn)

        return outputs


class LightNeck(nn.Module):
    def __init__(self, in_chs, out_ch=1):
        super().__init__()
        hidden_ch = in_chs[0]
        self.convfinal = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.conv0 = nn.Conv2d(in_chs[4], hidden_ch, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(in_chs[3], hidden_ch, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_chs[2], hidden_ch, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_chs[1], hidden_ch, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_chs[0], hidden_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, inputs):
        x0, x1, x2, x3, x4 = inputs

        x0 = self.conv0(x0)
        x0 = F.interpolate(x0, size=[x4.shape[2], x4.shape[3]], mode="bilinear", align_corners=True)

        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=[x4.shape[2], x4.shape[3]], mode="bilinear", align_corners=True)

        x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=[x4.shape[2], x4.shape[3]], mode="bilinear", align_corners=True)

        x3 = self.conv3(x3)
        x3 = F.interpolate(x3, size=[x4.shape[2], x4.shape[3]], mode="bilinear", align_corners=True)

        x4 = self.conv4(x4)
        x = x0 + x1 + x2 + x3 + x4
        return self.convfinal(x)


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")
