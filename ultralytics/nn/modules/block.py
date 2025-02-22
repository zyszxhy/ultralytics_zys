# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, DCNv2, SpatialAttention
from .transformer import TransformerBlock

import math

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'C2_5', 'ASA', 'DP', 'DP_DCNv2',
           'FFB', 'HWT', 'Pass', 'C2f_SFE', 'CoordAtt', 'Morph_pre', 'Multi_resolution', 'MultiSpectral', 'DWT')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


# MLSDNet blocks

class C2_5(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2//2, 'input output channel numbers wrong!'
        self.dwconv1 = DWConv(c1//2, c1//2, 3, 2, 1, act='ReLU')
        self.dwconv2 = DWConv(c1//2, c1//2, 3, 2, 1, act='ReLU')
        self.CBR = Conv(c1//2, c1, 3, 1, None, 1, 1, 'ReLU')
        self.Ghost = GhostConv(c1//2, c1, 3, 1, 1, 'ReLU')
        self.BR = nn.Sequential(nn.BatchNorm2d(c1), nn.ReLU())

    def forward(self, x):
        _, c, _, _ = x.size()
        x1 = x[:, :c//2, :, :]
        x2 = x[:, c//2:, :, :]
        x1 = self.CBR(self.dwconv1(x1))
        x2 = self.BR(self.Ghost(self.dwconv2(x2)))
        x_out = torch.cat((x1, x2), 1)
        return self.channel_shuffle(x_out, 4)
    
    def channel_shuffle(self, x, groups=4):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # print(channels_per_group)
        # reshape
        # b, c, h, w =======>  b, g, c_per, h, w
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x
  
class DP(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.dconv1 = nn.Conv2d(c1//4, c1//4, 3, 1, autopad(3, p, 1), 1, 1)
        self.dconv2 = nn.Conv2d(c1//4, c1//4, 3, 1, autopad(3, p, 3), 3, 1)
        self.dconv3 = nn.Conv2d(c1//4, c1//4, 3, 1, autopad(3, p, 5), 5, 1)
        self.dconv4 = nn.Conv2d(c1//4, c1//4, 3, 1, autopad(3, p, 7), 7, 1)

        
    def forward(self, x):
        _, c, _, _ = x.size()
        x1 = x[:, :c//4, :, :]
        x2 = x[:, c//4:c//2, :, :]
        x3 = x[:, c//2:c//4*3, :, :]
        x4 = x[:, c//4*3:, :, :]
        
        x1 = self.dconv1(x1)
        x2 = self.dconv2(x2)
        x3 = self.dconv3(x3)
        x4 = self.dconv4(x4)

        return torch.cat((x1, x2, x3, x4), 1)

class FA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.xavgpool = nn.AdaptiveAvgPool2d([1,None])
        self.yavgpool = nn.AdaptiveAvgPool2d([None,1])
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(c1, c1, 1)
        self.conv2 = nn.Conv2d(c1, c1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, x):
        _, _, h, w = x.size()
        wv = self.yavgpool(x)
        wq = self.xavgpool(x)
        wa = torch.cat((wv[:,:,:,0], wq[:,:,0,:]), -1)
        wa = self.relu(wa)
        wv = torch.unsqueeze(wa[:, :, :h], -1)
        wq = torch.unsqueeze(wa[:, :, -w:], 2)
        wv = self.sigmoid1(self.conv1(wv))
        wq = self.sigmoid2(self.conv2(wq))

        return (x*wv)*wq

class CO(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.convv = nn.Conv2d(c1, c1//2, 1)
        self.convq = nn.Conv2d(c1, 1, 1)
        self.softmax = nn.Softmax(1)
        self.conv1 = nn.Conv2d(c1//2, c1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        bs, c, h, w = x.size()
        wv = self.convv(x)
        wq = self.convq(x)
        wv = wv.reshape(bs, c//2, -1)
        wq = wq.reshape(bs, -1, 1)
        wq = self.softmax(wq)

        wa = torch.unsqueeze(torch.matmul(wv, wq), -1)
        wa = self.sigmoid(self.conv1(wa))

        return wa*x
    
class ASA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.DP = DP(c1, c2)
        self.FA = FA(c1, c2)
        self.CO = CO(c1, c2)
        
    def forward(self, x):

        return self.CO(self.FA(self.DP(x)))

class DP_DCNv2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.dcn1 = DCNv2(c1//4, c1//4, 3, 1, 1, None, True)
        self.dcn2 = DCNv2(c1//4, c1//4, 3, 1, 1, None, True)
        self.dcn3 = DCNv2(c1//4, c1//4, 3, 1, 1, None, True)
        self.dcn4 = DCNv2(c1//4, c1//4, 3, 1, 1, None, True)

        
    def forward(self, x):
        _, c, _, _ = x.size()
        x1 = x[:, :c//4, :, :]
        x2 = x[:, c//4:c//2, :, :]
        x3 = x[:, c//2:c//4*3, :, :]
        x4 = x[:, c//4*3:, :, :]
        
        x1 = self.dcn1(x1)
        x2 = self.dcn2(x2)
        x3 = self.dcn3(x3)
        x4 = self.dcn4(x4)

        return torch.cat((x1, x2, x3, x4), 1)


# MFFN blocks

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ChannelAttention_2(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.avgpool(x)) + self.fc(self.maxpool(x)))


class FFB(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'
        self.conv1_1 = nn.Conv2d(c1*2, c1*2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(c1*2, c1*2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv2_1 = nn.Conv2d(c1*2, c1*2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(c1*2, c1*2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.ch_att = ChannelAttention_2(c1*2)
        self.sp_att = SpatialAttention(7)

        self.conv3 = nn.Conv2d(c1*2, c1, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        f = torch.cat(x, 1) # 2c*h*w
        f_r = self.conv1_2(self.relu1(self.conv1_1(f)))
        f_r = f + f_r
        f_r = self.conv2_2(self.relu2(self.conv2_1(f_r)))
        f_att = self.sp_att(self.ch_att(f))


        return self.conv3(f_r + f_att)

import numpy as np
import pywt
import cv2

class HWT(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == 3 and c2 == 4, 'input output channel numbers wrong!'

        
    def forward(self, x):
        b, c, h, w = x.size()
        # device = x.device
        x_hwt = torch.empty((b, 4, h//2, w//2), dtype=x.dtype).to(x.device)
        # x_hwt = torch.empty((b, 3, h, w), dtype=x.dtype).to(x.device)
        for i in range(b):
            img = np.transpose(x[i, :, :, :].cpu().numpy(), [1, 2, 0]).astype(np.float32) * 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            coeffs = pywt.dwt2(img, 'haar')
            cA, (cH, cV, cD) = coeffs
            # 将各个子图进行拼接，最后得到一张图
            AH = np.concatenate([cA, cH], axis=1)
            VD = np.concatenate([cV, cD], axis=1)
            # img_hwt = np.concatenate([AH, VD], axis=0)

            img_hwt = np.array([cA, cH, cV, cD]) / 255
            x_hwt[i, :, :, :] = torch.tensor(img_hwt)

            # c_hwt = F.interpolate(torch.tensor(np.array([cH, cV])).unsqueeze(0), size=[h, w])
            # img_hwt = torch.cat([torch.tensor(img).unsqueeze(0), c_hwt.squeeze(0)], 0) / 255
            # x_hwt[i, :, :, :] = img_hwt

            # cv2.imwrite(f'./img_hwt_{i}.jpg', img_hwt_1)
            # cv2.imwrite(f'./img_{i}.jpg', img)
            

        return x_hwt

class Pass(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2, 'input output channel numbers wrong!'

        
    def forward(self, x):
        return x
    

class C2f_SFE(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c_h = c2 // 2
        self.cv1 = Conv(c1, self.c_h, 3)
        self.cv1_1 = Conv(self.c_h, self.c_h, 1)

        self.acs1 = Conv(self.c_h//4, self.c_h//4, 1, 1, None, 1, 1)
        self.acs2 = Conv(self.c_h//4, self.c_h//4, 3, 1, None, 1, 6)
        self.acs3 = Conv(self.c_h//4, self.c_h//4, 3, 1, None, 1, 12)
        self.acs4 = Conv(self.c_h//4, self.c_h//4, 3, 1, None, 1, 18)
        # self.acs5 = nn.AdaptiveAvgPool2d(1)
        self.cv2 = Conv(self.c_h, c2, 1)

    def forward(self, x):
        x_in = self.cv1_1(self.cv1(x))

        x_1 = self.acs1(x_in[:, :self.c_h//4, :, :])
        x_2 = self.acs2(x_in[:, self.c_h//4:self.c_h//2, :, :])
        x_3 = self.acs3(x_in[:, self.c_h//2:self.c_h//4*3, :, :])
        x_4 = self.acs4(x_in[:, self.c_h//4*3:, :, :])

        x_in = torch.cat((x_1, x_2, x_3, x_4), 1) + x_in

        # x_5 = self.acs5(x_in)
        # x = torch.add(x_in, x_5)
        
        return self.cv2(x_in)

class FESA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        

    def forward(self, x):

        return


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
 

    def forward(self, x):
        identity = x
 
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



class Dilation_my(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        # assert c1==3 and c2==1, 'Dilation operate require in_ch=1 and out_ch=1!'
        self.ksize = k
        # self.dila_kernel = nn.Parameter(torch.randn(self.ksize, self.ksize), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x[:, :1, :, :]
        padding = (self.ksize - 1) // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        patches = x_pad.unfold(dimension=2, size=self.ksize, step=1)
        patches = patches.unfold(dimension=3, size=self.ksize, step=1)
        # dilate, _ = (patches + self.dila_kernel).reshape(b, c, h, w, -1).max(dim=-1)
        dilate, _ = (patches).reshape(b, c, h, w, -1).max(dim=-1)
        return dilate

class Erode_my(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        # assert c1==3 and c2==1, 'Dilation operate require in_ch=1 and out_ch=1!'
        self.ksize = k
        # self.erode_kernel = nn.Parameter(torch.randn(self.ksize, self.ksize), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x[:, :1, :, :]
        padding = (self.ksize - 1) // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        patches = x_pad.unfold(dimension=2, size=self.ksize, step=1)
        patches = patches.unfold(dimension=3, size=self.ksize, step=1)
        # eroded, _ = (patches - self.erode_kernel).reshape(b, 1, h, w, -1).min(dim=-1)
        eroded, _ = (patches).reshape(b, 1, h, w, -1).min(dim=-1)
        return eroded

class Morph_pre(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        assert c1==3 and c2==3, 'preprosess: in_ch=3 and out_ch=3.'
        self.openop_e = Erode_my(c1, 1, k)
        self.openop_d = Dilation_my(1, 1, k)

        self.closeop_d = Dilation_my(1, 1, k)
        self.closeop_e = Erode_my(1, 1, k)
        
        self.dilate = Dilation_my(1, 1, k)

    def forward(self, x):
        x_open = self.openop_d(self.openop_e(x))
        x_d = self.dilate(x_open)
        x_close = self.closeop_e(self.closeop_d(x_open))
        x_pre = x_d - x_close
        return  torch.cat([x[:, :2, :, :], x_pre], 1)



def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class Multi_resolution(nn.Module):
    def __init__(self, c1, c2):
        super(Multi_resolution, self).__init__()
        assert c1 == c2, 'c1 must be same as c2!'
        self.ch1_3 = math.floor(c1/3)
        self.ch2 = c1 - 2*self.ch1_3

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(self.ch1_3, self.ch1_3, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(self.ch2, self.ch2, 1, 1, 0, bias=True)
        self.fc3 = nn.Conv2d(self.ch1_3, self.ch1_3, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
        

    def forward(self, x):
        x_LL, x_HL, x_LH, _ = self.dwt_init(x)

        x1 = x_LL[:, :self.ch1_3, :, :]
        x2 = x_LH[:, self.ch1_3:self.ch2+self.ch1_3, :, :]
        x3 = x_HL[:, -self.ch1_3:, :, :]

        x1 = x[:, :self.ch1_3, :, :] * self.act(self.fc1(self.pool(x1)))
        x2 = x[:, self.ch1_3:self.ch2+self.ch1_3, :, :] * self.act(self.fc2(self.pool(x2)))
        x3 = x[:, -self.ch1_3:, :, :] * self.act(self.fc3(self.pool(x3)))

        return torch.cat([x1, x2, x3], 1)
    
    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self, c1, c2):
        super(DWT, self).__init__()
        assert c1 == c2, 'c1 must be same as c2!'
        

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.dwt_init(x[:, :1, :, :])
        x_HL = F.interpolate(x_HL, scale_factor=2)
        x_LH = F.interpolate(x_LH, scale_factor=2)

        return torch.cat([x[:, :1, :, :], x_HL, x_LH], 1)
    
    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectral(nn.Module):
    def __init__(self, channel, c2, dct_h, dct_w, reduction = 16, freq_sel_method = 'low4'):
        super(MultiSpectral, self).__init__()
        assert channel == c2, 'in_ch must be the same as out_ch!'
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter