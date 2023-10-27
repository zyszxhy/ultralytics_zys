# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, C2_5, ASA, DP, DP_DCNv2, FFB, HWT, Pass, C2f_SFE, CoordAtt, Morph_pre,
                    Multi_resolution, MultiSpectral, DWT)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, Add, Modal_norm, DCNv2, GAMAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .head_m import RTDETRDecoder_m
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

from .superyolo_models import (DeepLab)
from .goldyolo_gdneck import (GDNeck, DevideOutputs_gd, GD_Multimodal, GDNeck_P3)
from .lsknet import LSKNet
from .fpn import FPN
from .pe_yolo import PENet
from .pp_lcnet import (SELayer, DepSepConv)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'Add',
           'RTDETRDecoder_m', 'Modal_norm', 'DeepLab', 'C2_5', 'ASA', 'GDNeck', 'DevideOutputs_gd', 'GD_Multimodal',
           'DP', 'GDNeck_P3', 'DCNv2', 'DP_DCNv2', 'GAMAttention', 'FFB', 'HWT', 'Pass', 'LSKNet', 'FPN', 'PENet',
           'C2f_SFE', 'CoordAtt', 'Morph_pre', 'Multi_resolution', 'MultiSpectral', 'DWT', 'SELayer', 'DepSepConv')
