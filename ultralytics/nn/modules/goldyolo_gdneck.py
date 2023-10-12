# 2023.09.18-Changed for Neck implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

TORCH_VERSION = torch.__version__
def _get_norm():
    if TORCH_VERSION == 'parrots':
        # from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        # SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
        raise ValueError('Fuck you, CV!')
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

import numpy as np
import warnings
import inspect


class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class BottleRep(nn.Module):
    
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0
    
    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_C3(nn.Module):
    '''Standard convolution in BepC3-Block'''
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ConvWrapper(nn.Module):
    '''Wrapper for normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def forward(self, x):
        return self.block(x)

class BepC3(nn.Module):
    '''Beer-mug RepC3 Block'''
    
    def __init__(self, in_channels, out_channels, n=1, e=0.5, concat=True,
                 block=RepVGGBlock):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())
        
        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)
    
    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                    *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                      range(n - 1))) if n > 1 else None
    
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

class AdvPoolFusion(nn.Module):
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)

class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channel_list[0], out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        
        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)

def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in ['zero', 'reflect', 'replicate']:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        if padding_type == 'zero':
            layer = nn.ZeroPad2d(*args, **kwargs, **cfg_)
        elif padding_type == 'reflect':
            layer = nn.ReflectionPad2d(*args, **kwargs, **cfg_)
        elif padding_type == 'replicate':
            layer = nn.ReplicationPad2d(*args, **kwargs, **cfg_)

    return layer

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in ['Conv1d', 'Conv2d', 'Conv3d', 'Conv']:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        if layer_type == 'Conv1d':
            layer = nn.Conv1d(*args, **kwargs, **cfg_)
        elif layer_type == 'Conv2d':
            layer = nn.Conv2d(*args, **kwargs, **cfg_)
        elif layer_type == 'Conv3d':
            layer = nn.Conv3d(*args, **kwargs, **cfg_)
        elif layer_type == 'Conv':
            layer = nn.Conv2d(*args, **kwargs, **cfg_)

    return layer

def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'

class SyncBatchNorm(SyncBatchNorm_):

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)

def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in ['BN', 'BN1d', 'BN2d', 'BN3d', 'SyncBN', 'GN', 'LN', 'IN', 'IN1d', 'IN2d', 'IN3d']:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        if layer_type == 'BN':
            layer = nn.BatchNorm2d(num_features, **cfg_)
        elif layer_type == 'BN1d':
            layer = nn.BatchNorm1d(num_features, **cfg_)
        elif layer_type == 'BN2d':
            layer = nn.BatchNorm2d(num_features, **cfg_)
        elif layer_type == 'BN3d':
            layer = nn.BatchNorm3d(num_features, **cfg_)
        elif layer_type == 'SyncBN':
            layer = SyncBatchNorm(num_features, **cfg_)
        elif layer_type == 'LN':
            layer = nn.LayerNorm(num_features, **cfg_)
        elif layer_type == 'IN':
            layer = nn.InstanceNorm2d(num_features, **cfg_)
        elif layer_type == 'IN1d':
            layer = nn.InstanceNorm1d(num_features, **cfg_)
        elif layer_type == 'IN2d':
            layer = nn.InstanceNorm2d(num_features, **cfg_)
        elif layer_type == 'IN3d':
            layer = nn.InstanceNorm3d(num_features, **cfg_)
        
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = nn.GroupNorm(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad
    
    abbr = infer_abbr(type(layer))
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    return name, layer

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            if act_cfg_['type'] == 'ReLU':
                self.activate = nn.ReLU(act_cfg_)
            elif act_cfg_['type'] == 'LeakyReLU':
                self.activate = nn.LeakyReLU(act_cfg_)
            elif act_cfg_['type'] == 'PReLU':
                self.activate = nn.PReLU(act_cfg_)
            elif act_cfg_['type'] == 'RReLU':
                self.activate = nn.RReLU(act_cfg_)
            elif act_cfg_['type'] == 'ReLU6':
                self.activate = nn.ReLU6(act_cfg_)
            elif act_cfg_['type'] == 'ELU':
                self.activate = nn.ELU(act_cfg_)
            elif act_cfg_['type'] == 'Sigmoid':
                self.activate = nn.Sigmoid(act_cfg_)
            elif act_cfg_['type'] == 'Tanh':
                self.activate = nn.Tanh(act_cfg_)
            elif act_cfg_['type'] == 'GELU':
                self.activate = nn.GELU(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool

class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        
        if not global_inp:
            global_inp = inp
        
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
    
    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg, act_layer=act_layer))
    
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x




class GDNeck(nn.Module):
    def __init__(
            self,
            channel_in_list=[128, 256, 512, 1024],
            channel_out = [256, 512, 1024]
    ):
        super().__init__()
        
        assert channel_in_list is not None        
        self.fusion_in = sum(channel_in_list)

        if self.fusion_in == 480:   # n
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 96
            self.fuse_block_num = 3
            self.trans_channels = [64, 32, 64, 128]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 352
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
            
        elif self.fusion_in == 960: # s
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 128
            self.fuse_block_num = 3
            self.trans_channels = [128, 64, 128, 256]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 704
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2


        elif self.fusion_in == 1184:    # m
            block = BottleRep
            rep = 'BepC3'

            self.embed_dim_p = 192
            self.fuse_block_num = 3
            self.trans_channels = [192, 96, 192, 384]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 7
            self.csp_e = float(2) / 3,
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 1056
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
        

        elif self.fusion_in == 1408:    # l
            block = BottleRep
            rep = 'BepC3'
            
            self.embed_dim_p = 192
            self.fuse_block_num = 3
            self.trans_channels = [256, 128, 256, 512]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 12
            self.csp_e = float(1) / 2,
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 3
            self.embed_dim_n = 1408
            self.key_dim = 8
            self.num_heads = 8
            self.mlp_ratios = 1
            self.attn_ratios = 2
        
        inj_block = InjectionMultiSum_Auto_pool
        
        # low
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(self.fusion_in, self.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(self.embed_dim_p, self.embed_dim_p) for _ in range(self.fuse_block_num)],
                Conv(self.embed_dim_p, sum(self.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channel_in_list[3],  # C5 1024
                out_channels=channel_in_list[1],  # C3 256
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channel_in_list[2], channel_in_list[2]],  # C4 512, 512
                out_channels=channel_in_list[1],  # C3 256
        )
        self.Inject_p4 = inj_block(channel_in_list[1], channel_in_list[1], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C3
        
        if rep == 'RepBlock':
            self.Rep_p4 = RepBlock(
                    in_channels=channel_in_list[1],  # C3 256
                    out_channels=channel_in_list[1],  # C3 256
                    n=self.num_repeats,
                    block=block
            )
        elif rep == 'BepC3':
            self.Rep_p4 = BepC3(
                    in_channels=channel_in_list[1],  # C3 256
                    out_channels=channel_in_list[1],  # C3 256
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channel_in_list[1],  # C3 256
                out_channels=channel_in_list[0],  # C2 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channel_in_list[1], channel_in_list[1]],  # C3 C3 512, 256
                out_channels=channel_in_list[0],  # C2 256
        )
        self.Inject_p3 = inj_block(channel_in_list[0], channel_in_list[0], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C2
        
        if rep == 'RepBlock':
            self.Rep_p3 = RepBlock(
                in_channels=channel_in_list[0],  # C2 128
                out_channels=channel_in_list[0],  # C2 128
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_p3 = BepC3(
                in_channels=channel_in_list[0],  # C2 128
                out_channels=channel_in_list[0],  # C2 128
                n=self.num_repeats,
                e=self.csp_e,
                block=block
            )

        
        # high
        self.high_FAM = PyramidPoolAgg(stride=self.c2t_stride, pool_mode=self.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=self.depths,
                embedding_dim=self.embed_dim_n,
                key_dim=self.key_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratios,
                attn_ratio=self.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=self.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(self.embed_dim_n, sum(self.trans_channels[2:4]), 1, 1, 0)
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = inj_block(channel_in_list[1], channel_in_list[1], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C3
        
        if rep == 'RepBlock':
            self.Rep_n4 = RepBlock(
                in_channels=channel_in_list[0] + channel_in_list[0],  # C2 + C2 128 + 128
                out_channels=channel_in_list[1],  # C3 256
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_n4 = BepC3(
                    in_channels=channel_in_list[0] + channel_in_list[0],  # C2 + C2 128 + 128
                    out_channels=channel_in_list[1],  # C3 256
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = inj_block(channel_in_list[2], channel_in_list[2], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C4
        
        if rep == 'RepBlock':
            self.Rep_n5 = RepBlock(
                in_channels=channel_in_list[1] + channel_in_list[1],  # C3 + C3 256 + 256
                out_channels=channel_in_list[2],  # C4 512
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_n5 = BepC3(
                    in_channels=channel_in_list[1] + channel_in_list[1],  # C3 + C3 256 + 256
                    out_channels=channel_in_list[2],  # C4 512
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
    
    def forward(self, input):
        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs

class DevideOutputs_gd(nn.Module):
    def __init__(self, level_num):
        super(DevideOutputs_gd, self).__init__()
        self.n = level_num

    def forward(self, x):
        return x[self.n]


class GD_Multimodal(nn.Module):
    def __init__(
            self,
            channel_in_list=[128, 256, 512, 1024, 256, 512, 1024],
            channel_out = [256, 512, 1024]
    ):
        super().__init__()
        
        assert channel_in_list is not None        
        self.fusion_in = sum(channel_in_list[:4])

        if self.fusion_in == 480:   # n
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 192
            self.fuse_block_num = 3
            self.trans_channels = [128, 64, 128, 256]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 352
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
            
        elif self.fusion_in == 960: # s
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 256
            self.fuse_block_num = 3
            self.trans_channels = [256, 128, 256, 512]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 704
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2

        elif self.fusion_in == 1184:    # m
            block = BottleRep
            rep = 'BepC3'

            self.embed_dim_p = 384
            self.fuse_block_num = 3
            self.trans_channels = [384, 192, 384, 512]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 7
            self.csp_e = float(2) / 3
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 1088
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
       
        elif self.fusion_in == 1408:    # l
            block = BottleRep
            rep = 'BepC3'
            
            self.embed_dim_p = 384
            self.fuse_block_num = 3
            self.trans_channels = [512, 256, 512, 512]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 12
            self.csp_e = float(1) / 2
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 3
            self.embed_dim_n = 1280
            self.key_dim = 8
            self.num_heads = 8
            self.mlp_ratios = 1
            self.attn_ratios = 2
        
        inj_block = InjectionMultiSum_Auto_pool
        
        # low
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(self.fusion_in, self.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(self.embed_dim_p, self.embed_dim_p) for _ in range(self.fuse_block_num)],
                Conv(self.embed_dim_p, sum(self.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        # self.reduce_layer_c5 = SimConv(
        #         in_channels=channel_in_list[3],  # C5 1024
        #         out_channels=channel_in_list[1],  # C3 256
        #         kernel_size=1,
        #         stride=1
        # )
        # self.LAF_p4 = SimFusion_3in(
        #         in_channel_list=[channel_in_list[2], channel_in_list[2]],  # C4 512, 512
        #         out_channels=channel_in_list[1],  # C3 256
        # )
        self.Inject_p4 = inj_block(channel_in_list[2], channel_in_list[2], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C4
        
        if rep == 'RepBlock':
            self.Rep_p4 = RepBlock(
                    in_channels=channel_in_list[2],  # C4 512
                    out_channels=channel_in_list[2],  # C4 512
                    n=self.num_repeats,
                    block=block
            )
        elif rep == 'BepC3':
            self.Rep_p4 = BepC3(
                    in_channels=channel_in_list[2],  # C3 512
                    out_channels=channel_in_list[2],  # C3 512
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
        # self.reduce_layer_p4 = SimConv(
        #         in_channels=channel_in_list[1],  # C3 256
        #         out_channels=channel_in_list[0],  # C2 128
        #         kernel_size=1,
        #         stride=1
        # )
        # self.LAF_p3 = SimFusion_3in(
        #         in_channel_list=[channel_in_list[1], channel_in_list[1]],  # C3 C3 512, 256
        #         out_channels=channel_in_list[0],  # C2 256
        # )
        self.Inject_p3 = inj_block(channel_in_list[1], channel_in_list[1], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C3
        
        if rep == 'RepBlock':
            self.Rep_p3 = RepBlock(
                in_channels=channel_in_list[1],  # C3 256
                out_channels=channel_in_list[1],  # C3 256
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_p3 = BepC3(
                in_channels=channel_in_list[1],  # C3 256
                out_channels=channel_in_list[1],  # C3 256
                n=self.num_repeats,
                e=self.csp_e,
                block=block
            )

        
        # high
        self.high_FAM = PyramidPoolAgg(stride=self.c2t_stride, pool_mode=self.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=self.depths,
                embedding_dim=self.embed_dim_n,
                key_dim=self.key_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratios,
                attn_ratio=self.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=self.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(self.embed_dim_n, sum(self.trans_channels[2:4]), 1, 1, 0)
        
        # self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = inj_block(channel_in_list[2], channel_in_list[2], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C4
        
        if rep == 'RepBlock':
            self.Rep_n4 = RepBlock(
                in_channels=channel_in_list[1] + channel_in_list[1],  # C3 + C3 256 + 256
                out_channels=channel_in_list[2],  # C4 512
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_n4 = BepC3(
                    in_channels=channel_in_list[1] + channel_in_list[1],  # C3 + C3 256 + 256
                    out_channels=channel_in_list[2],  # C4 512
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
        # self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = inj_block(channel_in_list[3], channel_in_list[3], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C5
        
        if rep == 'RepBlock':
            self.Rep_n5 = RepBlock(
                in_channels=channel_in_list[3],  # C4 + C4 512 + 512
                out_channels=channel_in_list[3],  # C5 1024
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_n5 = BepC3(
                    in_channels=channel_in_list[3],  # C5
                    out_channels=channel_in_list[3],  # C5 1024
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
    
    def forward(self, input):
        (c2_ir, c3_ir, c4_ir, c5_ir, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM([c2_ir, c3_ir, c4_ir, c5_ir])
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        # c5_half = self.reduce_layer_c5(c5)
        # p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(c4, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        # p4_half = self.reduce_layer_p4(p4)
        # p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(c3, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5_ir])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        # n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(p4, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        # n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5 = self.Inject_n5(c5, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs


class GDNeck_P3(nn.Module):
    def __init__(
            self,
            channel_in_list=[128, 256, 512, 1024],
            channel_out = [256, 512, 1024]
    ):
        super().__init__()
        
        assert channel_in_list is not None        
        self.fusion_in = sum(channel_in_list)

        if self.fusion_in == 480:   # n
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 96
            self.fuse_block_num = 3
            self.trans_channels = [64, 32, 64, 128]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 352
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
            
        elif self.fusion_in == 960: # s
            block = RepVGGBlock
            rep = 'RepBlock'

            self.embed_dim_p = 128
            self.fuse_block_num = 3
            self.trans_channels = [128, 64, 128, 256]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 4
            self.csp_e = None
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 704
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2


        elif self.fusion_in == 1184:    # m
            block = BottleRep
            rep = 'BepC3'

            self.embed_dim_p = 192
            self.fuse_block_num = 3
            self.trans_channels = [192, 96, 192, 384]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 7
            self.csp_e = float(2) / 3,
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 2
            self.embed_dim_n = 1056
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
        

        elif self.fusion_in == 1408:    # l
            block = BottleRep
            rep = 'BepC3'
            
            self.embed_dim_p = 192
            self.fuse_block_num = 3
            self.trans_channels = [256, 128, 256, 512]
            self.norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.num_repeats = 12
            self.csp_e = float(1) / 2,
            self.c2t_stride = 2
            self.pool_mode = 'torch'
            self.drop_path_rate = 0.1
            self.depths = 3
            self.embed_dim_n = 1408
            self.key_dim = 8
            self.num_heads = 8
            self.mlp_ratios = 1
            self.attn_ratios = 2
        
        inj_block = InjectionMultiSum_Auto_pool
        
        # low
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(self.fusion_in, self.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(self.embed_dim_p, self.embed_dim_p) for _ in range(self.fuse_block_num)],
                Conv(self.embed_dim_p, sum(self.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channel_in_list[3],  # C5 1024
                out_channels=channel_in_list[1],  # C3 256
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channel_in_list[2], channel_in_list[2]],  # C4 512, 512
                out_channels=channel_in_list[1],  # C3 256
        )
        self.Inject_p4 = inj_block(channel_in_list[1], channel_in_list[1], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C3
        
        if rep == 'RepBlock':
            self.Rep_p4 = RepBlock(
                    in_channels=channel_in_list[1],  # C3 256
                    out_channels=channel_in_list[1],  # C3 256
                    n=self.num_repeats,
                    block=block
            )
        elif rep == 'BepC3':
            self.Rep_p4 = BepC3(
                    in_channels=channel_in_list[1],  # C3 256
                    out_channels=channel_in_list[1],  # C3 256
                    n=self.num_repeats,
                    e=self.csp_e,
                    block=block
            )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channel_in_list[1],  # C3 256
                out_channels=channel_in_list[0],  # C2 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channel_in_list[1], channel_in_list[1]],  # C3 C3 512, 256
                out_channels=channel_in_list[0],  # C2 256
        )
        self.Inject_p3 = inj_block(channel_in_list[0], channel_in_list[0], norm_cfg=self.norm_cfg,
                                   activations=nn.ReLU6)    # C2
        
        if rep == 'RepBlock':
            self.Rep_p3 = RepBlock(
                in_channels=channel_in_list[0],  # C2 128
                out_channels=channel_in_list[0],  # C2 128
                n=self.num_repeats,
                block=block
            )
        elif rep == 'BepC3':
            self.Rep_p3 = BepC3(
                in_channels=channel_in_list[0],  # C2 128
                out_channels=channel_in_list[0],  # C2 128
                n=self.num_repeats,
                e=self.csp_e,
                block=block
            )

    
    def forward(self, input):
        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        return p3
