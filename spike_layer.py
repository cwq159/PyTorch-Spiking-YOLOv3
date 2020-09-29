import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from spike_tensor import SpikeTensor


def generate_spike_mem_potential(out_s, mem_potential, Vthr, reset_mode):
    """
    out_s: is a Tensor of the output of different timesteps [timesteps, *sizes]
    mem_potential: is a placeholder Tensor with [*sizes]
    """
    spikes = []
    for t in range(out_s.size(0)):
        mem_potential += out_s[t]
        spike = (mem_potential >= Vthr).float()
        if reset_mode == 'zero':
            mem_potential *= (1 - spike)
        elif reset_mode == 'subtraction':
            mem_potential -= spike * Vthr
        else:
            raise NotImplementedError
        spikes.append(spike)
    return spikes


class SpikeReLU(nn.Module):
    def __init__(self, quantize=False):
        super().__init__()
        self.max_val = 1
        self.quantize = quantize
        self.activation_bitwidth = None

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            return x
        else:
            x_ = F.relu(x)
            if self.quantize:
                bits = self.activation_bitwidth
                if self.training:
                    xv = x_.view(-1)
                    # max_val=torch.kthvalue(xv,int(0.99*xv.size(0)))[0]
                    max_val = xv.max()
                    if self.max_val is 1:
                        self.max_val = max_val.detach()
                    else:
                        self.max_val = (self.max_val * 0.95 + max_val * 0.05).detach()
                rst = torch.clamp(torch.round(x_ / self.max_val * 2 ** bits), 0, 2 ** bits) * (self.max_val / 2 ** bits)
                return rst
            else:
                return x_


class SpikeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', bn=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'
        self.bn = bn

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = F.conv2d(x.data, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups)
            if self.bn is not None:
                out = self.bn(out)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.bn is not None:
                out = self.bn(out)
            return out


class SpikeConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, output_padding, groups, bias, dilation, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                     self.groups, self.dilation)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                     self.groups, self.dilation)
            return out


class SpikeAvgPool2d(nn.Module):
    """
    substitute all 1 Depthwise Convolution for AvgPooling
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            stride = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.register_buffer('Vthr', torch.ones(1) * np.prod(self.kernel_size))
        self.reset_mode = 'subtraction'

    def forward(self, x):
        Vthr = self.Vthr
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        if isinstance(x, SpikeTensor):
            if stride is None:
                stride = kernel_size
            weight = torch.ones([x.chw[0], 1, *_pair(kernel_size)]).to(x.data.device)
            out = F.conv2d(x.data, weight, None, _pair(stride), _pair(padding), 1,
                           groups=x.chw[0])
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, x.scale_factor)
            return out
        else:
            out = F.avg_pool2d(x, kernel_size, stride, padding)
            return out


class SpikeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, last_layer=False):
        super().__init__(in_features, out_features, bias)
        self.last_layer = last_layer
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_features))
        self.reset_mode = 'subtraction'

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1)
            out = F.linear(x.data, self.weight, self.bias)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.linear(x, self.weight, self.bias)
            if not self.last_layer:
                out = F.relu(out)
            return out
