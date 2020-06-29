import torch
import torch.nn as nn
import numpy as np
import functools
import torch.nn.functional as F
from torch.nn import init

def weights_init(m):
    init_style = "kaiming"
    classname = m.__class__.__name__
    if init_style == "normal":
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    if init_style == "kaiming":
        scale = 0.1
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            m.bias.data.fill_(0)

def get_conv(x, kernel):

    out = F.conv2d(x, kernel)
    return out

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class InterpolatedConv2D(nn.Module):
    """
            Args:
                    mode: Upsampling method [nearest|bilinear]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale_factor=2, mode='nearest'):
        super(InterpolatedConv2D, self).__init__()
        self.up = Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):

        y = self.up(x)
        y = self.conv(y)
        return y

    def __repr__(self):

        return 'InterpolatedConv2D'

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_activation_layer(activation_type):
    if activation_type == 'relu':
        activation_layer = nn.ReLU
    elif activation_type == 'leakyrelu':
        activation_layer = nn.LeakyReLU(0.2, True)
    else:
        print('activation layer [%s] is not found' % activation_type)
    return activation_layer


def define_G(input_nc, output_nc, ngf, norm='batch', activation='leakyrelu', use_dropout=False, upConvType='ConvT', net_type='fusenet', blockType="SE", n_blocks=9, gpu_ids=[], n_downsampling=3):
    netG = None
    use_gpu = True
    norm_layer = get_norm_layer(norm_type=norm)
    activation_layer = get_activation_layer(activation_type=activation)
    if use_gpu:
        assert(torch.cuda.is_available())
    

    netG = UnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, activation_layer=activation_layer,
                                use_dropout=use_dropout, blockType=blockType, n_blocks=n_blocks, upConvType=upConvType,
                                gpu_ids=gpu_ids, n_downsampling=3)


    else:
        raise NotImplementedError
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, n_layers=3, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = NLayerDiscriminator(
        input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)