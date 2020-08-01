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

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(
                #    real_tensor, requires_grad=False)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                        (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(
                #    fake_tensor, requires_grad=False)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())

    
## Special UNet Designed for Pansharpening
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True),
                 use_dropout=False, blockType="Resnet", n_blocks=6, upConvType='ConvT', gpu_ids=[], n_downsampling=3):
        assert (n_blocks >= 0)
        super(UnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.blockType = blockType
        if upConvType == 'ConvT':
            upLayer = nn.ConvTranspose2d
        elif upConvType == 'ResizeConv':
            upLayer = InterpolatedConv2D
        else:
            raise NotImplementedError

        color1 = [nn.Conv2d(self.input_nc-1, ngf, kernel_size=3, padding=1),
                  norm_layer(ngf, affine=True),
                  activation_layer,
                  nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
                  norm_layer(ngf, affine=True),
                  activation_layer,
                  ]

        color2 = [nn.Conv2d(ngf, ngf * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * 2, affine=True),
                      activation_layer, ]
        color3 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3,
                            stride=2, padding=1),
                  norm_layer(ngf * 4, affine=True),
                  activation_layer, ]

        model1 = [nn.Conv2d(1, ngf, kernel_size=3, padding=1),
                    norm_layer(ngf, affine=True),
                    activation_layer,
                    nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
                    norm_layer(ngf, affine=True),
                    activation_layer,
                    ]

        model2 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3,
                            stride=2, padding=1),
                  norm_layer(ngf * 2, affine=True),
                  activation_layer, ]

        model3 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3,
                            stride=2, padding=1),
                  norm_layer(ngf * 4, affine=True),
                  activation_layer, ]

        model_resnet = []



        for i in range(n_blocks):
            model_resnet += [ResnetBlock(ngf * 8, 'zero',
                                  norm_layer=norm_layer, use_dropout=use_dropout)]

        model4 = [nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, padding=1),
                  norm_layer(ngf * 4, affine=True),
                  activation_layer, ]

        model5 = [upLayer(ngf * 8, ngf * 4,
                          kernel_size=3, stride=2,
                          padding=1, output_padding=1) if upConvType == 'ConvT' else upLayer(ngf * 8, ngf * 4,
                                                                                             kernel_size=3,
                                                                                             stride=1, padding=1,
                                                                                             scale_factor=2),
                  norm_layer(ngf * 4, affine=True),
                  activation_layer, ]

        model6 = [nn.Conv2d(ngf*4, ngf * 2, kernel_size=3, padding=1),
                  norm_layer(ngf*2, affine=True),
                  activation_layer, ]
        model7 = [upLayer(ngf * 4, ngf*2,
                          kernel_size=3, stride=2,
                          padding=1, output_padding=1) if upConvType == 'ConvT' else upLayer(ngf * 4, ngf*2,
                                                                                             kernel_size=3,
                                                                                             stride=1, padding=1,
                                                                                             scale_factor=2),
                  norm_layer(ngf*2, affine=True),
                  activation_layer, ]
        model8 = [nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1),
                  norm_layer(ngf, affine=True),
                  activation_layer, ]


        model9 = [nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1),
                      norm_layer(ngf, affine=True),
                      activation_layer,
                      nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
                      ]
        out_model = [nn.Tanh()]

        self.color1 = nn.Sequential(*color1)
        self.color2 = nn.Sequential(*color2)
        self.color3 = nn.Sequential(*color3)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_resnet = nn.Sequential(*model_resnet)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.model9 = nn.Sequential(*model9)
        self.out_model = nn.Sequential(*out_model)

    def forward(self, input):

        input_pan, input_ms = input[:, 4, :,
                              :].unsqueeze(1), input[:, :4, :, :]

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            m1 = nn.parallel.data_parallel(self.model1, input_pan, self.gpu_ids)
            c1 = nn.parallel.data_parallel(self.color1, input_ms, self.gpu_ids)
            mc1 = torch.cat((m1, c1), dim=1)
            m2 = nn.parallel.data_parallel(self.model2, mc1, self.gpu_ids)
            c2 = nn.parallel.data_parallel(self.color2, c1, self.gpu_ids)
            mc2 = torch.cat((m2, c2), dim=1)
            m3 = nn.parallel.data_parallel(self.model3, mc2, self.gpu_ids)
            c3 = nn.parallel.data_parallel(self.color3, c2, self.gpu_ids)
            mc3 = torch.cat((m3, c3), dim=1)
            res = nn.parallel.data_parallel(self.model_resnet, mc3, self.gpu_ids)
            m4 = nn.parallel.data_parallel(self.model4, res, self.gpu_ids)
            m34 = torch.cat((m3, m4), dim=1)
            m5 = nn.parallel.data_parallel(self.model5, m34, self.gpu_ids)
            m6 = nn.parallel.data_parallel(self.model6, m5, self.gpu_ids)
            m26 = torch.cat((m2, m6), dim=1)
            m7 = nn.parallel.data_parallel(self.model7, m26, self.gpu_ids)
            m8 = nn.parallel.data_parallel(self.model8, m7, self.gpu_ids)
            m18 = torch.cat((m1, m8), dim=1)
            m9 = nn.parallel.data_parallel(self.model9, m18, self.gpu_ids)
            out = nn.parallel.data_parallel(self.out_model, m9, self.gpu_ids)
            return out
        else:
            m1 = self.model1(input_pan)
            c1 = self.color1(input_ms)
            mc1 = torch.cat((m1, c1), dim=1)
            m2 = self.model2(mc1)
            c2 = self.color2(c1)
            mc2 = torch.cat((m2, c2), dim=1)
            m3 = self.model3(mc2)
            c3 = self.color3(c2)
            mc3 = torch.cat((m3, c3), dim=1)
            res = self.model_resnet(mc3)
            m4 = self.model4(res)
            m34 = torch.cat((m3, m4), dim=1)
            m5 = self.model5(m34)
            m6 = self.model6(m5)
            m26 = torch.cat((m2, m6), dim=1)
            m7 = self.model7(m26)
            m8 = self.model8(m7)
            m18 = torch.cat((m1, m8), dim=1)
            m9 = self.model9(m18)
            out = self.out_model(m9)
            return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        assert(padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.LeakyReLU(0.2, True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Defines the PatchGAN discriminator.

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), norm_layer(ndf * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                      padding=padw), norm_layer(ndf * nf_mult,
                                                affine=True), nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        #self.gpu_ids = [2,3]
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
