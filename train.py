import argparse
import os
from math import log10
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network, get_conv, edge_loss
from dataset import GrayMSDataset, GrayMSPreprocessedDataset
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from util import save_figure
from torch.utils.data.sampler import RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from metrics import sCC
from metrics import ERGAS as ergas
from metrics import sam2 as sam
import pytorch_ssim

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Training settings
parser = argparse.ArgumentParser(description='PanColorGAN-PyTorch-implementation')
#parser.add_argument('--datasetSource', required=True, help='path of dataset')
#parser.add_argument('--datasetTarget', required=False, help='path of dataset')
parser.add_argument('--dataPath', help='path of data')
parser.add_argument('--dataset', type=str, default='pleiades')
parser.add_argument('--savePath', required=True, help='path of save')
parser.add_argument('--batchSize', type=int, default=16,
                    help='training batch size')
parser.add_argument('--testBatchSize', type=int,
                    default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--model', type=str, default='PanColorGAN')
parser.add_argument("--useRD", action='store_true')
parser.add_argument('--input_nc', type=int, default=5,
                    help='input image channels')
parser.add_argument('--output_nc', type=int, default=4,
                    help='output image channels')
parser.add_argument('--ngf', type=int, default=64,
                    help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64,
                    help='discriminator filters in first conv layer')
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--nblocks', type=int, default=6)
parser.add_argument('--ndowns', type=int, default=2)
parser.add_argument('--gtype', type=str, default='fusenet')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Learning Rate. Default=0.0002')
parser.add_argument('--adjustLR', action='store_true',
                    help='decrease LR by %1 after each epoch 100th')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
parser.add_argument('--lamb', type=float, default=1,
                    help='weight on L1 term in objective')
parser.add_argument('--weightG', type=float, default=1)
parser.add_argument("--useL2", action='store_true')
parser.add_argument('--useDropout', action='store_true')
parser.add_argument("--lsgan", action='store_true', default=False,
                    help='use lsgan loss in D')
parser.add_argument('--upConvType', type=str, default='ConvT',
                    help='type of upsampling conv, default is ConvTranspose2d')
parser.add_argument('--blockType', type=str, default='SE',
                    help='Type of Generator block (Resnet, RRDB)')
parser.add_argument('--lossType', type=str, default='ragan',
                    help='Type of GAN Loss (Normal, ragan(relativistic average gan)')
parser.add_argument('--cont', action='store_true', help='continue from')
parser.add_argument('--checkpointPath')
parser.add_argument('--contEpoch', type=int,
                    help='contiune from where we left', default=0)
parser.add_argument("--hddPath", type=str, default='./')
parser.add_argument("--regTerm", type=float, default=0.0)
parser.add_argument('--gpuSet', type=int, default=1)
opt = parser.parse_args()

print(opt)

if not os.path.exists("results-{}".format(opt.savePath)):
    os.mkdir("results-{}".format(opt.savePath))


f = open('results-{}/psnr.txt'.format(opt.savePath), 'w+')
F = open("results-{}/params.txt".format(opt.savePath), 'w')
# F.write(str(opt).split())
for i in str(opt).split():
    F.write(i + '\n')
F.close()
f.close()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
cudnn.benchmark = True

gpus = [gpu for gpu in range(opt.gpuSet)]
torch.cuda.set_device(gpus[0])

if opt.model == 'PanColorGAN':
    train_set = PanColorDataset(mode='train', dataset=opt.dataset, random_downsampling=opt.useRD)
    test_set = PanColorDataset(mode='test', dataset=opt.dataset)
elif opt.model == 'PanSRGAN':
    train_set = PanSRDataset(mode='train', dataset=opt.dataset)
    test_set = PanSRDataset(mode='test', dataset=opt.dataset)

training_data_loader = DataLoader(
    dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(
    dataset=test_set,num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)