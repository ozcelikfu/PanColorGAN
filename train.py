import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network, get_conv
from dataset import PanColorDataset, PanSRDataset
import torch.backends.cudnn as cudnn
from util import save_figure, visualize_tensor, avg_metric
import numpy as np
import matplotlib.pyplot as plt
from metrics import sCC
from metrics import ERGAS as ergas
from metrics import sam2 as sam

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
parser.add_argument('--nEpochs', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--testEveryNEpochs', type=int, default=2,
                    help='Test every n epochs')
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


f = open('results-{}/metrics.txt'.format(opt.savePath), 'w+')
F = open("results-{}/params.txt".format(opt.savePath), 'w')
# F.write(str(opt).split())
for i in str(opt).split():
    F.write(i + '\n')
F.close()
f.close()

## GPU Initialization

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
cudnn.benchmark = True

gpus = [gpu for gpu in range(opt.gpuSet)]
torch.cuda.set_device(gpus[0])

## Data Initialization
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



## Define Network
netG = define_G(opt.input_nc, opt.output_nc,
                    opt.ngf, 'batch','leakyrelu', opt.useDropout, opt.upConvType, opt.gtype, opt.blockType, opt.nblocks, gpus, n_downsampling=opt.ndowns)
netD = define_D(opt.input_nc + opt.output_nc,
                    opt.ndf, 'batch', not opt.lsgan, opt.nlayers, gpus)

## Define Losses
criterionGAN = GANLoss(use_lsgan=opt.lsgan)
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

## Define Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(
    opt.beta1, 0.999), weight_decay=opt.regTerm)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(
    opt.beta1, 0.999), weight_decay=opt.regTerm)

## Continue from a checkpoint

if opt.cont:
    ## Load Networks
    netG.load_state_dict(torch.load(
        'checkpoint/{}/netG_model_epoch_{}.pth'.format(opt.savePath, opt.contEpoch),
        map_location=lambda storage, loc: storage))
    netD.load_state_dict(torch.load(
        'checkpoint/{}/netD_model_epoch_{}.pth'.format(opt.savePath, opt.contEpoch),
        map_location=lambda storage, loc: storage))

    ## Load Optimizers
    optimizerG.load_state_dict(
        torch.load('{}checkpoint/{}/optimG_model_epoch_{}.pth'.format(opt.hddPath, opt.savePath, opt.contEpoch), map_location=lambda storage, loc: storage))
    for state in optimizerG.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    optimizerD.load_state_dict(
        torch.load('{}checkpoint/{}/optimD_model_epoch_{}.pth'.format(opt.hddPath, opt.savePath, opt.contEpoch), map_location=lambda storage, loc: storage))
    for state in optimizerD.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    opt.nEpochs += opt.contEpoch

## Loss Lists Initialization
losses_dict = {"batchId": {}, "d": {}, "g": {},
                   "grad": {}, "recon": {}, "adv": {}}

losses_g = []
losses_d = []

losses_recon = []
losses_adv = []

losses_recon_epoch = []
losses_adv_epoch = []

losses_g_epoch = []
losses_d_epoch = []

## Tensors Initialized
real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

## Switch to CUDA tensors
if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    netD.train()
    netG.train()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()


def train(epoch):
    netG.train()
    netD.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        real_a_cpu, real_b_cpu = batch[0].view(
            -1, opt.input_nc, 256, 256), batch[1].view(-1, opt.output_nc, 256, 256)
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        losses_dict["batchId"]["{}-{}".format(epoch, iteration)] = batch[2]

        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake= netD.forward(fake_ab.detach())

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)

        # Relativistic Average GAN Loss or Normal GAN Loss
        if opt.lossType == 'ragan':
            loss_d_fake = criterionGAN(
                pred_fake - torch.mean(pred_real), False)
            loss_d_real = criterionGAN(pred_real - torch.mean(pred_fake), True)
        elif opt.lossType == 'gan':
            loss_d_fake = criterionGAN(pred_fake, False)
            loss_d_real = criterionGAN(pred_real, True)

        del fake_ab
        del pred_fake
        del real_ab
        del pred_real
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        losses_d.append(loss_d.data.cpu().item())
        losses_dict["d"]["{}-{}".format(epoch, iteration)
                         ] = loss_d.data.cpu().item()

        del loss_d_fake
        del loss_d_real
        del loss_d

        # Parameter optimizing
        nn.utils.clip_grad_norm_(netD.parameters(), 0.3)
        nn.utils.clip_grad_norm_(netG.parameters(), 0.3)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator

        if opt.lossType == 'ragan':

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = netD.forward(fake_ab)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = netD.forward(real_ab)

            loss_g_gan = (criterionGAN(pred_real - torch.mean(pred_fake), False) +
                          criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            
            del fake_ab
            del pred_fake
            del real_ab
            del pred_real
        elif opt.lossType == 'gan':
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = netD.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)
            del fake_ab
            del pred_fake
        # Second, G(A) = B

        
        loss_g_l1 = (criterionL1(fake_b, real_b) ) * opt.lamb
        loss_g = (loss_g_gan * opt.weightG) + loss_g_l1
        loss_g.backward()
        losses_g.append(loss_g.data.cpu().item())
        losses_adv.append(loss_g_gan.data.cpu().item())
        losses_recon.append(loss_g_l1.data.cpu().item() * 1 / opt.lamb)

        losses_dict["g"]["{}-{}".format(epoch, iteration)
                         ] = loss_g.data.cpu().item()
        losses_dict["adv"]["{}-{}".format(epoch,
                                          iteration)] = loss_g_gan.data.cpu().item()
        losses_dict["recon"]["{}-{}".format(epoch, iteration)] = loss_g_l1.data.cpu().item() * 1 / opt.lamb

        del loss_g
        del loss_g_l1
        del loss_g_gan

        # Parameter optimizing
        # nn.utils.clip_grad_norm(netD.parameters(), 0.3)
        # nn.utils.clip_grad_norm(netG.parameters(), 0.3)
        optimizerG.step()

        #torch.cuda.synchronize()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}, Loss_Recon: {:.7f} ".format(
            epoch, iteration, len(training_data_loader), losses_d[-1], losses_g[-1], losses_recon[-1]))

        del fake_b



    len_loader = len(training_data_loader)

    losses_d_epoch.append(
        np.mean(losses_d[-len_loader:]))
    losses_g_epoch.append(
        np.mean(losses_g[-len_loader:]))
    losses_adv_epoch.append(
        np.mean(losses_adv[-len_loader:]))
    losses_recon_epoch.append(
        np.mean(losses_recon[-len_loader:]))


def test(epoch):
    netG.eval()
    if not os.path.isdir("results-{}/test".format(opt.savePath)):
        os.mkdir("results-{}/test".format(opt.savePath))
    avg_sCC = []
    avg_sam = []
    avg_ergas = []
    for it, batch in enumerate(testing_data_loader):
        input,target= batch[0],batch[1]

        input = input.view(-1, opt.input_nc, 256, 256)

        target = target.cuda().view(-1, opt.output_nc, 256, 256)
        prediction = netG(input.view(-1, opt.input_nc, 256, 256).cuda())

        sam_val = avg_metric(target, prediction, sam)
        avg_sam.append(sam_val)
        sCC_val = avg_metric(target, prediction, sCC)
        avg_sCC.append(sCC_val)
        ergas_val = avg_metric(target, prediction, ergas)
        avg_ergas.append(ergas_val)
        
        del input
        del target
        del prediction
        del sCC_val
        del sam_val
        del ergas_val
    res = "===> Avg. SAM: {:.4f} , Avg sCC: {:.4f}, Avg ERGAS: {:.4f}   epoch: {} \n".format(
        np.mean(avg_psnr), np.std(avg_psnr), np.mean(avg_sam), np.mean(avg_sCC), np.mean(avg_ergas), epoch)
    f = open('results-{}/metrics.txt'.format(opt.savePath), 'a')
    f.write(res)
    print(res)


def checkpoint(epoch):
    if opt.hddPath is not None:
        path = os.path.join(opt.hddPath, "checkpoint")
    else:
        path = "checkpoint"
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, opt.savePath)):
        os.mkdir(os.path.join(path, opt.savePath))
    net_g_model_out_path = "{}/{}/netG_model_epoch_{}.pth".format(path,
                                                                  opt.savePath, epoch)
    net_d_model_out_path = "{}/{}/netD_model_epoch_{}.pth".format(path,
                                                                  opt.savePath, epoch)
    optim_d_model_out_path = "{}/{}/optimD_model_epoch_{}.pth".format(path,
                                                                      opt.savePath, epoch)
    optim_g_model_out_path = "{}/{}/optimG_model_epoch_{}.pth".format(path,
                                                                      opt.savePath, epoch)
    torch.save(netG.state_dict(), net_g_model_out_path)
    torch.save(netD.state_dict(), net_d_model_out_path)
    torch.save(optimizerD.state_dict(), optim_d_model_out_path)
    torch.save(optimizerG.state_dict(), optim_g_model_out_path)

    print("Checkpoint saved to {}".format(path + opt.savePath))


for epoch in range(opt.contEpoch, opt.nEpochs + 1):
    train(epoch)
    if epoch % opt.testEveryNEpochs == 0:
        checkpoint(epoch)
        print('===> Testing')
        test(epoch)
    np.savez("results-{}/losses".format(opt.savePath), np.array(losses_dict))

f.close()