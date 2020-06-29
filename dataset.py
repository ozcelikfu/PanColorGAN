from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
from skimage.transform import resize
import skimage
from scipy import ndimage
from util import rgb2gray, scale_range

class PanColorDataset(Dataset):
    def __init__(self, mode='train', dataset='pleiades', random_downsampling=False):
        self.mode = mode
        self.dataset = dataset
        self.random_downsampling = random_downsampling

        ## arbitrary file counts for now
        self.train_samples = 30120
        self.test_samples = 5775

    def __getitem__(self, index):        
        filename=""
        if self.mode == "train":
            filename = index+1
        elif self.mode == "test":
            filename = str(int(index+1) + self.train_samples)

        ms_orig = np.load('./dataset/new_npy/msimage_{}.npy'.format(filename))
        pan_orig = np.load('./dataset/new_npy/panimage_{}.npy'.format(filename))

        gray_ms = self.rgb2gray(ms_orig)

        ms_orig = ms_orig.astype(np.float32)
        gray_ms = gray_ms.astype(np.float32)

        ms_norm = np.array([self.scale_range(i, -1, 1) for i in ms_orig.transpose((2,0,1))])
        gray_ms_norm = self.scale_range(gray_ms, -1, 1)

        ms_down = [resize(i,(64,64), 3) for i in ms_norm]
        ms_up = [resize(i, (256, 256), 3) for i in ms_down]

        ms_up = np.clip(ms_up,-1.0,1.0)

        inp = np.concatenate((ms_up,np.expand_dims(gray_ms_norm,axis=0)),axis=0)
        out=ms_norm

        del ms_orig
        del pan_orig
        del ms_down
        del gray_ms

        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.train_samples
        elif self.mode == "test":
            return self.test_samples

class PanSRDataset(Dataset):
    def __init__(self, mode='train', dataset='pleiades'):

    def __getitem__(self, index):        
        return 0

    def __len__(self):
        return 0


class PansharpeningDataset(Dataset):
    def __init__(self, mode='train', dataset='pleiades'):

    def __getitem__(self, index):        
        return 0

    def __len__(self):
        return 0