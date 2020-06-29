from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
from skimage.transform import resize
import skimage
from scipy import ndimage
from util import rgb2gray, scale_range

class PanColorDataset(Dataset):
    def __init__(self):

    def __getitem__(self, index):        
        return 0

    def __len__(self):
        return 0

class PanSRDataset(Dataset):
    def __init__(self):

    def __getitem__(self, index):        
        return 0

    def __len__(self):
        return 0


class PansharpeningDataset(Dataset):
    def __init__(self):

    def __getitem__(self, index):        
        return 0

    def __len__(self):
        return 0