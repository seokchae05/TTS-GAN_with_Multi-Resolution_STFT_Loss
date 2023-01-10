import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class vibrationData(Dataset):
    def __init__(self, root_path, transform=None, n_cls=2):
        self.transform = transform
        self.class_list = os.listdir(os.path.join(root_path, 'converted'))
        self.dataset = []
        self.n_cls = n_cls
        for folder in self.class_list:
            if folder =='.DS_Store':
                continue
            #1 . signal data load
            x = []
            y = []

            x_txt = os.path.join(root_path, 'signal', folder, 'x_data.txt')
            y_txt = os.path.join(root_path, 'signal', folder, 'y_data.txt')

            with open(x_txt, 'r') as f:
                lines = f.readlines()
                x = []
                for line in lines:
                    x.append(line.strip())
            with open(y_txt, 'r') as f:
                lines = f.readlines()
                y = []
                for line in lines:
                    y.append(line.strip())


            #2. img data load
            wavlet_imgs = []
            corr_imgs = []

            img_dir = os.path.join(root_path, 'converted', folder)
            imgs = os.listdir(img_dir)

            dataset = []
            for idx in range(0, 5121024, 1024):
                
                wavlet = os.path.join(img_dir, f'wavelet_{idx+1024}.png')
                corr = os.path.join(img_dir, f'correlation_{idx+1024}.png')
                
                if not os.path.isfile(wavlet):
                    print(f'No such file {wavlet}')
                    exit()
                if not os.path.isfile(corr):
                    print(f'No such file {corr}')
                    exit()
                
                x_sample = x[idx:idx+1024]
                y_sample = y[idx:idx+1024]
                
                data ={
                    'x' : x_sample,
                    'y' : y_sample,
                    'wavlet' : wavlet,
                    'corr' : corr,
                    'cls' : folder
                }
                
                dataset.append(data)
            
            self.dataset +=dataset    
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        x = data['x']
        y = data['y']
        cls = int(data['cls'])
        wavlet = data['wavlet']
        corr = data['corr']
        
        signal = [x, y]
        signal = torch.tensor(signal, dtype=float)
        cls_onehot = torch.eye(self.n_cls)
        
        wavlet_img = Image.open(wavlet)
        corr_img = Image.open(corr)
        if self.transform:
            wavlet_img = self.transform(wavlet_img)
            corr_img = self.transform(corr_img)
        
        return signal, wavlet_img, corr_img, cls_onehot[cls]