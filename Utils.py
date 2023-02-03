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
    def __init__(self, root_path, cls_name= '7'):
        self.cls_name = cls_name
        self.class_list = os.listdir(os.path.join(root_path, 'signal'))
        self.dataset = []
        for folder in self.class_list:
            if folder =='.DS_Store':
                continue
            #1 . signal data load
            x = []
            y = []

            x_txt = os.path.join(root_path, 'signal', cls_name, 'x_data.txt')
            y_txt = os.path.join(root_path, 'signal', cls_name, 'y_data.txt')

            with open(x_txt, 'r') as f:
                lines = f.readlines()
                x = []
                for line in lines:
                    x.append(float(line.strip()))
            with open(y_txt, 'r') as f:
                lines = f.readlines()
                y = []
                for line in lines:
                    y.append(float(line.strip()))

            dataset = []
            for idx in range(0, 5121024, 1024):
                
                x_sample = x[idx:idx+1024]
                y_sample = y[idx:idx+1024]
                
                data ={
                    'x' : x_sample,
                    'y' : y_sample
                }
                
                dataset.append(data)
            
            self.dataset +=dataset    
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        x = data['x']
        y = data['y']
        
        signal = [[x], [y]]
        signal = torch.tensor(signal, dtype=float)

        return signal, self.cls_name