# Generator synthetic Running and cls7 data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os
    
class Syn_Vibration_Dataset(Dataset):
    def __init__(self, 
                 cls0_model_path = os.path.join(os.getcwd(), 'logs', 'class_0','Model','checkpoint'),
                 cls7_model_path = os.path.join(os.getcwd(), 'logs', 'class_7','Model','checkpoint'),
                 sample_size = 100
                 ):
        
        self.sample_size = sample_size
        
        #Generate cls0 Data
        cls0_gen_net = Generator(seq_len=1024, channels=2, latent_dim=100)
        cls0_ckp = torch.load(cls0_model_path)
        cls0_gen_net.load_state_dict(cls0_ckp['gen_state_dict'])
        
        #Generate cls7 Data
        cls7_gen_net = Generator(seq_len=1024, channels=2, latent_dim=100)
        cls7_ckp = torch.load(cls7_model_path)
        cls7_gen_net.load_state_dict(cls7_ckp['gen_state_dict'])
        
        
        #generate synthetic cls0 data label is 0
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_cls0 = cls0_gen_net(z)
        self.syn_cls0 = self.syn_cls0.detach().numpy()
        self.cls0_label = np.zeros(len(self.syn_cls0))
        
        #generate synthetic cls7 data label is 1
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_cls7 = cls7_gen_net(z)
        self.syn_cls7 = self.syn_cls7.detach().numpy()
        self.cls7_label = np.ones(len(self.syn_cls7))
        
        self.combined_train_data = np.concatenate((self.syn_cls0, self.syn_cls7), axis=0)
        self.combined_train_label = np.concatenate((self.cls0_label, self.cls7_label), axis=0)
        self.combined_train_label = self.combined_train_label.reshape(self.combined_train_label.shape[0], 1)
        
        print(self.combined_train_data.shape)
        print(self.combined_train_label.shape)
        
        
    def __len__(self):
        return self.sample_size * 2
    
    def __getitem__(self, idx):
        return self.combined_train_data[idx], self.combined_train_label[idx]