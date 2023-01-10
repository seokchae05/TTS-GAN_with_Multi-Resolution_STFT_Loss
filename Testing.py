import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import Utils
import os

from Net import Encoder, Generator, Discriminator, D_loss, EG_loss, initialize_weights

from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    n_epochs = 500
    l_rate = 2e-5

    print('Model Creating...')
    E = Encoder(isize=512, nz=512, nc=4, ndf=64, ngpu=0).to(device)
    G = Generator(n_cls=8, isize=512, nz=512, nc=4, ngf=64, ngpu=0).to(device)
    D = Discriminator(n_cls=8, isize=512, nz=512, nc=4, ngf=64, ngpu=0).to(device)

    E.apply(initialize_weights)
    G.apply(initialize_weights)
    D.apply(initialize_weights)

    optimizer_DEG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()) + list(D.parameters()), 
                                    lr=l_rate, betas=(0.5, 0.999), weight_decay=1e-5)

    print(f'Data loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[512,512]),
        transforms.Normalize(mean=(0.5,), std=(0.5, ))
    ])

    root_path = os.path.join(os.getcwd(), 'data')
    dataset = Utils.vibrationData(root_path=root_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


    print('Start learning...')
    
    for epoch in range(n_epochs):
        D_loss_acc = 0.
        EG_loss_acc = 0.
        D.train()
        E.train()
        G.train()
        
        for i, (signal, images, corr_img, c) in tqdm(enumerate(data_loader)):
            
            images = images.to(device)
            
            rand_z = 2 * torch.rand(images.size(0), 512, 2, 2) - 1
            rand_z = rand_z.to(device)
            
            c = c.to(device)
            #compute G(z, c) and E(X)
            
            Gz = G(rand_z, c)
            EX = E(images)
            
            # print(f'image : {images.size()}, Gz : {Gz.size()}')
            # print(f'rand_z : {rand_z.size()}, Ex : {EX.size()}')
            DG = D(Gz, rand_z, c)
            DE = D(images, EX, c)
            
            #compute losses
            loss_D = D_loss(DG, DE)
            loss_EG = EG_loss(DG, DE)
            D_loss_acc += loss_D.item()
            EG_loss_acc += loss_EG.item()
            
            loss_DEG= loss_D +loss_EG
            
            optimizer_DEG.zero_grad()
            loss_DEG.backward()
            optimizer_DEG.step()
            
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Avg_Loss_D: {:.4f}, Avg_Loss_EG: {:.4f}'
                        .format(epoch + 1, n_epochs, D_loss_acc / i, EG_loss_acc / i))
                n_show = 10