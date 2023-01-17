import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import DistributedOptimizer
##prarallel
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from Net import NetD, Encoder, Decoder, weights_init
import Utils
from tqdm import tqdm
import logging
from loss import l2_loss
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
device ='cuda'

class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, extralayers):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(isize, nz, nc, ngf, ngpu, extralayers)
        self.decoder = Decoder(isize, nz, nc, ngf, ngpu, extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        return gen_imag, latent_i

if __name__ =='__main__':
    ######################################################################
    # 0. Logger Setting
    ###################################################################### 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('test')
    writer = SummaryWriter(f'runs/{nowDatetime}')
    logger = logging.getLogger(__name__)
    fileHandler = logging.FileHandler('./train.log')
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.INFO)
    
    ######################################################################
    # 1. Data Loading
    ######################################################################
    
    root_path = os.path.join(os.getcwd(), 'data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[512,512]),
        transforms.Normalize(mean=(0.5,), std=(0.5, ))
    ])
    dataset = Utils.vibrationData(root_path=root_path, transform=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    batch_size =4
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ######################################################################
    # 2. Model Generating
    ######################################################################
    
    D = NetD(isize=512, nc=4, ngf=64, ngpu=1, extralayers=0).to(device)
    G = NetG(isize=512, nz=100, nc=4, ngf=64, ngpu=1, extralayers=0).to(device)

    l_G = nn.L1Loss()
    l_D = nn.BCELoss()

    optim_D = optim.Adam(D.parameters(), lr=0.0002 ,betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.0002 ,betas=(0.5, 0.999))
    
    ######################################################################
    # 3. Start Trainning
    ######################################################################
    n_epochs= 200

    for epoch in tqdm(range(n_epochs)):
        D_loss_acc = 0.0
        G_loss_acc = 0.0
        D.train()
        G.train()

        for i, (signal, images, corr_img, c) in enumerate(train_data_loader):
            
            images = images.to(device)
                    
            rand_z = torch.rand(images.size())
            rand_z = rand_z.to(device)
            
            real_label = torch.ones (size=(batch_size,), dtype=torch.float32, device=device)
            fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32, device=device)
            
            
            ## forward G and D
            fake_img, _ = G(rand_z)
            
            pred_real, _ = D(images)
            pred_fake, _ = D(fake_img.detach())
            
            
            ## calculate err
            optim_G.zero_grad()
            optim_D.zero_grad()
            
            err_g = l_G(fake_img, images)
            err_d_real = l_D(pred_real, real_label)
            err_d_fake = l_D(pred_fake, fake_label)
            err_d = (err_d_fake+err_d_real)*0.5
            
            ## backward G and D
            err_g.backward(retain_graph=True)
            optim_G.step()
            
            err_d.backward()
            optim_D.step()
            if err_d.item() < 1e-5:
                D.apply(weights_init)
        
        if (epoch + 1) % 10 == 0:
            D.eval()
            G.eval()
            
            n_show = 10
            with torch.no_grad():
                real = images[:n_show]
            
                rand_z = torch.rand(images.size())
                rand_z = rand_z.to(device)
                
                gener = G(rand_z)
                real = real.reshape(n_show, 4, 512, 512)
                writer.add_image(
                    "fake",
                    vutils.make_grid(gener.data[:n_show], normalize=True),
                    epoch
                )
                
                writer.add_image(
                    "real",
                    vutils.make_grid(real.data[:n_show], normalize=True),
                    epoch
                )
            os.makedirs(f"./weight/{nowDatetime}/{epoch}")
            torch.save({'D_state_dict': D.state_dict(),
                    'G_state_dict': G.state_dict()
                    },f"./weight/{nowDatetime}/{epoch}/model_{epoch}.tar")
                
                
            writer.add_scalar("Enc_Gen_loss",G_loss_acc/(len(dataset)/batch_size), epoch)
            writer.add_scalar("dis_loss", D_loss_acc/(len(dataset)/batch_size), epoch)
            logger.info(f'loss_D : {D_loss_acc/(len(dataset)/batch_size)}, loss_EG : {G_loss_acc/(len(dataset)/batch_size)} at epoch-{epoch}')