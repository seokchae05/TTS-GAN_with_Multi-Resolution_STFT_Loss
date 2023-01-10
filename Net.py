import torch
import torch.nn as nn
import torch.nn.parallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        """_summary_

        Args:
            isize (int): input image size
            nz (int): size of the latent z vector
            nc (int): input image channels
            ndf (int): _description_
            ngpu (int): number of GPUs to use
            n_extra_layers (int, optional): number of layers on gen and disc. Defaults to 0.
            add_final_conv (bool, optional): _description_. Defaults to True.
        """
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 3, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 3, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 3, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        with torch.autograd.set_detect_anomaly(True):
            if self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output
    
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        """_summary_

        Args:
            isize (int): input image size
            nz (int): size of the latent z vector
            nc (int): input image channels
            ngf (_type_): _description_
            ngpu (int): number of GPUs to use
            n_extra_layers (int, optional): number of layers on gen and disc. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        cngf, tisize = ngf // 2, 4
        
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 3, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        with torch.autograd.set_detect_anomaly(True):
            if self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output
    
    
    
class Generator(nn.Module):
    def __init__(self, n_cls, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Generator, self).__init__()
        self.n_cls = n_cls
        self.pre_z = nn.Sequential(
            nn.Linear(2048, 3*nz),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pre_c = nn.Sequential(
            nn.Linear(n_cls, 1*nz),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layers = Decoder(isize=isize, nz=nz, nc=nc, ngf=ngf, ngpu=ngpu, n_extra_layers=n_extra_layers)
        
    def forward(self, z, c):
        with torch.autograd.set_detect_anomaly(True):
            batch_size = z.size(0)
            
            z = z.view([batch_size, -1]).to(device)
            z = self.pre_z(z)
            z = z.view([batch_size, -1, 2, 2]).to(device)
            c = self.pre_c(c)
            c = c.view([batch_size, -1, 2, 2]).to(device)
            
            zc = torch.cat([z,c], dim=1)

            return self.layers(zc)


class Discriminator(nn.Module):
    def __init__(self,n_cls=2, isize=512, nz=100, nc=4, ngf=64, ngpu=0):
        super(Discriminator, self).__init__()
        
        self.pre_x = Encoder(isize=512, nz=100, nc=4, ndf=64, ngpu=0)
        # 2x2x100
        
        self.pre_z = nn.Sequential(
            nn.Linear(2048, 3*nz),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # flatten(2x2x75)
        
        
        self.pre_c = nn.Sequential(
            nn.Linear(n_cls, 1*nz),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # flatten(2x2x25)
        
        self.layers = nn.Sequential(
            nn.Linear(2448, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
        
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

        
    def forward(self, x, z, c):
        with torch.autograd.set_detect_anomaly(True):
            batch_size = x.size(0)
            
            x = self.pre_x(x)
            z = z.view(batch_size, -1).to(device)

            z = self.pre_z(z)
            
            
            
            z = z.view([batch_size, -1, 2, 2]).to(device)
            c = self.pre_c(c)
            c = c.view([batch_size, -1, 2, 2]).to(device)
            
            zc = torch.cat([z,c], dim=1)
            xzc = torch.cat([x,zc], dim=1)
            
            # 2x2x200
            xzc = xzc.view(batch_size, -1).to(device)
            
            result = self.layers(xzc)
            
        
            return result

def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
    return -torch.mean(loss)


def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data,0)
    elif classname.find('Linear') != -1:
        nn.init.constant_(model.bias.data, 0)