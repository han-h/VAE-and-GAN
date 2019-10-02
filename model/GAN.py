import torch as t
import torch.nn as nn

class GAN():
    def __init__(self,args):
        self.args=args

        # generator_lr = 1e-3
        # discriminator_lr = 1e-4

        ngf=4
        ndf=4
        nz=self.args.latent_dim # nz is the length of the z input vector, latent_dim
        nc=1 # nc is the number of channels in the output image

        self.generator=nn.Sequential(
            # input is Z, going into a convolution
            # input is batch_size x nz x 1 x 1
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.discriminator=nn.Sequential(
            # input is batch_size x 1 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.generator=nn.Sequential(
        #     # input is batch_size x latent_dim
        #     nn.Linear(self.args.latent_dim,self.args.hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(self.args.hidden_dim,self.args.hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(self.args.hidden_dim,self.args.input_dim),
        #     nn.Tanh()
        # )
        #
        # self.discriminator=nn.Sequential(
        #     # input is batch_size x input_dim
        #     nn.Linear(self.args.input_dim,self.args.hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(self.args.hidden_dim,self.args.hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(self.args.hidden_dim,1)
        # )

        # if discriminator's last layer is sigmoid, use BCELoss
        # else, use BCEWithLogitsLoss
        self.bce=nn.BCELoss()

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)


