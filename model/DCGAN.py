import torch as t
import torch.nn as nn

class DCGAN():
    def __init__(self,args):
        self.args=args

        self.generator=nn.Sequential(
            # input is Z, going into a convolution
            # input is batch_size x nz x 1 x 1
            nn.ConvTranspose2d(self.args.latent_dim,256,7,1,0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. 256 x 7 x 7
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # state size. 128 x 14 x 14
            nn.ConvTranspose2d(128,1,4,2,1),
            # state size. 1 x 28 x 28
            nn.Tanh()
        )

        self.discriminator=nn.Sequential(
            # input is batch_size x 1 x 28 x 28
            nn.Conv2d(1,128,4,2,1),
            nn.LeakyReLU(0.2),
            # state size. 128 x 15 x 15
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. 256 x 7 x 7
            nn.Conv2d(256,1,7,1,0),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

        # if discriminator's last layer is sigmoid, use BCELoss
        # else, use BCEWithLogitsLoss
        self.bce=nn.BCELoss()

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)


