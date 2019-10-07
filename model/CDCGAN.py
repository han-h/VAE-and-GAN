import torch as t
import torch.nn as nn

class CDCGAN():
    def __init__(self,args):
        self.args=args

        nz=self.args.latent_dim+self.args.num_classes # nz is the length of the z input vector, latent_dim
        nc=1+self.args.num_classes # nc is the number of channels in the output image

        self.generator=nn.Sequential(
            # input is Z, going into a convolution
            # input is batch_size x nz x 1 x 1
            nn.ConvTranspose2d(nz,256,7,1,0),
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
            # input is batch_size x nc x 28 x 28
            nn.Conv2d(nc,128,4,2,1),
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


