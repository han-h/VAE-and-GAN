import torch as t
import torch.nn as nn

class WGAN():
    def __init__(self,args):
        self.args=args

        # input_dim = 784
        # latent_dim = 10

        self.generator=nn.Sequential(
            nn.Linear(self.args.latent_dim,32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,self.args.input_dim),
            nn.Sigmoid()
        )

        self.discriminator=nn.Sequential(
            nn.Linear(self.args.input_dim,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,1)
        )

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)


