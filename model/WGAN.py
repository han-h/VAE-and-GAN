import torch as t
import torch.nn as nn

class WGAN():
    def __init__(self,args):
        self.args=args

        # input_dim = 784
        # latent_dim = 10

        self.generator=nn.Sequential(
            nn.Linear(self.args.latent_dim,64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,self.args.input_dim),
            nn.Tanh()
        )

        self.discriminator=nn.Sequential(
            nn.Linear(self.args.input_dim,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1)
        )

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)


