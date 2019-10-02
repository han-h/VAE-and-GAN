import torch as t
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args

        # input_dim = 784
        # hidden_dim = 256
        # latent_dim = 2 or 10

        # reduce dimension
        self.encoder=nn.Linear(self.args.input_dim,self.args.hidden_dim)

        # generate \mu
        self.mean_generator=nn.Linear(self.args.hidden_dim,self.args.latent_dim)
        # generate log \sigma ^ 2
        self.log_var_generator=nn.Linear(self.args.hidden_dim,self.args.latent_dim)

        self.decoder=nn.Sequential(
            nn.Linear(self.args.latent_dim,self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim,self.args.input_dim),
            nn.Sigmoid()
        )

        self.bce=nn.BCELoss(reduction='none')

    # batch: batch_size * input_dim
    def forward(self,batch):
        hidden=t.relu(self.encoder(batch))
        # \mu
        mean=self.mean_generator(hidden)
        # log \sigma ^ 2
        log_var=self.log_var_generator(hidden)
        # \epsilon
        Z=t.randn(batch.shape[0],self.args.latent_dim).cuda()
        # \mu + \epsilon * \sigma
        Z=mean+Z*t.exp(log_var/2)
        output=self.decoder(Z)

        reconstruction_loss=t.sum(self.bce(output,batch),dim=-1)

        kl_loss=t.sum(0.5*(-log_var+mean*mean+t.exp(log_var)-1),dim=-1)

        vae_loss=t.mean(reconstruction_loss+kl_loss)

        return output,vae_loss

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)



