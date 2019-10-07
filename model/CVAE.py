import torch as t
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args

        # reduce dimension
        self.encoder=nn.Linear(self.args.input_dim,self.args.hidden_dim)

        # every class has a special mean value objective
        # input should be one-hot
        self.condition=nn.Linear(self.args.num_classes,self.args.latent_dim)

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
    # label: batch_size * num_classes
    def forward(self,batch,label):
        hidden=t.relu(self.encoder(batch))
        condition_mean=self.condition(label)
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

        kl_loss=t.sum(0.5*(-log_var+(mean-condition_mean)*(mean-condition_mean)+t.exp(log_var)-1),dim=-1)

        cvae_loss=t.mean(reconstruction_loss+kl_loss)

        return output,cvae_loss

    def sample(self,num):
        # sample
        return t.randn(num,self.args.latent_dim)



