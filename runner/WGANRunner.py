from model.WGAN import WGAN
from .BasicRunner import BasicRunner
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import logging


# lr = 1e-4
# epoch_num = 10
class WGANRunner(BasicRunner):
    def __init__(self,args):
        super(WGANRunner,self).__init__(args)

    def _build_model(self):
        self.wgan=WGAN(self.args)
        self.wgan.generator.cuda()
        self.wgan.discriminator.cuda()

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.wgan.generator.parameters(),lr=self.args.lr,betas=(0.5,0.999))
        self.discriminator_optimizer=Adam(self.wgan.discriminator.parameters(),lr=self.args.lr,betas=(0.5,0.999))

    def _get_fixed_noise_for_evaluation(self):
        n=self.args.test_num
        self.Z=self.wgan.sample(n*n).cuda().unsqueeze(-1).unsqueeze(-1)

    def eval(self,epoch):
        self.wgan.generator.eval()
        self.wgan.discriminator.eval()
        n=self.args.test_num
        digit_size=self.args.digit_size

        figure=np.zeros((digit_size*n, digit_size*n))

        # batch_size * digit_size * digit_size
        result=(self.wgan.generator(self.Z).squeeze().detach().cpu().numpy()+1)/2
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/WGAN/result"+str(epoch)+".png")
        plt.close('all')

    def _train_one_epoch(self,epoch):
        self.wgan.generator.train()
        self.wgan.discriminator.train()
        for batch_id,batch in enumerate(self.train_loader, 1):

            # batch_size * 1 * 28 * 28  grey image, scale to [-1,1]
            images=batch[0].cuda()*2-1

            # batch_size
            # label=batch[1]

            batch_size=images.shape[0]

            eps=t.rand(batch_size,1,1,1).cuda()
            noise=self.wgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            fake=self.wgan.generator(noise)
            inter=eps*images+(-eps+1)*fake


            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # batch_size * input_dim
            grad=t.autograd.grad(self.wgan.discriminator(inter),inter,t.ones(batch_size,1,1,1).cuda(),create_graph=True)[0]

            grad_penalty=self.args.lamda*((grad.norm(2,dim=1)-1)**2).mean()

            discriminator_loss=t.mean(self.wgan.discriminator(images)-self.wgan.discriminator(fake))+grad_penalty

            # train discriminator
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.wgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            fake=self.wgan.generator(noise)
            generator_loss=t.mean(self.wgan.discriminator(fake))

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
