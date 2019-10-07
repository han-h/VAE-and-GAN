from model.DCGAN import DCGAN
from .BasicRunner import BasicRunner
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import logging

# lr = 1e-4
# epoch_num =20
class DCGANRunner(BasicRunner):
    def __init__(self,args):
        super(DCGANRunner,self).__init__(args)

    def _build_model(self):
        self.dcgan=DCGAN(self.args)
        self.dcgan.generator.cuda()
        self.dcgan.discriminator.cuda()

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.dcgan.generator.parameters(),lr=self.args.lr)
        self.discriminator_optimizer=Adam(self.dcgan.discriminator.parameters(),lr=self.args.lr)

    def _get_fixed_noise_for_evaluation(self):
        n=self.args.test_num
        self.Z=self.dcgan.sample(n*n).cuda().unsqueeze(-1).unsqueeze(-1)

    def eval(self,epoch):
        self.dcgan.generator.eval()
        self.dcgan.discriminator.eval()
        n=self.args.test_num
        digit_size=self.args.digit_size

        figure=np.zeros((digit_size*n, digit_size*n))

        # batch_size * digit_size * digit_size
        result=(self.dcgan.generator(self.Z).squeeze().detach().cpu().numpy()+1)/2
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/DCGAN/result"+str(epoch)+".png")
        plt.close('all')

    def _train_one_epoch(self,epoch):
        self.dcgan.generator.train()
        self.dcgan.discriminator.train()
        for batch_id,batch in enumerate(self.train_loader, 1):

            # batch_size * 1 * 28 * 28  grey image, scale to [-1,1)
            images=batch[0].cuda()*2-1

            # batch_size
            # label=batch[1]

            batch_size=images.shape[0]

            real_target=t.ones(batch_size).cuda()
            fake_target=t.zeros(batch_size).cuda()

            # train discriminator
            noise=self.dcgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)

            # batch_size * input_dim
            # detach to stop gradient back propagating to generator
            fake=self.dcgan.generator(noise).detach()
            dis_fake=self.dcgan.discriminator(fake).squeeze()
            dis_real=self.dcgan.discriminator(images).squeeze()

            discriminator_loss=self.dcgan.bce(dis_real,real_target)+self.dcgan.bce(dis_fake,fake_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.dcgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            # batch_size * input_dim
            fake=self.dcgan.generator(noise)

            dis_fake=self.dcgan.discriminator(fake).squeeze()
            generator_loss=self.dcgan.bce(dis_fake,real_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
