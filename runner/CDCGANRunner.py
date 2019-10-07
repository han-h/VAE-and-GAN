from model.CDCGAN import CDCGAN
from .BasicRunner import BasicRunner
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import logging

# lr=2e-4
# epoch_num = 10
class CDCGANRunner(BasicRunner):
    def __init__(self,args):
        super(CDCGANRunner,self).__init__(args)

    def _build_model(self):
        self.cdcgan=CDCGAN(self.args)
        self.cdcgan.generator.cuda()
        self.cdcgan.discriminator.cuda()

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.cdcgan.generator.parameters(),lr=self.args.lr,betas=(0.5,0.999))
        self.discriminator_optimizer=Adam(self.cdcgan.discriminator.parameters(),lr=self.args.lr,betas=(0.5,0.999))

    def _get_fixed_noise_for_evaluation(self):
        n=self.args.test_num
        self.Z=self.cdcgan.sample(n*n).cuda().unsqueeze(-1).unsqueeze(-1)
        self.objective=t.zeros(n*n,self.args.num_classes).cuda()
        for i in range(n):
            self.objective[i*n:(i+1)*n,i]=1

    def eval(self,epoch):
        self.cdcgan.generator.eval()
        self.cdcgan.discriminator.eval()
        n=self.args.test_num
        digit_size=self.args.digit_size
        figure=np.zeros((digit_size*n, digit_size*n))


        fused=t.cat([self.Z,self.objective.unsqueeze(-1).unsqueeze(-1)],dim=1)
        # batch_size * digit_size * digit_size
        result=(self.cdcgan.generator(fused).squeeze().detach().cpu().numpy()+1)/2
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/CDCGAN/result"+str(epoch)+".png")
        plt.close('all')

    def _train_one_epoch(self,epoch):
        self.cdcgan.generator.train()
        self.cdcgan.discriminator.train()
        for batch_id,batch in enumerate(self.train_loader, 1):

            # batch_size * 1 * 28 * 28  grey image, scale to [-1,1)
            images=batch[0]*2-1
            batch_size=images.shape[0]

            # batch_size
            label=batch[1]
            # batch_size * num_classes
            one_hot=t.zeros(batch_size,self.args.num_classes).scatter_(1,label.unsqueeze(-1),1).cuda()

            # batch_size * num_classes * 28 * 28
            one_hot_fused=one_hot.unsqueeze(-1).unsqueeze(-1)*t.ones(batch_size,self.args.num_classes,28,28).cuda()



            real_target=t.ones(batch_size).cuda()
            fake_target=t.zeros(batch_size).cuda()

            # train discriminator
            noise=self.cdcgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)

            # batch_size * nz * 1 * 1
            fused=t.cat([noise,one_hot.unsqueeze(-1).unsqueeze(-1)],dim=1)

            # batch_size * 1 * 28 * 28
            # detach to stop gradient back propagating to generator
            fake=self.cdcgan.generator(fused).detach()
            dis_fake=self.cdcgan.discriminator(t.cat([fake,one_hot_fused],dim=1)).squeeze()
            dis_real=self.cdcgan.discriminator(t.cat([images.cuda(),one_hot_fused],dim=1)).squeeze()

            discriminator_loss=self.cdcgan.bce(dis_real,real_target)+self.cdcgan.bce(dis_fake,fake_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.cdcgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            # batch_size * nz * 1 * 1
            fused=t.cat([noise,one_hot.unsqueeze(-1).unsqueeze(-1)],dim=1)

            # batch_size * 1 * 28 * 28
            fake=self.cdcgan.generator(fused)

            dis_fake=self.cdcgan.discriminator(t.cat([fake,one_hot_fused],dim=1)).squeeze()
            generator_loss=self.cdcgan.bce(dis_fake,real_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
