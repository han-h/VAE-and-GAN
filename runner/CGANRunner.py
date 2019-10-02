from model.CGAN import CGAN
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os

class CGANRunner:
    def __init__(self,args):
        super().__init__()
        self.args=args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self._build_loader()
        self._build_model()
        self._build_optimizer()

    def _get_dataset(self):
        transform=transforms.Compose([transforms.ToTensor()])
        train=datasets.MNIST(root=self.args.data_root,transform=transform,train=True,download=True)
        test=datasets.MNIST(root=self.args.data_root,transform=transform,train=False,download=True)
        return train,test

    def _build_model(self):
        self.cgan=CGAN(self.args)
        self.cgan.generator.cuda()
        self.cgan.discriminator.cuda()

    def _build_loader(self):
        train,test=self._get_dataset()
        self.train_loader=DataLoader(dataset=train, batch_size=self.args.batch_size,num_workers=self.args.num_workers,shuffle=True)
        self.test_loader=DataLoader(dataset=test,batch_size=self.args.batch_size, num_workers=self.args.num_workers,shuffle=False)

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.cgan.generator.parameters(),lr=self.args.lr,betas=(0.5,0.999))
        self.discriminator_optimizer=Adam(self.cgan.discriminator.parameters(),lr=self.args.lr,betas=(0.5,0.999))

    def train(self):
        for epoch in range(1,self.args.epoch_num+1):
            logging.info('Start Train Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            self.eval(epoch)

    def eval(self,epoch):
        self.cgan.generator.eval()
        self.cgan.discriminator.eval()
        n=self.args.test_num
        digit_size=28
        figure=np.zeros((digit_size*n, digit_size*n))
        Z=self.cgan.sample(n*n).cuda().unsqueeze(-1).unsqueeze(-1)
        objective=t.zeros(n*n,self.args.num_classes).cuda()
        for i in range(n):
            objective[i*n:(i+1)*n,i]=1
        fused=t.cat([Z,objective.unsqueeze(-1).unsqueeze(-1)],dim=1)
        # batch_size * digit_size * digit_size
        result=(self.cgan.generator(fused).squeeze().detach().cpu().numpy()+1)/2
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/CGAN/result"+str(epoch)+".png")
        plt.close('all')

    def _train_one_epoch(self,epoch):
        self.cgan.generator.train()
        self.cgan.discriminator.train()
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
            noise=self.cgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)

            # batch_size * nz * 1 * 1
            fused=t.cat([noise,one_hot.unsqueeze(-1).unsqueeze(-1)],dim=1)

            # batch_size * 1 * 28 * 28
            # detach to stop gradient back propagating to generator
            fake=self.cgan.generator(fused).detach()
            dis_fake=self.cgan.discriminator(t.cat([fake,one_hot_fused],dim=1)).squeeze()
            dis_real=self.cgan.discriminator(t.cat([images.cuda(),one_hot_fused],dim=1)).squeeze()

            discriminator_loss=self.cgan.bce(dis_real,real_target)+self.cgan.bce(dis_fake,fake_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.cgan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            # batch_size * nz * 1 * 1
            fused=t.cat([noise,one_hot.unsqueeze(-1).unsqueeze(-1)],dim=1)

            # batch_size * 1 * 28 * 28
            fake=self.cgan.generator(fused)

            dis_fake=self.cgan.discriminator(t.cat([fake,one_hot_fused],dim=1)).squeeze()
            generator_loss=self.cgan.bce(dis_fake,real_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
