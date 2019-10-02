from model.WGAN import WGAN
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os

class WGANRunner:
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
        self.wgan=WGAN(self.args)
        self.wgan.generator.cuda()
        self.wgan.discriminator.cuda()

    def _build_loader(self):
        train,test=self._get_dataset()
        self.train_loader=DataLoader(dataset=train, batch_size=self.args.batch_size,num_workers=self.args.num_workers,shuffle=True)
        self.test_loader=DataLoader(dataset=test,batch_size=self.args.batch_size, num_workers=self.args.num_workers,shuffle=False)

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.wgan.generator.parameters(),lr=self.args.generator_lr,betas=(0.5,0.999))
        self.discriminator_optimizer=Adam(self.wgan.discriminator.parameters(),lr=self.args.discriminator_lr,betas=(0.5,0.999))

    def train(self):
        for epoch in range(1,self.args.epoch_num+1):
            logging.info('Start Train Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            self.eval(epoch)

    def eval(self,epoch):
        self.wgan.generator.eval()
        self.wgan.discriminator.eval()
        n=self.args.test_num
        digit_size=28
        figure=np.zeros((digit_size*n, digit_size*n))
        Z=self.wgan.sample(n*n).cuda()
        # batch_size * digit_size * digit_size
        result=(self.wgan.generator(Z).squeeze().detach().cpu().numpy()+1)/2
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

            # batch_size * 1 * 28 * 28  grey image, [0,1]
            images=batch[0].cuda()

            # batch_size
            # label=batch[1]

            batch_size=images.shape[0]

            eps=t.rand(batch_size,1).cuda()
            noise=self.wgan.sample(batch_size).cuda()
            fake=self.wgan.generator(noise).detach()
            inter=eps*images.reshape(-1,self.args.input_dim)+(-eps+1)*fake
            inter.requires_grad=True

            grad=t.autograd.grad(self.wgan.discriminator(inter),inter,t.ones(batch_size,1).cuda(),create_graph=True)[0]

            grad_norm=t.sqrt(t.sum(grad**2,dim=1))

            grad_penalty=self.args.lamda*t.mean(t.relu(grad_norm-1))

            discriminator_loss=t.mean(self.wgan.discriminator(images.reshape(-1,self.args.input_dim))-self.wgan.discriminator(fake))+grad_penalty

            # train discriminator
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.wgan.sample(batch_size).cuda()
            fake=self.wgan.generator(noise)
            generator_loss=t.mean(self.wgan.discriminator(fake))

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
