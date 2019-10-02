from model.GAN import GAN
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os

class GANRunner:
    def __init__(self,args):
        super().__init__()
        self.args=args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self._build_loader()
        self._build_model()
        self._build_optimizer()

    def _get_dataset(self):
        transform=transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])
        train=datasets.MNIST(root=self.args.data_root,transform=transform,train=True,download=True)
        test=datasets.MNIST(root=self.args.data_root,transform=transform,train=False,download=True)
        return train,test

    def _build_model(self):
        self.gan=GAN(self.args)
        self.gan.generator.cuda()
        self.gan.discriminator.cuda()

    def _build_loader(self):
        train,test=self._get_dataset()
        self.train_loader=DataLoader(dataset=train, batch_size=self.args.batch_size,num_workers=self.args.num_workers,shuffle=True)
        self.test_loader=DataLoader(dataset=test,batch_size=self.args.batch_size, num_workers=self.args.num_workers,shuffle=False)

    def _build_optimizer(self):
        self.generator_optimizer=Adam(self.gan.generator.parameters(),lr=self.args.generator_lr)
        self.discriminator_optimizer=Adam(self.gan.discriminator.parameters(),lr=self.args.discriminator_lr)

    def train(self):
        for epoch in range(1,self.args.epoch_num+1):
            logging.info('Start Train Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            self.eval(epoch)

    def eval(self,epoch):
        self.gan.generator.eval()
        self.gan.discriminator.eval()
        n=self.args.test_num
        digit_size=64
        figure=np.zeros((digit_size*n, digit_size*n))
        Z=self.gan.sample(n*n).cuda().unsqueeze(-1).unsqueeze(-1)
        # batch_size * digit_size * digit_size
        result=(self.gan.generator(Z).squeeze().detach().cpu().numpy()+1)/2
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/GAN/result"+str(epoch)+".png")
        plt.close('all')

    def _train_one_epoch(self,epoch):
        self.gan.generator.train()
        self.gan.discriminator.train()
        for batch_id,batch in enumerate(self.train_loader, 1):

            # batch_size * 1 * 64 * 64  grey image, scale to [-1,1)
            images=batch[0]*2-1

            # batch_size
            # label=batch[1]

            batch_size=images.shape[0]

            real_target=t.ones(batch_size).cuda()
            fake_target=t.zeros(batch_size).cuda()

            # train discriminator
            noise=self.gan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)

            # batch_size * input_dim
            # detach to stop gradient back propagating to generator
            fake=self.gan.generator(noise).detach()
            dis_fake=self.gan.discriminator(fake).squeeze()
            dis_real=self.gan.discriminator(images.cuda()).squeeze()

            discriminator_loss=self.gan.bce(dis_real,real_target)+self.gan.bce(dis_fake,fake_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # train generator
            noise=self.gan.sample(batch_size).cuda().unsqueeze(-1).unsqueeze(-1)
            # batch_size * input_dim
            fake=self.gan.generator(noise)

            dis_fake=self.gan.discriminator(fake).squeeze()
            generator_loss=self.gan.bce(dis_fake,real_target)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, gen_loss = %.4f, dis_loss = %.4f' % (epoch,batch_id,generator_loss,discriminator_loss))
