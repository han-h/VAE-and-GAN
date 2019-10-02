from model.CVAE import CVAE
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os

class CVAERunner:
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
        self.cvae=CVAE(self.args).cuda()

    def _build_loader(self):
        train,test=self._get_dataset()
        self.train_loader=DataLoader(dataset=train, batch_size=self.args.batch_size,num_workers=self.args.num_workers,shuffle=True)
        self.test_loader=DataLoader(dataset=test,batch_size=self.args.batch_size, num_workers=self.args.num_workers,shuffle=False)

    def _build_optimizer(self):
        self.optimizer=Adam(self.cvae.parameters(),lr=self.args.lr)

    def train(self):
        for epoch in range(1,self.args.epoch_num+1):
            logging.info('Start Train Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            self.eval(epoch)

    def eval(self,epoch):
        self.cvae.eval()
        n=self.args.test_num
        digit_size=28
        figure=np.zeros((digit_size*n, digit_size*n))
        Z=self.cvae.sample(n*n).cuda()
        objective=t.zeros(n*n,self.args.num_classes).cuda()
        for i in range(n):
            objective[i*n:(i+1)*n,i]=1
        objective_mean=self.cvae.condition(objective)
        result=self.cvae.decoder(Z+objective_mean).detach().cpu().numpy()
        result=result.reshape(n*n,digit_size,digit_size)
        for i in range(n):
            for j in range(n):
                figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=result[i*n+j]

        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap='Greys_r')
        plt.savefig("./img/CVAE/result"+str(epoch)+".png")
        plt.close('all')



    def _train_one_epoch(self,epoch):
        self.cvae.train()
        for batch_id,batch in enumerate(self.train_loader, 1):

            # batch_size * 1 * 28 * 28  grey image 0.0-1.0
            images=batch[0]
            # batch_size
            label=batch[1]
            # batch_size * num_classes
            one_hot=t.zeros(images.shape[0],self.args.num_classes).scatter_(1,label.unsqueeze(-1),1).cuda()

            self.optimizer.zero_grad()

            output,cvae_loss=self.cvae(images.cuda().reshape(-1,self.args.input_dim),one_hot)

            cvae_loss.backward()
            self.optimizer.step()

            if batch_id % self.args.display_n_batches == 0:
                logging.info('Train Epoch %d, Batch %d, loss = %.4f' % (epoch,batch_id, cvae_loss))
