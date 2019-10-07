from model.CVAE import CVAE
from .BasicRunner import BasicRunner
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import logging

# input_dim = 784
# hidden_dim = 256
# latent_dim = 10
# lr = 2e-4
# epoch_num = 20
class CVAERunner(BasicRunner):
    def __init__(self,args):
        super(CVAERunner,self).__init__(args)

    def _build_model(self):
        self.cvae=CVAE(self.args).cuda()

    def _build_optimizer(self):
        self.optimizer=Adam(self.cvae.parameters(),lr=self.args.lr)

    def _get_fixed_noise_for_evaluation(self):
        n=self.args.test_num
        self.Z=self.cvae.sample(n*n).cuda()


    def eval(self,epoch):
        self.cvae.eval()
        n=self.args.test_num
        digit_size=self.args.digit_size

        figure=np.zeros((digit_size*n, digit_size*n))

        objective=t.zeros(n*n,self.args.num_classes).cuda()
        for i in range(n):
            objective[i*n:(i+1)*n,i]=1
        objective_mean=self.cvae.condition(objective)

        result=self.cvae.decoder(self.Z+objective_mean).detach().cpu().numpy()
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
