import os
import logging
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

class BasicRunner():
    def __init__(self,args):
        self.args=args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self._build_loader()
        self._build_model()
        self._build_optimizer()
        self._get_fixed_noise_for_evaluation()

    def _get_dataset(self):
        transform=transforms.Compose([transforms.ToTensor()])
        return datasets.MNIST(root=self.args.data_root,transform=transform,train=True,download=True)

    def _build_loader(self):
        self.train_loader=DataLoader(dataset=self._get_dataset(), batch_size=self.args.batch_size,num_workers=self.args.num_workers,shuffle=True)

    def _build_model(self):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def train(self):
        for epoch in range(1,self.args.epoch_num+1):
            logging.info('Start Train Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            self.eval(epoch)

    def _get_fixed_noise_for_evaluation(self):
        raise NotImplementedError

    def eval(self,epoch):
        raise NotImplementedError

    def _train_one_epoch(self,epoch):
        raise NotImplementedError

