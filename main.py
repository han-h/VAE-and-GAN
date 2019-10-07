from runner.VAERunner import VAERunner
from runner.CVAERunner import CVAERunner
from runner.DCGANRunner import DCGANRunner
from runner.CDCGANRunner import CDCGANRunner
from runner.WGANRunner import WGANRunner
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='wgan', help='')
    parser.add_argument('--batch-size', type=int, default=128, help='')
    parser.add_argument('--num-workers', type=int,default=5,help='')
    parser.add_argument('--display-n-batches', type=int, default=50, help='')
    parser.add_argument('--gpu', type=str, default='3', help='')
    parser.add_argument('--data-root',type=str, default='./data',help='')

    parser.add_argument('--digit-size', type=int, default=28, help='')
    parser.add_argument('--input-dim', type=int, default=784, help='')
    parser.add_argument('--hidden-dim', type=int, default=256, help='')
    parser.add_argument('--latent-dim', type=int, default=10, help='')
    parser.add_argument('--epoch-num', type=int, default=10, help='')

    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--test-num', type=int,default=10,help='')
    parser.add_argument('--num-classes', type=int,default=10,help='')
    parser.add_argument('--lamda', type=float, default=10., help='')

    return parser.parse_args()


def main(args):
    print(args)
    if args.model=='ave':
        runner=VAERunner(args)
    elif args.model=='cvae':
        runner=CVAERunner(args)
    elif args.model=='dcgan':
        runner=DCGANRunner(args)
    elif args.model=='cdcgan':
        runner=CDCGANRunner(args)
    elif args.model=='wgan':
        runner=WGANRunner(args)
    else:
        runner=VAERunner(args)
    runner.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args=parse_args()
    main(args)
