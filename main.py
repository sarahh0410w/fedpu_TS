from options import opt
from config import set_config
from modules.model import CNNMnist, ResNet9, ResNet34, CNNCifar, ClipModelat, DinoVisionTransformerClassifier
from roles.FmpuTrainer import FmpuTrainer

import os
import torch

from options import *
# args = args()


def main():
    device = args.device

    #torch.set_default_device('cuda:1')

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    set_config(opt)
    print("Acc from:", opt, "\n")

    if opt.dataset == 'MNIST':
        model = CNNMnist().to(device)
        #model = DinoVisionTransformerClassifier().to(device)


        trainer = FmpuTrainer(model)
    if opt.dataset == 'CIFAR10':
        model = ResNet9().to(device)
        #model = DinoVisionTransformerClassifier().to(device)

        trainer = FmpuTrainer(model)
    if opt.dataset == 'chest':
        model = ResNet34().to(device)

        trainer = FmpuTrainer(model)
    if opt.dataset == 'isic':
        #model = ResNet34().to(device)
        model = DinoVisionTransformerClassifier().to(device)

        trainer = FmpuTrainer(model)
    trainer.begin_train()



if __name__ == '__main__':
    # merge config
    main()

