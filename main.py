#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from data import DonutDataset
from model import Model, CGAN


def train(args):
    train_loader = DataLoader(DonutDataset(root=args.data_root, is_train=True),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=3)
    test_loader = DataLoader(DonutDataset(root=args.data_root, is_train=False),
                             batch_size=args.batch_size, shuffle=True, drop_last=False,
                             num_workers=3)
    if args.model == 'regressor':
        model = Model(args)
    elif args.model == 'gan':
        model = CGAN(args)
    else:
        raise Exception('Not implemented')
    for epoch in range(args.epochs):
        print("EPOCH: ", epoch)
        model.train_one_epoch(train_loader, epoch)
        model.test_one_epoch(test_loader, epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='donut exp')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='regressor', choices=['regressor', 'gan'],
                        help='Use regressor or GAN')
    parser.add_argument('--nn', type=str, default='resnet18', metavar='N',
                        choices=['resnet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152'],
                        help='Embedding nn to use, [resnet18, resnet34,'
                             'resnet50, resnet101, resnet152]')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of train batch')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Use SGD or ADAM')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_parameters', type=int, default=46,
                        help='num of parameters to predict')
    parser.add_argument('--loss', type=str, default='l2',
                        help='loss to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='pretrained model')
    parser.add_argument('--gan_mode', type=str, default='vanilla',
                        choices=['lsgan', 'vanilla', 'wgangp'],
                        help='Type of GAN loss to use')
    parser.add_argument('--non_local', type=bool, default=False,
                        help='use non local after each res block')
    parser.add_argument('--anti_alias', type=bool, default=False,
                        help='use anti_alias after each pool/strideconv')
    parser.add_argument('--scale_param', type=bool, default=False,
                        help='scale param')
    parser.add_argument('--use_psf', type=bool, default=False,
                        help='use psf')
    parser.add_argument('--data_root', type=str, default='data',
                        help='data_root')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
