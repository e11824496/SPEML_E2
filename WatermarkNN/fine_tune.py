"""Train CIFAR with PyTorch."""
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.loaders import *
from helpers.utils import re_initializer_layer, parse_args
from trainer import test, train_epoch

def fine_tune(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    RUNNAME = 'fine_tune_' + args.runname
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(RUNNAME) + '.txt')

    trainloader, testloader, n_classes = getdataloader(
        args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

    # load watermark images
    print('Loading watermark images')
    wmloader = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

    # Loading model.
    print('==> loading model...')
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

    net = net.to(device)
    # support cuda
    if device == 'cuda':
        print('Using CUDA')
        print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    if args.reinitll:
        net, _ = re_initializer_layer(net, n_classes)

    if device is 'cuda':
        net.module.unfreeze_model()
    else:
        net.unfreeze_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.fine_tune_lr, momentum=0.9, weight_decay=5e-4)

    # start training loop
    print("WM acc:")
    test(net, criterion, logfile, wmloader, device)
    print("Test acc:")
    test(net, criterion, logfile, testloader, device)

    acc = None
    wm_acc = None

    # start training
    for epoch in range(start_epoch, start_epoch + args.fine_tune_epochs):
        train_epoch(epoch, net, criterion, optimizer, logfile,
                trainloader, device, wmloader=False, tune_all=args.tunealllayers)

        print("Test acc:")
        acc = test(net, criterion, logfile, testloader, device)

        print("WM acc:")
        wm_acc = test(net, criterion, logfile, wmloader, device)

        print('Saving..')
        state = {
            'net': net.module if device is 'cuda' else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, str(RUNNAME) + str(args.save_model)))

    return acc, wm_acc

if __name__ == '__main__':
    args = parse_args()

    fine_tune(args)