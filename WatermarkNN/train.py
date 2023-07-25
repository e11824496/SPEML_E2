from __future__ import print_function

import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.loaders import getdataloader, getwmloader
from helpers.utils import adjust_learning_rate, parse_args
from models import ResNet18
from trainer import test, train_epoch
from helpers.consts import MODELS

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    RUNNAME = 'train_' + args.runname
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(RUNNAME) + '.txt')
    confgfile = os.path.join(LOG_DIR, 'conf_' + str(RUNNAME) + '.txt')

    trainloader, testloader, n_classes = getdataloader(
        args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

    wmloader = None
    if args.wmtrain:
        print('Loading watermark images')
        wmloader = getwmloader(args.wm_path, args.wm_batch_size, args.wm_lbl)

    # create the model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.load_path), 'Error: no checkpoint found!'
        checkpoint = torch.load(args.load_path)
        net = checkpoint['net']
        acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        net_gen = MODELS[args.model]
        net = net_gen(num_classes=n_classes)

    net = net.to(device)
    # support cuda
    if device == 'cuda':
        print('Using CUDA')
        print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # loading wm examples
    if args.wmtrain:
        print("WM acc:")
        test(net, criterion, logfile, wmloader, device)

    acc = None
    acc_list = []
    wm_acc = None
    wm_acc_list = []

    # start training
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        # adjust learning rate
        adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)

        train_epoch(epoch, net, criterion, optimizer, logfile,
            trainloader, device, wmloader)

        print("Test acc:")
        acc = test(net, criterion, logfile, testloader, device)
        acc_list.append(acc.item())

        if args.wmtrain:
            print("WM acc:")
            wm_acc = test(net, criterion, logfile, wmloader, device)
            wm_acc_list.append(wm_acc.item())

        print('Saving..')
        state = {
            'net': net.module if device == 'cuda' else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, args.save_model))

    if wm_acc:
        return acc_list, wm_acc_list

    return acc_list


if __name__ == '__main__':
    args = parse_args()

    train(args)
