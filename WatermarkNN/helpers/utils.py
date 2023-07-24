
"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""

import os
import sys
import time
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def re_initializer_layer(model, num_classes, layer=None):
    """remove the last layer and add a new one"""
    indim = model.module.linear.in_features
    private_key = model.module.linear
    if layer:
        model.module.linear = layer
    else:
        model.module.linear = nn.Linear(indim, num_classes).cuda()
    return model, private_key


def load_yaml(path):
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return argparse.Namespace(**config_dict)


# Function to parse arguments and return a Namespace object
def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
    parser.add_argument('--config', default='config.yaml', help='path to the configuration file in YAML format')
    args = parser.parse_args()
    return load_yaml(args.config)
