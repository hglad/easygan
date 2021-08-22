import numpy as np
import torch
import os
import sys
import imageio

def real_data_target(size, delta=0, cuda=False):
    '''
    Tensor containing ones, with shape = size
    '''
    # data = torch.ones(size, 1) - torch.rand(size, 1)*delta
    data = torch.ones(size, 1) - delta
    if cuda:
        return data.cuda()
    return data


def fake_data_target(size, delta=0, cuda=False):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1) + delta
    if cuda:
        return data.cuda()
    return data


def noise(size, n, cuda=False):
    """
    Create latent vector for generator input
    """
    noise = torch.randn(size, n)
    if cuda:
        return noise.cuda()
    return noise


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def normalize(img, mean=0.5, std=0.5):
    return (img - mean)/std


def un_normalize(img, mean=0.5, std=0.5):
    return mean + img*std
