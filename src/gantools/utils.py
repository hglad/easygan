import numpy as np
import torch


def real_data_target(size, delta, flip_labels=False, cuda=False):
    '''
    Tensor containing ones, with shape = size
    '''
    # data = torch.ones(size, 1) - torch.rand(size, 1)*delta
    data = torch.ones(size, 1) - delta
    if flip_labels:
        inds = np.random.randint(0,size,int(size/8))
        for i in inds:
            # data[i] = torch.zeros(1) + delta
            data[i] = delta*torch.rand(1)
    if cuda:
        return data.cuda()
    return data

def fake_data_target(size, delta=0, cuda=False):
    '''
    Tensor containing zeros, with shape = size
    '''
    # data = torch.zeros(size, 1) + delta
    data = delta*torch.rand(size, 1)            # number between 0 and delta
    # print (data.min(), data.max())
    # data = torch.zeros(size, 1)
    if cuda:
        return data.cuda()
    return data

def noise(size, n, cuda=False):
    noise = torch.randn(size, n)
    if cuda:
        return noise.cuda()
    return noise
