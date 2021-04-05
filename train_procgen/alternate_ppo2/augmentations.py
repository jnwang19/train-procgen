import numpy as np
import torch

# Based on https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0, use_cuda=False):
    '''Compute the mixup data. Return mixed inputs and targets.'''
    
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
        
    batch_size = x.size()[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    
    return mixed_x, mixed_y
