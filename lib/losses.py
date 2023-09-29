import numpy as np
import torch


def cross_entropy(outputs, m, labels, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR with alpha=1
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]
    rc = [n_classes] * batch_size # idx to extract rejector function
    outputs = -m * torch.log2(outputs[range(batch_size), rc]) - torch.log2(outputs[range(batch_size), labels])
    return torch.sum(outputs) / batch_size


def ova(outputs, m, labels, n_classes):
    batch_size = outputs.size()[0]
    l1 = logistic_loss(outputs[range(batch_size), labels], 1)
    l2 = torch.sum(logistic_loss(outputs[:,:n_classes], -1), dim=1) - logistic_loss(outputs[range(batch_size),labels],-1)
    l3 = logistic_loss(outputs[range(batch_size), n_classes], -1)
    l4 = logistic_loss(outputs[range(batch_size), n_classes], 1)

    l5 = m * (l4 - l3)

    l = (l1 + l2) + l3 + l5

    return torch.mean(l)

# TODO: double check this
def logistic_loss(outputs, y):
    outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
    l = torch.log2(1 + torch.exp((-1*y)*outputs))
    return l
