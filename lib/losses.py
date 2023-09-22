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
