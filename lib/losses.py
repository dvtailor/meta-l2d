import functools
import numpy as np
import torch
import functorch
import torch.nn.functional as F


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

    l1 = F.binary_cross_entropy_with_logits(outputs[range(batch_size), labels], torch.ones(batch_size, device=outputs.device), reduction='none')
    bce_partial_c0 = functorch.vmap(functools.partial(F.binary_cross_entropy_with_logits, target=torch.zeros(batch_size, device=outputs.device), reduction='none'), in_dims=1, out_dims=1)
    l2 = torch.sum(bce_partial_c0(outputs[:,:n_classes]), dim=-1) - F.binary_cross_entropy_with_logits(outputs[range(batch_size), labels], torch.zeros(batch_size, device=outputs.device), reduction='none')
    l3 = F.binary_cross_entropy_with_logits(outputs[range(batch_size), n_classes], torch.zeros(batch_size, device=outputs.device), reduction='none')
    l4 = F.binary_cross_entropy_with_logits(outputs[range(batch_size), n_classes], torch.ones(batch_size, device=outputs.device), reduction='none')

    l5 = m * (l4 - l3)

    l = (l1 + l2) + l3 + l5

    return torch.mean(l)
