import numpy as np
import torch


def onehot(t, num_classes):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    t = t.type(torch.LongTensor)
    assert isinstance(t, torch.Tensor)
    return torch.zeros(t.size()[0], num_classes).scatter_(1, t.view(-1, 1), 1)
