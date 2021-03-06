import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import pydoc


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, logits, targets):
        targets = targets.type(torch.cuda.LongTensor).view(-1)
        return self.loss(logits, targets)


class MixupCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MixupCrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        """
        in PyTorch's cross entropy, targets are expected to be labels
        so to predict probabilities this loss is needed
        suppose q is the target and p is the input
        loss(p, q) = -\sum_i q_i \log p_i
        """
        assert input.size() == target.size()
        assert isinstance(input, Variable) and isinstance(target, Variable)
        input = torch.log(F.softmax(input, dim=1).clamp(1e-5, 1))
        # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)

        loss = - torch.sum(input * target)
        return loss / input.size()[0] if self.size_average else loss


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 0
        for i in range(2):
            probs_flat = probs[:, i].contiguous().view(-1)
            targets_flat = (targets==i+1).float().contiguous().view(-1)
            loss += self.bce_loss(probs_flat, targets_flat)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce_with_logits((1 - torch.sigmoid(input)) ** self.gamma * F.logsigmoid(input), target)