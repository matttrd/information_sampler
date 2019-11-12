import torch
from exptutils import logical_index
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = logical_index(target, input.shape)
        loglogit = F.log_softmax(input, dim=-1)
        logit = loglogit.exp().clamp(self.eps, 1. - self.eps)

        loss = -1 * y.float() * loglogit # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum(dim=1)