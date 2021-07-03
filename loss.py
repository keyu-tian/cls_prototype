import numpy as np
import torch


class LabelSmoothCELoss(torch.nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        
        self.logsoft = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)
        
        loss = - torch.sum(self.logsoft(input) * (one_hot.detach())) / input.size(0)
        return loss


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(loss_fn, logits, tar_a, tar_b, lam):
    return lam * loss_fn(logits, tar_a) + (1 - lam) * loss_fn(logits, tar_b)
