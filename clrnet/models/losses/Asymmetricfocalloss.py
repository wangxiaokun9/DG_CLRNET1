import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        pos_factor = torch.pow(1. - scores, self.gamma_pos)
        neg_factor = torch.pow(scores, self.gamma_neg)
        factor = torch.where(labels.unsqueeze(1) == 1, pos_factor, neg_factor)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError("labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one. Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def focal_loss(input: torch.Tensor,
               target: torch.Tensor,
               alpha: float,
               gamma_pos: float = 2.0,
               gamma_neg: float = 2.0,
               reduction: str = 'none',
               eps: float = 1e-8) -> torch.Tensor:
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n, ) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))

    if not input.device == target.device:
        raise ValueError("input and target must be in the same device. Got: {} and {}".format(input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    pos_factor = torch.pow(1. - input_soft, gamma_pos)
    neg_factor = torch.pow(input_soft, gamma_neg)
    factor = torch.where(target_one_hot > 0, pos_factor, neg_factor)
    focal = -alpha * factor * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss

class FocalLoss_A(nn.Module):
    def __init__(self,
                 alpha: float,
                 gamma_pos: float = 2.0,
                 gamma_neg: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss_A, self).__init__()
        self.alpha: float = alpha
        self.gamma_pos: float = gamma_pos
        self.gamma_neg: float = gamma_neg
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma_pos, self.gamma_neg, self.reduction, self.eps)