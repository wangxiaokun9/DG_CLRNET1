import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, alpha=1.0, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

def one_hot(labels: torch.Tensor, num_classes: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, eps: Optional[float] = 1e-6) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError("labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one. Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

class SoftFocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', label_smoothing: float = 0.1) -> None:
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

        if not len(input.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(input.shape))

        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(input.size(0), target.size(0)))

        n = input.size(0)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))

        if not input.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {} and {}".format(input.device, target.device))

        # Compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # Create the labels one hot tensor with label smoothing
        target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype, eps=self.label_smoothing)

        # Compute the actual focal loss
        weight = torch.pow(1. - input_soft + self.eps, self.gamma)

        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))
        return loss

def soft_focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float = 2.0, reduction: str = 'none', label_smoothing: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))

    if not input.device == target.device:
        raise ValueError("input and target must be in the same device. Got: {} and {}".format(input.device, target.device))

    # Compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1) + eps

    # Create the labels one hot tensor with label smoothing
    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype, eps=label_smoothing)

    # Compute the actual focal loss
    weight = torch.pow(1. - input_soft + eps, gamma)

    focal = -alpha * weight * torch.log(input_soft)
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