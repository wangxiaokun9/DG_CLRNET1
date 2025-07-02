import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    """
    if not torch.is_tensor(labels):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(
                type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0],
                          num_classes,
                          *shape[1:],
                          device=device,
                          dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def focal_loss_with_label_smoothing(input: torch.Tensor,
                                    target: torch.Tensor,
                                    alpha: float,
                                    gamma: float = 2.0,
                                    reduction: str = 'none',
                                    smoothing: float = 0.1,
                                    eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss with Label Smoothing.

    Args:
        input (torch.Tensor): logits tensor with shape (N, C, *) where C is the number of classes.
        target (torch.Tensor): ground truth labels with shape (N, *).
        alpha (float): focal loss alpha parameter.
        gamma (float): focal loss gamma parameter.
        reduction (str): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        smoothing (float): label smoothing factor, where 0.0 means no smoothing and 1.0 means maximum smoothing.
        eps (float): small value to avoid evaluation of log(0) when computing the loss.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if not len(input.shape) >= 2:
        raise ValueError(
            "Invalid input shape, we expect BxCx*. Got: {}".format(
                input.shape))

    if input.size(0) != target.size(0):
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).'.
            format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n, ) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".
            format(input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor with label smoothing
    target_one_hot: torch.Tensor = one_hot(target,
                                           num_classes=input.shape[1],
                                           device=input.device,
                                           dtype=input.dtype)
    # Apply label smoothing
    target_one_hot = (1 - smoothing) * target_one_hot + smoothing / input.shape[1]

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(
            "Invalid reduction mode: {}".format(reduction))
    return loss

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self,
                 alpha: float,
                 gamma: float = 2.0,
                 reduction: str = 'none',
                 smoothing: float = 0.1) -> None:
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.smoothing: float = smoothing
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_label_smoothing(input, target, self.alpha, self.gamma,
                                               self.reduction, self.smoothing, self.eps)