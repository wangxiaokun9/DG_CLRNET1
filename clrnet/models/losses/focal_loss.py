# pylint: disable-all
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://github.com/kornia/kornia/blob/f4f70fefb63287f72bc80cd96df9c061b1cb60dd/kornia/losses/focal.py


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):  # ignore_lb 忽略的标签，通常用于处理数据集中某些未标注的区域，默认值为 255
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma   # 控制难易样本的权重，gamma 值越大，对难分类的样本的关注度越高
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)  # 初始化 NLLLoss 损失函数，用于计算对数似然损失，并且可以指定忽略的标签。

    def forward(self, logits, labels): #   logits：模型的原始输出，形状为 (N, C, *)，其中 N 是批量大小，C 是类别数，* 表示任意其他维度。 labels：真实标签，形状为 (N, *)
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)  # 用于调整损失函数，使得难分类的样本有更高的权重。
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

# 将整数标签转换为 one-hot 编码的张量。
def one_hot(labels: torch.Tensor, # 整数标签张量，形状为 (N, *)
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None, #返回张量的数据类型，默认为输入张量的类型。
            eps: Optional[float] = 1e-6) -> torch.Tensor: # 一个小的值，用于防止数值不稳定，默认为 1e-6
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

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
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


def focal_loss(input: torch.Tensor,
               target: torch.Tensor,
               alpha: float,
               gamma: float = 2.0,
               reduction: str = 'none', # 损失值的归约方式，可选值为 'none'、'mean' 和 'sum'
               eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
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

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target,
                                           num_classes=input.shape[1],
                                           device=input.device,
                                           dtype=input.dtype)

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


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """
    def __init__(self,
                 alpha: float,
                 gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma,
                          self.reduction, self.eps)
