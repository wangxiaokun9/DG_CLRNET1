import torch
import math

def perspective_aware_length(img_h, y_coords, min_length=5, max_length=20):
    """
    根据y坐标动态计算长度，考虑透视效应
    Args:
        img_h: 图像高度
        y_coords: y坐标张量 (归一化或像素坐标)
        min_length: 最小长度
        max_length: 最大长度
    """
    # 假设y_coords是归一化的，转换为像素坐标
    y_pixels = y_coords * img_h
    # 计算动态长度，远处的点(length接近min_length)，近处的点(length接近max_length)
    lengths = min_length + (max_length - min_length) * (1 - y_coords)
    return lengths

def curvature_aware_adjustment(points, img_w, base_length, alpha=0.3):
    """
    根据曲率调整长度
    Args:
        points: 车道线点坐标 (归一化)
        img_w: 图像宽度
        base_length: 基础长度
        alpha: 曲率影响系数
    """
    if len(points) < 3:
        return base_length
    
    # 转换为像素坐标
    points_pixel = points * (img_w - 1)
    
    # 计算相邻点之间的水平偏移
    dx = torch.abs(points_pixel[1:] - points_pixel[:-1])
    avg_dx = torch.mean(dx)
    
    # 曲率影响因子
    curvature_factor = 1 + alpha * avg_dx / base_length
    
    return base_length * curvature_factor

def dynamic_line_iou(pred, target, img_w, img_h, min_length=5, max_length=20, aligned=True):
    '''
    改进的Line IoU计算，考虑透视效应和曲率
    Args:
        pred: 预测车道线，shape: (num_pred, 72)
        target: 真实车道线，shape: (num_target, 72)
        img_w: 图像宽度
        img_h: 图像高度
        min_length: 最小长度
        max_length: 最大长度
        aligned: 是否对齐计算
    '''
    # 生成动态长度
    y_coords = torch.linspace(1, 0, steps=pred.shape[-1], device=pred.device)
    
    # 计算基础长度（透视感知）
    base_lengths = perspective_aware_length(img_h, y_coords, min_length, max_length)
    
    # 对预测和真实车道线分别计算动态长度
    pred_lengths = base_lengths.clone()
    target_lengths = base_lengths.clone()
    
    # 考虑曲率调整
    for i in range(pred.shape[0]):
        pred_lengths = curvature_aware_adjustment(pred[i], img_w, pred_lengths)
    for i in range(target.shape[0]):
        target_lengths = curvature_aware_adjustment(target[i], img_w, target_lengths)
    
    # 计算IoU
    px1 = pred - pred_lengths
    px2 = pred + pred_lengths
    tx1 = target - target_lengths
    tx2 = target + target_lengths
    
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou

def dynamic_liou_loss(pred, target, img_w, img_h, min_length=5, max_length=20):
    return (1 - dynamic_line_iou(pred, target, img_w, img_h, min_length, max_length)).mean()