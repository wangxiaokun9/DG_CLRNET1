import torch

def g_laine_iou(pred, target, img_w, prior_ys, img_h, base_length=15, curvature_alpha=0.1, aligned=True):
    '''
    结合动态宽度、曲率调整和GLIoU的改进版IoU计算
    '''
    # 计算动态长度（来自laine_iou）
    dynamic_lengths = base_length * (prior_ys * 0.5 + 0.5)
    
    # 曲率计算（来自laine_iou）
    def compute_curvature(x):
        curvature = torch.zeros_like(x)
        if x.shape[0] == 0:
            return curvature
        left_diff = torch.abs(x[:, 1:] - x[:, :-1])
        curvature[:, :-1] += left_diff
        right_diff = torch.abs(x[:, :-1] - x[:, 1:])
        curvature[:, 1:] += right_diff
        return curvature
    
    curvature_pred = compute_curvature(pred)
    curvature_target = compute_curvature(target)
    
    # 曲率归一化并调整长度
    curvature_pred_normalized = curvature_pred / img_w
    curvature_target_normalized = curvature_target / img_w
    adjusted_lengths_pred = dynamic_lengths * (1 + curvature_alpha * curvature_pred_normalized)
    adjusted_lengths_target = dynamic_lengths * (1 + curvature_alpha * curvature_target_normalized)
    
    # 生成扩展区间（结合动态长度）
    px1 = pred - adjusted_lengths_pred
    px2 = pred + adjusted_lengths_pred
    tx1 = target - adjusted_lengths_target
    tx2 = target + adjusted_lengths_target
    
    # 计算IoU（来自gline_iou）
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        cx1 = torch.min(px1, tx1)  # 最小外接左边界
        cx2 = torch.max(px2, tx2)   # 最大外接右边界
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = torch.min(px2.unsqueeze(1), tx2.unsqueeze(0)) - torch.max(px1.unsqueeze(1), tx1.unsqueeze(0))
        union = torch.max(px2.unsqueeze(1), tx2.unsqueeze(0)) - torch.min(px1.unsqueeze(1), tx1.unsqueeze(0))
        cx1 = torch.min(px1.unsqueeze(1), tx1.unsqueeze(0))  # 三维外接
        cx2 = torch.max(px2.unsqueeze(1), tx2.unsqueeze(0))  # 三维外接

    # 处理无效区域
    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    cx1[invalid_masks] = 0.
    cx2[invalid_masks] = 0.
    
    # 计算IoU和闭合区域
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    c = cx2 - cx1  # 闭合区域长度
    c_sum = c.sum(dim=-1)
    union_sum = union.sum(dim=-1)
    
    # 计算GLIoU（来自gliou_loss）
    gliou = iou - (c_sum - union_sum) / (c_sum + 1e-9)
    return gliou

def g_laiou_loss(pred, target, img_w, prior_ys, img_h, base_length=15, curvature_alpha=0.1):
    '''
    结合动态宽度、曲率调整和GLIoU的新损失函数
    '''
    gliou = g_laine_iou(pred, target, img_w, prior_ys, img_h, base_length, curvature_alpha, aligned=True)
    return (1 - gliou).mean()