# import torch

# def laine_iou(pred, target, img_w, img_h, length=15, aligned=True):
#     '''
#     Calculate the line IoU with dynamic width based on相邻点斜率.
#     '''
#     n_offsets = pred.shape[1]
#     dy_pixel = img_h / (n_offsets - 1)  # 垂直步长（像素）

#     # 计算动态宽度
#     def get_dynamic_length(x):
#         dx = x[:, 1:] - x[:, :-1]
#         dx = torch.cat([dx, dx[:, -1:]], dim=1)  # 边缘点处理
#         length_i = length * torch.sqrt(dx**2 + dy_pixel**2) / dy_pixel
#         return length_i

#     length_pred = get_dynamic_length(pred)
#     length_target = get_dynamic_length(target)

#     px1 = pred - length_pred
#     px2 = pred + length_pred
#     tx1 = target - length_target
#     tx2 = target + length_target

#     if aligned:
#         invalid_mask = (target < 0) | (target >= img_w)
#         ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
#         union = torch.max(px2, tx2) - torch.min(px1, tx1)
#     else:
#         num_pred = pred.shape[0]
#         invalid_mask = (target < 0) | (target >= img_w)
#         invalid_mask = invalid_mask.unsqueeze(0).repeat(num_pred, 1, 1)
#         ovr = (torch.min(px2.unsqueeze(1), tx2.unsqueeze(0)) - 
#                torch.max(px1.unsqueeze(1), tx1.unsqueeze(0)))
#         union = (torch.max(px2.unsqueeze(1), tx2.unsqueeze(0)) - 
#                  torch.min(px1.unsqueeze(1), tx1.unsqueeze(0)))

#     ovr[invalid_mask] = 0.0
#     union[invalid_mask] = 0.0
#     iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
#     return iou

# def center_distance_loss(pred, target, img_w):
#     valid_mask = (target >= 0) & (target < img_w)
#     pred_valid = pred[valid_mask]
#     target_valid = target[valid_mask]
#     if pred_valid.numel() == 0:
#         return torch.tensor(0.0, device=pred.device)
#     distance = torch.abs(pred_valid - target_valid)
#     normalized_distance = distance / img_w
#     return normalized_distance.mean()

# def laiou_loss(pred, target, img_w, img_h, length=15, alpha=0.5):
#     iou = laine_iou(pred, target, img_w, img_h, length, aligned=True)
#     center_loss = center_distance_loss(pred, target, img_w)
#     total_loss = (1 - iou).mean() + alpha * center_loss
#     return total_loss

import torch

def laine_iou(pred, target, img_w, prior_ys, img_h, base_length=15, curvature_alpha=0.1, aligned=True):
    '''
    Calculate the line IoU with dynamic widths based on y-coordinates and curvature adjustment.
    '''
    # Calculate dynamic lengths based on y-coordinates
    dynamic_lengths = base_length * (prior_ys * 0.5 + 0.5)  # Adjust scaling as needed
    
    # Compute curvature for pred and target
    def compute_curvature(x):
        curvature = torch.zeros_like(x)
        if x.shape[0] == 0:
            return curvature
        # Left difference for all except first point
        left_diff = torch.abs(x[:, 1:] - x[:, :-1])
        curvature[:, :-1] += left_diff
        # Right difference for all except last point
        right_diff = torch.abs(x[:, :-1] - x[:, 1:])
        curvature[:, 1:] += right_diff
        return curvature
    
    curvature_pred = compute_curvature(pred)
    curvature_target = compute_curvature(target)
    
    # Normalize curvature by image width
    curvature_pred_normalized = curvature_pred / img_w
    curvature_target_normalized = curvature_target / img_w
    
    # Adjust lengths based on curvature
    adjusted_lengths_pred = dynamic_lengths * (1 + curvature_alpha * curvature_pred_normalized)
    adjusted_lengths_target = dynamic_lengths * (1 + curvature_alpha * curvature_target_normalized)
    
    # Calculate intervals
    px1 = pred - adjusted_lengths_pred
    px2 = pred + adjusted_lengths_pred
    tx1 = target - adjusted_lengths_target
    tx2 = target + adjusted_lengths_target
    
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = torch.min(px2.unsqueeze(1), tx2.unsqueeze(0)) - torch.max(px1.unsqueeze(1), tx1.unsqueeze(0))
        union = torch.max(px2.unsqueeze(1), tx2.unsqueeze(0)) - torch.min(px1.unsqueeze(1), tx1.unsqueeze(0))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou

def laiou_loss(pred, target, img_w, prior_ys, img_h, base_length=15, curvature_alpha=0.1):
    return (1 - laine_iou(pred, target, img_w, prior_ys, img_h, base_length, curvature_alpha, aligned=True)).mean()