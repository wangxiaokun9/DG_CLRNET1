import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.hub import load_state_dict_from_url
from torchvision.ops import DeformConv2d
from clrnet.models.registry import BACKBONES
from torchvision.ops import deform_conv2d



model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':
    'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2':
    'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class DeformableTransposedConv(nn.Module):
    """可变形反卷积模块，结合横向连接特征生成偏移量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # 常规反卷积参数
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=False
        )
        
        # 偏移量生成网络（输入来自浅层特征）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * kernel_size * kernel_size, kernel_size=3, padding=1),
            nn.Tanh()  # 限制偏移量范围
        )
        
        # 可变形卷积权重（复用反卷积权重）
        self.weight = self.trans_conv.weight

    def forward(self, x, lateral_feat):
        # 第一步：常规反卷积上采样
        x = self.trans_conv(x)
        
        # 第二步：基于横向连接特征生成偏移量
        offset = self.offset_conv(lateral_feat)
        
        # 第三步：应用可变形卷积进行特征校准
        N, C, H, W = x.size()
        offset = offset.repeat(1, 1, H, W)  # 扩展偏移量到完整特征图
        
        return deform_conv2d(
            x, offset, self.weight, 
            padding=self.padding,
            stride=1  # 保持空间尺寸
        )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 enable_lateral=False,
                 use_cbam=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # 主分支卷积层
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.dwa = DynamicWeightAggregationBlock(planes)

        # 下采样分支
        self.downsample = downsample
        self.stride = stride

        # 只在非下采样块中使用注意力
        if use_cbam and stride == 1:
            self.cbam = EnhancedCBAM(planes)
        else:
            self.cbam = None

        # 横向连接系统
        self.enable_lateral = enable_lateral
        if self.enable_lateral:
            # 可变形反卷积模块
            self.deform_deconv = DeformableTransposedConv(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=2  # 上采样比例
            )
            
            # 特征融合模块
            self.lateral_fusion = nn.Sequential(
                nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, lateral_feat=None):
        identity = x

        # 主分支前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dwa(out)  # 添加动态权重聚合

        # 横向连接处理
        if self.enable_lateral and lateral_feat is not None:
            # 使用可变形反卷积进行空间对齐
            aligned_feat = self.deform_deconv(out, lateral_feat)
            
            # 特征融合（拼接+融合卷积）
            fused_feat = torch.cat([aligned_feat, lateral_feat], dim=1)
            out = self.lateral_fusion(fused_feat)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        # 在残差连接前加入注意力机制
        if self.cbam is not None:
            out = self.cbam(out)
        

        out += identity
        out = self.relu(out)

        return out


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64')
#         # if dilation > 1:
#         #     raise NotImplementedError(
#         #         "Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes, dilation=dilation)
#         self.bn2 = norm_layer(planes)
      
#         # # 添加 ECBAM注意力机制
#         # self.ecbam = EnhancedAttention(planes)

#         self.downsample = downsample
#         self.stride = stride

       

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         # out = self.ecbam(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)

       

#         out += identity
#         out = self.relu(out)

#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#蛇形1
class DynamicSerpentineConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DynamicSerpentineConv, self).__init__()
        # x 方向卷积
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                stride=stride, padding=(padding, 0), dilation=dilation, bias=False)
        # y 方向卷积
        self.conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                stride=stride, padding=(0, padding), dilation=dilation, bias=False)
        # 正常 3x3 卷积
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=False)
        # 融合卷积
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False)

    def forward(self, x):
        # x 方向卷积
        out_x = self.conv_x(x)
        # y 方向卷积
        out_y = self.conv_y(x)
        # 正常 3x3 卷积
        out_3x3 = self.conv_3x3(x)
        # 拼接特征图
        out = torch.cat([out_x, out_y, out_3x3], dim=1)
        # 融合特征图
        out = self.fusion_conv(out)
        return out

class BasicBlockWithDSC(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlockWithDSC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        # 动态蛇形卷积层
        self.conv1 = DynamicSerpentineConv(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        # # 添加 SE 注意力模块
        # self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # # 应用 SE 注意力模块
        # out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#动态蛇形2
class DynamicSnakeDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DynamicSnakeDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 正常的 3x3 卷积
        self.conv_normal = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, bias=False)

        # 用于生成偏移量的卷积层
        self.offset_conv_x = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=3, stride=stride,
                                       padding=1, bias=False)
        self.offset_conv_y = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=3, stride=stride,
                                       padding=1, bias=False)

        # 可变形卷积层
        self.deform_conv_x = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=False)
        self.deform_conv_y = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=False)

        # 融合卷积层
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # 正常的 3x3 卷积
        out_normal = self.conv_normal(x)

        # 生成 x 和 y 方向的偏移量
        offset_x = self.offset_conv_x(x)
        offset_y = self.offset_conv_y(x)

        # 应用可变形卷积
        # 简单示例：直接使用偏移量
        offset = torch.cat([offset_x, offset_y], dim=1)
        out_x = self.deform_conv_x(x, offset)
        out_y = self.deform_conv_y(x, offset)

        # 拼接特征图
        out = torch.cat([out_normal, out_x, out_y], dim=1)

        # 融合特征图
        out = self.fusion_conv(out)

        return out

class BasicBlockwithDSC(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlockwithDSC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError(
        #         "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = DynamicSnakeDeformConv2d(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DynamicSnakeDeformConv2d(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        # 添加 ECBAM注意力机制
        self.ecbam = EnhancedAttention(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ecbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 多向动态蛇形3
# 定义蛇形卷积
class SnakeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SnakeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        return self.conv(x)
# 定义 4 个方向的蛇形卷积组
class MultiDirectionSnakeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(MultiDirectionSnakeConv, self).__init__()
        self.conv_0 = SnakeConv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_45 = SnakeConv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_90 = SnakeConv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_135 = SnakeConv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out_0 = self.conv_0(x)
        out_45 = self.conv_45(x)
        out_90 = self.conv_90(x)
        out_135 = self.conv_135(x)

        # 计算注意力权重
        attention_scores = torch.cat([out_0.mean(dim=(2, 3), keepdim=True),
                                      out_45.mean(dim=(2, 3), keepdim=True),
                                      out_90.mean(dim=(2, 3), keepdim=True),
                                      out_135.mean(dim=(2, 3), keepdim=True)], dim=1)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=1)

        # 加权求和
        out = attention_weights[:, 0:1, :, :] * out_0 + \
              attention_weights[:, 1:2, :, :] * out_45 + \
              attention_weights[:, 2:3, :, :] * out_90 + \
              attention_weights[:, 3:4, :, :] * out_135

        return out
# 定义可变形反卷积
class DeformableDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DeformableDeconv, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out

class MultiDirectionSnakeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(MultiDirectionSnakeBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        
        # 多向动态蛇形卷积
        self.conv1 = MultiDirectionSnakeConv(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = MultiDirectionSnakeConv(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        
        # 添加 ECBAM注意力机制
        self.ecbam = EnhancedAttention(planes)
        # # 添加 CBAM注意力机制
        # self.cbam = CBAM(planes)
        
        self.downsample = downsample
        self.stride = stride
        # 可变形反卷积
        self.deform_deconv = DeformableDeconv(planes, planes)

    def forward(self, x, shallow_feature=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ecbam(out)
        # out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 横向连接
        if shallow_feature is not None:
            shallow_feature = self.deform_deconv(shallow_feature)
            out += shallow_feature

        out += identity
        out = self.relu(out)

        return out

#多向动态蛇形4
class EfficientSnakeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=4):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.group_conv = nn.Conv2d(
            in_channels, 
            out_channels*4,  # 输出四组特征
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups  # 参数共享
        )
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 方向偏移初始化
        self._initialize_direction_offsets(kernel_size)

    def _initialize_direction_offsets(self, kernel_size):
        # 定义四个方向的采样偏移
        offsets = torch.zeros(4, 2, kernel_size, kernel_size)
        
        # 水平方向
        offsets[0, 0, :, :] = torch.linspace(-1, 1, kernel_size).view(1, -1)
        # 垂直方向
        offsets[1, 1, :, :] = torch.linspace(-1, 1, kernel_size).view(-1, 1)
        # 对角线1
        diag = torch.linspace(-1, 1, kernel_size)
        offsets[2, 0, :, :] = diag.view(1, -1)
        offsets[2, 1, :, :] = diag.view(-1, 1)
        # 对角线2
        offsets[3, 0, :, :] = diag.view(1, -1)
        offsets[3, 1, :, :] = -diag.view(-1, 1)
        
        self.register_buffer('offsets', offsets.repeat_interleave(3, dim=0))

    def forward(self, x):
        # 分组卷积提取基础特征
        group_feat = self.group_conv(x)
        b, c, h, w = group_feat.shape
        
        # 应用方向偏移
        sampled_feats = []
        for i in range(4):
            feat = F.grid_sample(
                group_feat[:, i*(c//4):(i+1)*(c//4)], 
                self.offsets[i*3:(i+1)*3].unsqueeze(0).expand(b, -1, h, w),
                padding_mode='zeros',
                align_corners=False
            )
            sampled_feats.append(feat)
        
        # 动态特征融合
        attention = torch.cat([f.mean(dim=(2,3), keepdim=True) for f in sampled_feats], dim=1)
        attention = F.softmax(attention / self.temperature, dim=1)
        
        return sum([a * f for a, f in zip(attention.chunk(4,1), sampled_feats)])

class LightweightAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        return x * ca * sa

class EfficientDeformDeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 轻量级偏移生成
        self.offset_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 18, 3, padding=1)
        )
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        offset = self.offset_gen(x)
        return self.deform_conv(x, offset)

class OptimizedSnakeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('OptimizedSnakeBlock only supports groups=1 and base_width=64')

        # 计算padding以保持空间尺寸
        padding = dilation

        # 主干卷积
        self.conv1 = EfficientSnakeConv(inplanes, planes, stride=stride, dilation=dilation,  padding=padding)
        self.bn1 = norm_layer(planes)
        
        # 深度可分离残差路径
        self.dw_conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=dilation, dilation=dilation, groups=planes),
            norm_layer(planes),
            nn.ReLU(inplace=True)
        )
        self.pw_conv = nn.Conv2d(planes, planes, 1)
        
        # 注意力机制
        self.attention = LightweightAttention(planes)
        
        # 下采样
        self.downsample = downsample
        self.stride = stride
        
        # 可变形上采样
        self.deform_up = EfficientDeformDeconv(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shallow_feature=None):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 轻量残差
        residual = self.dw_conv(out)
        residual = self.pw_conv(residual)
        
        # 特征融合
        out = self.attention(out + residual)
        
        # 横向连接
        if shallow_feature is not None:
            out += self.deform_up(shallow_feature)
            
        # 跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

#通过动态地聚合多个分支的输出，可以提高网络的适应性和泛化能力。
class DynamicWeightAggregationBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(DynamicWeightAggregationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlockWithDWA(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDWA, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.dwa = DynamicWeightAggregationBlock(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dwa(out)  # 添加动态权重聚合
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

#通道注意力机制
class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力机制
class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(x.size(0), -1, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class EnhancedAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)  # 深度可分离卷积减少参数量
        self.channel_att = EnhancedChannelAttention(channels, reduction_ratio)
        self.spatial_att = EnhancedSpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.spatial_conv(x)
        spatial_att = self.spatial_att(x)
        channel_att = self.channel_att(x, spatial_att)  # 显式传递空间注意力
        att = channel_att * spatial_att
        return x * att

class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction_ratio),  # 输入包含空间注意力信息
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x, spatial_att):
        avg_out = self.avg_pool(x * spatial_att).view(x.size(0), -1)  # 空间注意力加权后的特征
        max_out = self.max_pool(x * spatial_att).view(x.size(0), -1)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.fc(combined).view(x.size(0), -1, 1, 1)

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))

class EnhancedCBAM(nn.Module):
    """
    增强版CBAM模块，修复通道不匹配问题
    """
    def __init__(self, channels, reduction_ratio=16, kernel_sizes=[3, 5, 7]):
        super(EnhancedCBAM, self).__init__()
        self.channel_attention = EnhancedChannelAttention2(channels, reduction_ratio)
        self.spatial_attention = EnhancedSpatialAttention2(channels, kernel_sizes)
        self.coord_attention = CoordAtt(channels, reduction_ratio)
        
        # 修复1: 在__init__中定义融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels*3, kernel_size=1, groups=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels*3, channels*3, kernel_size=1)
        )
        
    def forward(self, x):
        # 并行处理三种注意力机制
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        coord_out = self.coord_attention(x)
        
        # 加权融合
        fused = torch.cat([channel_out, spatial_out, coord_out], dim=1)
        weights = F.adaptive_avg_pool2d(fused, 1)
        weights = self.fusion_conv(weights)
        
        # 修复2: 正确的权重分割和softmax计算
        weights = weights.view(weights.size(0), 3, -1)
        weights = torch.softmax(weights, dim=1)
        weights = weights.view(weights.size(0), -1, 1, 1)
        
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        
        # 残差连接
        out = x + w1 * channel_out + w2 * spatial_out + w3 * coord_out
        return out

class EnhancedChannelAttention2(nn.Module):
    """修复的通道注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super(EnhancedChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用1x1卷积代替全连接层
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 标准分支
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        
        # 特征融合
        out = avg_out + max_out
        return x * self.sigmoid(out)

class EnhancedSpatialAttention2(nn.Module):
    """修复的空间注意力模块"""
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super(EnhancedSpatialAttention2, self).__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            # 修复3: 添加缺失的括号
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=k, padding=padding, bias=False),
                    nn.BatchNorm2d(1)
                )  # 添加缺失的括号
            )
        
        # 车道线特征增强
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(1)
        )
        self.vertical_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(15, 1), padding=(7, 0), bias=False),
            nn.BatchNorm2d(1)
        )
        
        # 修复4: 调整融合卷积的输入通道数
        self.conv_fuse = nn.Conv2d(len(kernel_sizes) + 2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道压缩
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compress = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度特征提取
        scale_features = []
        for conv in self.convs:
            scale_features.append(conv(x_compress))
        
        # 车道线特征增强
        horizontal_feat = self.horizontal_conv(avg_out)
        vertical_feat = self.vertical_conv(avg_out)
        
        # 特征融合
        all_features = torch.cat(scale_features + [horizontal_feat, vertical_feat], dim=1)
        fused = self.conv_fuse(all_features)
        
        return x * self.sigmoid(fused)

class CoordAtt(nn.Module):
    """修复的坐标注意力机制"""
    def __init__(self, inp, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 坐标信息嵌入
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x)  # [n, c, 1, w]
        
        # 修复5: 正确的特征融合方式
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, w, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 调整维度回[n, c, 1, w]
        
        # 注意力图生成
        a_h = self.conv_h(x_h).sigmoid()  # [n, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [n, c, 1, w]
        
        # 修复6: 正确的注意力图扩展方式
        a_h = a_h.expand(-1, -1, -1, w)
        a_w = a_w.expand(-1, -1, h, -1)
        
        return identity * a_h * a_w

class ECABasicBlock(nn.Module):
    """修复的ResNet BasicBlock集成"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        # 修复7: 只在非下采样块中使用注意力
        if use_cbam and stride == 1:
            self.cbam = EnhancedCBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 在残差连接前加入注意力机制
        if self.cbam is not None:
            out = self.cbam(out)
        
        out += identity
        out = self.relu(out)

        return out

@BACKBONES.register_module
class ResNetWrapper(nn.Module):
    def __init__(self,
                 resnet='resnet18',
                 pretrained=True,
                 replace_stride_with_dilation=[False, False, False],
                 out_conv=False,
                 fea_stride=8,
                 out_channel=128,
                 in_channels=[64, 128, 256, 512],
                 cfg=None):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            in_channels=self.in_channels)
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = conv1x1(out_channel * self.model.expansion,
                               cfg.featuremap_out_channel)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 in_channels=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = in_channels
        self.layer1 = self._make_layer(block, in_channels[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       in_channels[1],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       in_channels[2],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(
                block,
                in_channels[3],
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2])
        self.expansion = block.expansion

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_layers = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):    # hasattr 函数用于检查对象是否具有特定的属性。具体来说，它用于检查 self 对象是否具有名为 'layer1', 'layer2', 'layer3', 和 'layer4' 的属性。 hasattr(self, name) 用于确保在尝试访问 self.name 之前，self 确实有这个属性。这可以防止在 self 没有某个层的情况下引发 AttributeError。如果某个层不存在，hasattr 会跳过对该层的处理，继续检查下一个层。
                continue
            layer = getattr(self, name)   # getattr 在这里的作用是根据字符串名称动态地获取对象的属性，从而实现动态地访问和使用不同层的功能
            x = layer(x)
            out_layers.append(x)

        return out_layers


class SENet(nn.Module):

    def __init__(self,
                 block, 
                 layers, 
                 num_classes=2,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 in_channels=None):
        
        super(SENet, self).__init__()
        if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
                    # each element in the tuple indicates if we should replace
                    # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                            "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))

        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

       

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化

        # self.att1 = EnhancedAttention(channels=64)  # conv1的输出通道是64

        self.in_channels = in_channels
        self.layer1 = self._make_layer(block, in_channels[0] ,layers[0])
        # self.ecbam1 = EnhancedAttention(in_channels[0] * block.expansion)
       
        self.layer2 = self._make_layer(block, in_channels[1] , layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        # self.ecbam2 = EnhancedAttention(in_channels[1] * block.expansion)
        
        self.layer3 = self._make_layer(block, in_channels[2] , layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.ecbam3 = EnhancedAttention(in_channels[2] * block.expansion)
       
        # self.layer4 = self._make_layer(block, in_channels[3] , layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(
                block,
                in_channels[3],
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2])
            # self.cbam4 = CBAM(in_channels[3] * block.expansion)
            # self.ecbam4 = EnhancedAttention(in_channels[3] * block.expansion)

        # last_channels = self.inplanes  # 自动获取最后一层的输出通道数
        # self.att_last = EnhancedAttention(channels=last_channels)

        self.expansion = block.expansion
        # block：指定使用的基本块类型，可以是残差块或其他类型的块
        # 64、128、256、512：指定每个卷积层组中的通道数（即输出特征图的通道数）
        # layers[0]、layers[1]、layers[2]、layers[3]：指定每个卷积层组中的基本块数量
        # stride=2：指定每个卷积层组中的卷积层的步长（stride），默认为 2
        # self.avgpool = nn.AvgPool2d(7, stride=1)  # 平均池化
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接

      

        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m,  BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(self.inplanes, planes, stride, downsample, self.groups,
                    self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer))

            return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        

        x = self.maxpool(x)
        # x = self.att1(x)  # 应用最前面的注意力模块
 
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)  # 特征提取和降维
        out_layers = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):
                continue
            layer = getattr(self, name)
            x = layer(x)
        
            # ecbam = getattr(self, f'ecbam{name[-1]}')  # 获取对应的CBAM模块
            # x = ecbam(x)  # 串联操作应用CBAM模块
           
            out_layers.append(x)
            
        # # 应用最后的注意力模块并替换最后输出
        # if out_layers:  # 如果存在有效输出层
        #     x = self.att_last(x)
        #     out_layers[-1] = x  # 替换最后一个层的输出为注意力处理后的结果

        return out_layers
        
        # if hasattr(self, 'ecbam'):
        #     x = self.ecbam(x) * x
        # return out_layers
 
        # x = self.avgpool(x)  # 将特征降维为一维
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)  # 全连接层进行分类
        # print(x.shape)
 
        # return x
 

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = SENet(block, layers, **kwargs)
    if pretrained:
        print('pretrained model: ', model_urls[arch])
        # state_dict = torch.load(model_urls[arch])['net']
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',BasicBlock, [2, 2, 2, 2], pretrained, progress, 
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)

 

 
