import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from mmcv.cnn import ConvModule
from ..registry import NECKS
from .fpn import FPN

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

class DeformableDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DeformableDeconv, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out


   
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
 
class ECBAM(nn.Module):
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

   
@NECKS.register_module    
class FPN_E(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 cfg=None,
                 attention=False):
        super(FPN_E, self).__init__(in_channels,
                                    out_channels,
                                    num_outs,
                                    start_level,
                                    end_level,
                                    add_extra_convs,
                                    extra_convs_on_inputs,
                                    relu_before_extra_convs,
                                    no_norm_on_lateral,
                                    conv_cfg,
                                    norm_cfg,
                                    attention,
                                    act_cfg,
                                    cfg=cfg)
        
        # 添加额外的自底向上的路径
        self.downsample_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        self.fpn_l_convs = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        
        # 使用多向动态蛇形卷积替换普通卷积
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = MultiDirectionSnakeConv(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            u_conv = MultiDirectionSnakeConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            fpn_l_conv = MultiDirectionSnakeConv(out_channels, out_channels, kernel_size=3, padding=1)
            b_conv = MultiDirectionSnakeConv(out_channels, out_channels, kernel_size=3, padding=1)
            
            self.downsample_convs.append(d_conv)
            self.upsample_convs.append(u_conv)
            self.fpn_l_convs.append(fpn_l_conv)
            self.bottom_up_convs.append(b_conv)
        
        # 引入注意力机制
        if attention:
            self.attention_modules = nn.ModuleList()
            for _ in range(self.backbone_end_level - self.start_level):
                self.attention_modules.append(CBAM(out_channels))
                # self.attention_modules.append(ECBAM(out_channels))

    def forward(self, inputs):
        assert len(inputs) >= len(self.in_channels)
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        # apply attention if needed
        if hasattr(self, 'attention_modules'):
            laterals = [self.attention_modules[i](laterals[i]) for i in range(used_backbone_levels)]

        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # build bottom-up path (path augmentation)
        for i in range(0, used_backbone_levels - 1):
            outs[i + 1] += self.bottom_up_convs[i](F.interpolate(outs[i], scale_factor=0.5))

        return tuple(outs)