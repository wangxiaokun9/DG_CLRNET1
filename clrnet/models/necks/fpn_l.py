import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..registry import NECKS
from .fpn import FPN

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
class FPN_L(FPN):
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
        super(FPN_L, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            extra_convs_on_inputs=extra_convs_on_inputs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            cfg=cfg
        )
        # 设置双线性插值参数
        self.upsample_cfg = dict(mode='bilinear', align_corners=False)
        
        # 初始化其他模块
        self.downsample_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        self.fpn_l_convs = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            u_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            fpn_l_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            b_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            self.downsample_convs.append(d_conv)
            self.upsample_convs.append(u_conv)
            self.fpn_l_convs.append(fpn_l_conv)
            self.bottom_up_convs.append(b_conv)


        # 引入注意力机制
        if attention:
            self.attention_modules = nn.ModuleList()
            for _ in range(self.backbone_end_level - self.start_level):
                self.attention_modules.append(CBAM(out_channels))
                


    def forward(self, inputs):
        assert len(inputs) >= len(self.in_channels)
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # 构建侧边特征
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自顶向下融合特征
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # 双线性插值调整尺寸
            interpolated = F.interpolate(
                laterals[i], 
                size=prev_shape, 
                **self.upsample_cfg
            )
            # 单位加融合
            laterals[i - 1] += interpolated

        # apply attention if needed
        if hasattr(self, 'attention_modules'):
            laterals = [self.attention_modules[i](laterals[i]) for i in range(used_backbone_levels)]

        # 构建输出特征
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 处理额外输出层
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
        # 自底向上增强路径
        for i in range(0, used_backbone_levels - 1):
            outs[i + 1] += self.bottom_up_convs[i](F.interpolate(
                outs[i], 
                scale_factor=0.5, 
                **self.upsample_cfg  # 使用相同的插值配置
            ))

        return tuple(outs)


# @NECKS.register_module
# class FPN_L(FPN):
#     def __init__(self,
#                     in_channels,
#                     out_channels,
#                     num_outs,
#                     start_level=0,
#                     end_level=-1,
#                     add_extra_convs=False,
#                     extra_convs_on_inputs=True,
#                     relu_before_extra_convs=False,
#                     no_norm_on_lateral=False,
#                     conv_cfg=None,
#                     norm_cfg=None,
#                     act_cfg=None,
#                     cfg=None,
#                     attention=False):
#             super(FPN_L, self).__init__(in_channels,
#                                         out_channels,
#                                         num_outs,
#                                         start_level,
#                                         end_level,
#                                         add_extra_convs,
#                                         extra_convs_on_inputs,
#                                         relu_before_extra_convs,
#                                         no_norm_on_lateral,
#                                         conv_cfg,
#                                         norm_cfg,
#                                         attention,
#                                         act_cfg,
#                                         cfg=cfg)
#             # add extra bottom up pathway   添加额外的自底向上的路径
#             self.downsample_convs = nn.ModuleList()  #用于下采样的卷积层列表。
#             self.upsample_convs = nn.ModuleList()    #用于上采样的卷积层列表。
#             self.fpn_l_convs = nn.ModuleList()   #用于进一步处理特征的卷积层列表
#             self.bottom_up_convs = nn.ModuleList()
#             for i in range(self.start_level + 1, self.backbone_end_level):  #通过循环从 start_level + 1 到 backbone_end_level，为每个特征层添加下采样卷积和特征处理卷积。
#                 d_conv = ConvModule(out_channels,
#                                     out_channels,
#                                     3,
#                                     stride=2,
#                                     padding=1,
#                                     conv_cfg=conv_cfg,
#                                     norm_cfg=norm_cfg,
#                                     act_cfg=act_cfg,
#                                     inplace=False)
#                 # 上采样卷积
#                 u_conv = ConvModule(out_channels,
#                                     out_channels,
#                                     3,
#                                     stride=1,
#                                     padding=1,
#                                     conv_cfg=conv_cfg,
#                                     norm_cfg=norm_cfg,
#                                     act_cfg=act_cfg,
#                                     inplace=False)
#                 fpn_l_conv = ConvModule(out_channels,
#                                         out_channels,
#                                         3,
#                                         padding=1,
#                                         conv_cfg=conv_cfg,
#                                         norm_cfg=norm_cfg,
#                                         act_cfg=act_cfg,
#                                         inplace=False)
#                 b_conv = ConvModule(out_channels,
#                                         out_channels,
#                                         3,
#                                         padding=1,
#                                         conv_cfg=conv_cfg,
#                                         norm_cfg=norm_cfg,
#                                         act_cfg=act_cfg,
#                                         inplace=False)

#                 self.downsample_convs.append(d_conv)
#                 self.upsample_convs.append(u_conv)
#                 self.fpn_l_convs.append(fpn_l_conv)
#                 self.bottom_up_convs.append(b_conv)


#     def forward(self, inputs):
#         # print(f"len(inputs): {len(inputs)}, len(self.in_channels): {len(self.in_channels)}")  # 打印长度信息
#         assert len(inputs) >= len(self.in_channels)  # 确保输入特征图的数量 inputs 至少与 in_channels 的数量相同，否则会抛出断言错误。
#         if len(inputs) > len(self.in_channels):
#             for _ in range(len(inputs) - len(self.in_channels)):
#                 del inputs[0]    # 如果输入特征图的数量多于 in_channels，则删除多余的输入特征图。

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             if 'scale_factor' in self.upsample_cfg:
#                 laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
#             else:
#                 prev_shape = laterals[i - 1].shape[2:]
#                 laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

#         # build outputs
#         # part 1: from original levels
#         outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]

#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     extra_source = inputs[self.backbone_end_level - 1]
#                 elif self.add_extra_convs == 'on_lateral':
#                     extra_source = laterals[-1]
#                 elif self.add_extra_convs == 'on_output':
#                     extra_source = outs[-1]
#                 else:
#                     raise NotImplementedError
#                 outs.append(self.fpn_convs[used_backbone_levels](extra_source))
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))

#         # build bottom-up path (path augmentation)
#         for i in range(0, used_backbone_levels - 1):
#             outs[i + 1] += self.bottom_up_convs[i](F.interpolate(outs[i], scale_factor=0.5))

#         return tuple(outs)