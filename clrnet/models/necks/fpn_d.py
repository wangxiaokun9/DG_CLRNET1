import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from .fpn import FPN
from ..registry import NECKS





class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFeatureFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = x1 + x2
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x1 * y + x2 * (1 - y)

class ContextEnhancementModule(nn.Module):
    def __init__(self, channels):
        super(ContextEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        attention = self.attention(x)
        return residual + x * attention

@NECKS.register_module
class FPN_D(FPN):
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
        super(FPN_D, self).__init__(in_channels,
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
                                    cfg=cfg,
                                    )
        # 添加自适应特征融合模块
        self.aff_modules = nn.ModuleList()
        # 添加上下文增强模块
        self.cem_modules = nn.ModuleList()
         # 确保长度与 laterals 的长度一致
        for i in range(len(self.lateral_convs)):
            aff = AdaptiveFeatureFusion(out_channels)
            cem = ContextEnhancementModule(out_channels)
            self.aff_modules.append(aff)
            self.cem_modules.append(cem)
def forward(self, inputs):
    assert len(inputs) >= len(self.in_channels)
    if len(inputs) > len(self.in_channels):
        for _ in range(len(inputs) - len(self.in_channels)):
            del inputs[0]

    laterals = [
        lateral_conv(inputs[i + self.start_level])
        for i, lateral_conv in enumerate(self.lateral_convs)
    ]

    used_backbone_levels = len(laterals)
    for i in range(used_backbone_levels - 1, 0, -1):
        if 'scale_factor' in self.upsample_cfg:
            laterals[i - 1] = self.aff_modules[i - 1](laterals[i - 1], F.interpolate(laterals[i], **self.upsample_cfg))
        else:
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.aff_modules[i - 1](laterals[i - 1], F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg))

    # 自底向上路径（Bottom-Up Path）
    for i in range(1, used_backbone_levels):
        if 'scale_factor' in self.downsample_cfg:
            laterals[i] = self.aff_modules[i - 1](laterals[i], F.interpolate(laterals[i - 1], **self.downsample_cfg))
        else:
            next_shape = laterals[i].shape[2:]
            laterals[i] = self.aff_modules[i - 1](laterals[i], F.interpolate(laterals[i - 1], size=next_shape, **self.downsample_cfg))

    # 应用上下文增强模块
    for i in range(used_backbone_levels):
        laterals[i] = self.cem_modules[i](laterals[i])

    outs = [
        self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
    ]

    # 添加额外级别
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
    return tuple(outs)