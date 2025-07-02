import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from .fpn import FPN
from ..registry import NECKS

@NECKS.register_module
class FPN_P(FPN):
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
            super(FPN_P, self).__init__(in_channels,
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
            # add extra bottom up pathway   添加额外的自底向上的路径
            self.downsample_convs = nn.ModuleList()  #用于下采样的卷积层列表。
            self.upsample_convs = nn.ModuleList()    #用于上采样的卷积层列表。
            self.fusion_convs = nn.ModuleList()  #用于进一步处理特征的卷积层列表
            for i in range(self.start_level + 1, self.backbone_end_level):  #通过循环从 start_level + 1 到 backbone_end_level，为每个特征层添加下采样卷积和特征处理卷积。
                d_conv = ConvModule(out_channels,
                                    out_channels,
                                    3,
                                    stride=2,
                                    padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    inplace=False)
                # 上采样卷积
                u_conv = ConvModule(out_channels,
                                    out_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    inplace=False)
            
                b_conv = ConvModule(out_channels,
                                        out_channels,
                                        3,
                                        padding=1,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg,
                                        inplace=False)

                self.downsample_convs.append(d_conv)
                self.upsample_convs.append(u_conv)
                self.fusion_convs.append(b_conv)


    def forward(self, inputs):
        # 确保输入特征图的数量 inputs 至少与 in_channels 的数量相同
        assert len(inputs) >= len(self.in_channels)
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # 构建横向特征
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自顶向下路径（Top-Down Path）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        # 自底向上路径（Bottom-Up Path）
        for i in range(1, used_backbone_levels):
            if 'scale_factor' in self.downsample_cfg:
                laterals[i] += F.interpolate(laterals[i - 1], **self.downsample_cfg)
            else:
                next_shape = laterals[i].shape[2:]
                laterals[i] += F.interpolate(laterals[i - 1], size=next_shape, **self.downsample_cfg)

        # 构建输出特征
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