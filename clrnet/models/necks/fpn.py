import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..registry import NECKS


@NECKS.register_module
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs, # 输出的特征图数量
                 start_level=0, # 从哪个层级开始构建FPN（默认从第0层开始）
                 end_level=-1, # 到哪个层级结束构建FPN（默认到最后一层）
                 add_extra_convs=False, # 是否添加额外的卷积层来增加特征图数量
                 extra_convs_on_inputs=True, # 额外的卷积层是否应用在输入上（默认为True）
                 relu_before_extra_convs=False, # 是否在额外的卷积层之前应用ReLU激活函数
                 no_norm_on_lateral=False, # 是否在横向连接中不使用归一化层
                 conv_cfg=None,  # 卷积层的配置
                 norm_cfg=None, # 归一化层的配置
                 attention=False, # 是否在FPN中使用注意力机制
                 act_cfg=None, # 激活函数的配置
                 upsample_cfg=dict(mode='nearest'), # 上采样方法的配置（默认使用最近邻插值）
                 downsample_cfg=dict(mode='nearest'), # 下采样方法的配置（默认使用最近邻插值）
                 init_cfg=dict(type='Xavier',  # 初始化方法的配置
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels) # 输入特征图的数量等于输入通道列表的长度
        self.num_outs = num_outs
        self.attention = attention
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.downsample_cfg = downsample_cfg.copy()

        if end_level == -1:    # 如果 end_level 为 -1，则 backbone_end_level 等于输入特征图的数量 num_ins
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level 
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs  # 是否添加额外的卷积层
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()  # lateral_convs: 横向连接的卷积层
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):   # 遍历从 start_level 到 backbone_end_level 的每个层级，创建相应的卷积层并添加到 lateral_convs 和 fpn_convs 中
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(out_channels,
                                  out_channels,
                                  3,
                                  padding=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg,
                                  inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)  计算需要添加的额外卷积层的数量 extra_levels
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels,
                                            out_channels,
                                            3,
                                            stride=2,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        # print(f"len(inputs): {len(inputs)}, len(self.in_channels): {len(self.in_channels)}")  # 打印长度信息
        assert len(inputs) >= len(self.in_channels)  # 确保输入特征图的数量 inputs 至少与 in_channels 的数量相同，否则会抛出断言错误。
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]    # 如果输入特征图的数量多于 in_channels，则删除多余的输入特征图。

        # build laterals    lateral_convs 是一个包含多个横向卷积层的列表。每个横向卷积层用于处理对应尺度的输入特征图。 通过遍历 lateral_convs，对每个尺度的输入特征图进行卷积操作，生成横向特征 laterals。
        # for i in range(len(self.lateral_convs)):
        #    print(f"Input shape at level {i + self.start_level}: {inputs[i + self.start_level].shape}")
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path   通过上采样（F.interpolate）将高分辨率的特征图与低分辨率的特征图进行融合。根据 upsample_cfg 中的配置（例如 scale_factor 或 size）进行上采样操作。
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 size=prev_shape,
                                                 **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)   # fpn_convs 是一个包含多个卷积层的列表，用于对横向特征进行进一步处理。对每个尺度的横向特征进行卷积操作，生成原始级别的输出特征图 outs
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):  # 如果需要更多的输出特征图（num_outs 大于当前的输出数量），则通过以下方式添加额外级别：
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2)) # 若无额外卷积，使用最大池化（F.max_pool2d）来生成额外级别。
            # add conv layers on top of original feature maps (RetinaNet)
            else:   # 若有额外卷积，根据配置 (on_input, on_lateral, on_output) 选择不同的输入源，并通过卷积层生成额外级别的特征图。
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
