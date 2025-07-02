import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..registry import NECKS
from .fpn import FPN

@NECKS.register_module
class BiFPN(FPN):
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
            super(BiFPN, self).__init__(in_channels,
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
            # add extra bottom up pathway   添加额外的自底向上的路径
            self.downsample_convs = nn.ModuleList()  #用于下采样的卷积层列表。
            self.upsample_convs = nn.ModuleList()    #用于上采样的卷积层列表。
            self.Bifpn_convs = nn.ModuleList()   #用于进一步处理特征的卷积层列表
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
                bifpn_conv = ConvModule(out_channels,
                                        out_channels,
                                        3,
                                        padding=1,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg,
                                        inplace=False)
                self.downsample_convs.append(d_conv)
                self.upsample_convs.append(u_conv)
                self.Bifpn_convs.append(bifpn_conv)
                
                


    # def __init__(self,
    #              in_channels,
    #              out_channels,
    #              num_outs,
    #              start_level=0,
    #              end_level=-1,
    #              add_extra_convs=False,
    #              extra_convs_on_inputs=True,
    #              relu_before_extra_convs=False,
    #              no_norm_on_lateral=False,
    #              conv_cfg=None,
    #              norm_cfg=None,
    #              act_cfg=None,
    #              cfg=None,
    #              attention=False,
    #              upsample_cfg=dict(mode='nearest'),):
    #     super(BiFPN, self).__init__()
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.num_ins = len(in_channels) # 输入特征图的数量等于输入通道列表的长度
    #     self.num_outs = num_outs
    #     self.attention = attention
    #     self.relu_before_extra_convs = relu_before_extra_convs
    #     self.no_norm_on_lateral = no_norm_on_lateral
    #     self.upsample_cfg = upsample_cfg.copy()

    #     self.stages = nn.ModuleList()
    #     for i in range(len(in_channels)):
    #         stage = nn.Sequential(
    #             nn.Conv2d(in_channels[i], out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             CSP(out_channels, num_repeat=2),
    #         )
    #         self.stages.append(stage)
 
    #     self.lateral_connections = nn.ModuleList()
    #     for i in range(len(in_channels) - 1):
    #         lateral_connection = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
    #         self.lateral_connections.append(lateral_connection)
 
    #     self.top_down_connections = nn.ModuleList()
    #     for i in range(len(in_channels) - 1):
    #         top_down_connection = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    #         self.top_down_connections.append(top_down_connection)
 
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
 # 保留原有 FPN 的部分代码，如 lateral 连接构建
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

        # build bottom-up path
        for i in range(0, used_backbone_levels - 1):
            laterals[i + 1] += F.interpolate(laterals[i], size=laterals[i + 1].shape[2:], **self.upsample_cfg)
            
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
 
        # outputs = []
        # for i in range(len(features)):
        #     stage_output = self.stages[i](features[i])
        #     if i != 0:
        #         lateral_connection = self.lateral_connections[i - 1](features[i - 1])
        #         stage_output = stage_output + lateral_connection
 
        #     if i != len(features) - 1:
        #         top_down_connection = self.top_down_connections[i](features[i + 1])
        #         stage_output = stage_output + top_down_connection
 
        #     outputs.append(stage_output)
 
        # return outputs