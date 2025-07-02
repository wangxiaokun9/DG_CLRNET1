import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..registry import NECKS
from .fpn import FPN


@NECKS.register_module
class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """
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
        super(PAFPN, self).__init__(in_channels,
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
        self.pafpn_convs = nn.ModuleList()   #用于进一步处理特征的卷积层列表
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
            pafpn_conv = ConvModule(out_channels,
                                    out_channels,
                                    3,
                                    padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)  #确保输入的特征图数量大于等于 in_channels 的长度。

        if len(inputs) > len(self.in_channels):   # 如果输入特征图的数量多于 in_channels 的数量，删除多余的输入特征图
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals   #构建横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])    #使用横向卷积（lateral_convs）对输入特征图进行处理，生成横向特征图
            for i, lateral_conv in enumerate(self.lateral_convs) 
        ]

        # build top-down path    从最高分辨率的特征图开始，逐层向上插值并与上一层的特征图相加，构建自顶向下的特征金字塔。
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i],
                                             size=prev_shape,
                                             mode='nearest')

        # build outputs  构建输出特征图
        # part 1: from original levels  
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)  #使用 fpn_convs 对横向特征图进行进一步处理，生成中间输出特征图。
        ]

        # part 2: add bottom-up path 添加自底向上的路径
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])   #通过 downsample_convs 对中间输出特征图进行下采样，并与下一层的特征图相加，构建自底向上的特征路径。
        
        # 构建最终输出
        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)     #将最底层的特征图和经过 pafpn_convs 处理的特征图组合成最终的输出特征图
        ])

        # part 3: add extra levels  添加额外的特征层
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs  使用最大池化生成更多的特征层
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)  在原始特征图上添加卷积层
            else:     #如果需要的输出特征层数量大于当前生成的数量，可以通过最大池化或添加额外卷积层来生成更多的特征层。
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
