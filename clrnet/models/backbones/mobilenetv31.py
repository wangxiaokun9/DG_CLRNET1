from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
import torch.nn.functional as F
from os.path import join
import torch.utils.model_zoo as model_zoo
from clrnet.models.registry import BACKBONES


__all__ = ['MobileNetV3_large', 'mobilenetv3_large']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV3Wrapper_large(nn.Module):
    def __init__(self,
                 mobilenet_large='mobilenetv3_large',
                 in_channels=[24,40,80,112,160,576],
                 pretrained=True,
                 cfg=None):
        super(MobileNetV3Wrapper_large, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.model = eval(mobilenet_large)(
            pretrained=pretrained
            )
       

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNetV3_large(nn.Module):
    def __init__(self, input_size=224, mode='large', width_mult=1.0):   # n_class=1000 n_class：模型的输出类别数量，默认是 1000。 input_size：输入图像的大小，默认是 224x224。 dropout=0.8 dropout：dropout 比例，默认是 0.8。 mode：模型的大小，可以是 ‘large’ 或 ‘small’，默认是 ‘small’。 width_mult：宽度乘数，用于调整模型的宽度，默认是 1.0。
        super(MobileNetV3_large, self).__init__()
        input_channel = 16   # 第一个卷积层的输出通道数，默认为 16。
        last_channel = 576  #最后一层的输出通道数，默认为 1280。
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,   k：可能是卷积核的大小（kernel size）。 exp：扩张通道数。 c：输出通道数。 se：代表是否使用SE（Squeeze-and-Excitation）模块，一个用于增强模型性能的注意力机制。True表示使用，False表示不使用。 nl：非线性函数的类型。例如，'RE’可能代表ReLU（Rectified Linear Unit）激活函数，而’HS’可能代表Hswish激活函数。 s：步长（stride），用于卷积操作中的下采样
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0    # 确保输入图片的尺寸是 32 的倍数。因为许多卷积神经网络在设计上需要输入图片的尺寸是某个特定值的倍数（通常是 32 的倍数），以确保网络的层数和步幅等参数设置合理
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel   # 根据 width_mult 调整 last_channel 的值。width_mult 是一个缩放因子，用于调整网络的宽度。如果 width_mult 大于 1.0，则使用 make_divisible 函数确保 last_channel 的值是某个倍数的整数。如果 width_mult 小于或等于 1.0，则 last_channel 保持不变。
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]  # 定义了模型的第一层 conv_bn 是一个函数，用于创建一个卷积层并跟随一个批量归一化层。这里使用了 3 个输入通道（通常是 RGB 图像），input_channel 是输出通道数，步幅为 2。nlin_layer=Hswish 表示使用 Hswish 作为非线性激活函数
        # self.classifier = []   # 初始化了一个空的分类器层列表

        # building mobile blocks   # 用于构建多个 MobileBottleneck 模块。mobile_setting 是一个列表，包含了多个参数元组，每个元组定义了一个 MobileBottleneck 模块的参数
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)  # 使用了 make_divisible 函数，确保 last_conv 的值是某个倍数的整数。960 是原始的卷积通道数，width_mult 是一个缩放因子，用于调整网络的宽度。
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)) # 添加了一个 1x1 的卷积层，后面跟着批量归一化和 Hswish 激活函数。Hswish 是一种非线性激活函数，类似于 ReLU，但在输入为负值时有更好的表现。
            self.features.append(nn.AdaptiveAvgPool2d(1)) # 这行代码添加了一个自适应全局平均池化层，将输入特征图的大小调整为 1x1，即对每个通道进行全局平均池化。
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0)) # 这行代码添加了一个 1x1 的卷积层，将 last_conv 个通道转换为 last_channel 个输出通道
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            # self.features.append(nn.AdaptiveAvgPool2d(1))
            # self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            # self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)  # 这一行代码将 self.features 中的模块组合成一个顺序结构的神经网络。
        self.part0 = nn.Sequential(*self.features[:1])
        self.part1 = nn.Sequential(*self.features[1:4])
        self.part2 = nn.Sequential(*self.features[4:11])
        self.part3 = nn.Sequential(*self.features[11:])
        # # # building classifier
        # self.classifier = nn.Sequential(   # self.classifier 是一个由 Dropout 和 Linear 层组成的分类器，也被封装在 nn.Sequential 中。
        #     nn.Dropout(p=dropout),    # refer to paper section 6
        #     nn.Linear(last_channel, n_class),
        #     print(last_channel, n_class)
        # )

        self._initialize_weights()
       

    def forward(self, x):
        # y = []
        # x = self.features(x)
        x = self.part0(x)
        # x = self.part1(x)
        # x = self.part2(x)
        # x = self.part3(x)
        out_layers = []
        for name in ['part1', 'part2', 'part3']:
            if not hasattr(self, name):    # hasattr 函数用于检查对象是否具有特定的属性 hasattr(self, name) 用于确保在尝试访问 self.name 之前，self 确实有这个属性。这可以防止在 self 没有某个层的情况下引发 AttributeError。如果某个层不存在，hasattr 会跳过对该层的处理，继续检查下一个层。
                continue
            layer = getattr(self, name)   # getattr 在这里的作用是根据字符串名称动态地获取对象的属性，从而实现动态地访问和使用不同层的功能
            x = layer(x)
            out_layers.append(x)

        return out_layers
        # x = x.mean(3).mean(2) # 对输入张量 x 进行两次平均池化操作。    x.mean(3)：表示对 x 的最后一个维度 width 进行平均池化     x.mean(2)：表示对 x 的倒数第二个维度 height 进行平均池化
        # x = self.classifier(x)
        # y.append(x)
      
              

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

       
def mobilenetv3_large(pretrained=True, **kwargs):
    model = MobileNetV3_large(**kwargs)
    if pretrained:
        # state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        state_dict = torch.load('./mobilenet_v3_large-8738ca79.pth')  
        # model.load_state_dict(state_dict, strict=True)
        model.load_state_dict(state_dict, False)
        # raise NotImplementedError       
    return model


# model = mobilenetv3(pretrained= True)
# summary(model.feature,input_size=(1,3,640,640))    

# if __name__ == '__main__':
#    model = mobilenetv3(pretrained= True)
#    summary(model.feature,input_size=(1,3,640,640))