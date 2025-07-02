import torch
import torch.nn as nn
import math
 
 
__all__ = ['SENet', 'se_resnet_18', 'se_resnet_34', 'se_resnet_50', 'se_resnet_101',
           'se_resnet_152']
 
 
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)
 
 
def conv3x3(in_planes, out_planes, stride=1):
    # 定义3x3卷积，并且填充数为1
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
 
class BasicBlock(nn.Module):
    expansion = 1  # 扩展倍数的属性
 
    # 用于在 ResNet 中确定每个 BasicBlock 层的输入通道和输出通道之间的倍数关系。通过将输入通道数乘以 expansion，可以得到输出通道数
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3x3卷积
        self.bn1 = nn.BatchNorm2d(planes)  # 归一化
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # 3x3卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 传入的下采样方法保存为一个属性
        self.stride = stride  # 步长
 
        if planes == 64:  # 输出特征图的通道数
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)  # 全局平均池化
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))  # 全连接
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)  # 全连接
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)  # 3x3conv，s=1
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)  # 3x3conv，s=0
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        original_out = out
        out = self.globalAvgPool(out)  # # 全局平均池化
        out = out.view(out.size(0), -1)
        # out.size(0) 表示第一个维度的大小保持不变，而 -1 表示在保持其他维度的前提下，自动调整第二个维度的大小
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)  # 新张量的形状为 (out.size(0), out.size(1), 1, 1)
        out = out * original_out
 
        out += residual  # 残差连接
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4  # 扩展倍数的属性
 
    # 用于在 ResNet 中确定每个 BasicBlock 层的输入通道和输出通道之间的倍数关系。通过将输入通道数乘以 expansion，可以得到输出通道数
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1x1conv
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,  # 3x3conv
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 1x1conv,输出通道是输入通道的4倍
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)  # 平均池化
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))  # 除 4 取整
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
 
        out += residual
        out = self.relu(out)
 
        return out
 
 
class SENet(nn.Module):
    # SE注意力机制
 
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # block：指定使用的基本块类型，可以是残差块或其他类型的块
        # 64、128、256、512：指定每个卷积层组中的通道数（即输出特征图的通道数）
        # layers[0]、layers[1]、layers[2]、layers[3]：指定每个卷积层组中的基本块数量
        # stride=2：指定每个卷积层组中的卷积层的步长（stride），默认为 2
        # self.avgpool = nn.AvgPool2d(7, stride=1)  # 平均池化
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接
 
        for m in self.modules():  # 遍历模型的所有子模块
            if isinstance(m, nn.Conv2d):  # 如果m模块时卷积模块
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算卷积核的参数个数 n
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 对卷积核的权重进行初始化。这里使用了正态分布来初始化权重，均值为 0，标准差为 math.sqrt(2. / n)。这种初始化方法可以帮助模型更好地收敛
            elif isinstance(m, nn.BatchNorm2d):  # 如果m模块是归一化模块
                m.weight.data.fill_(1)  # 归一化层的权重进行初始化，将所有权重设置为 1
                m.bias.data.zero_()  # 归一化层的偏移量进行初始化，将所有偏移量设置为 0
 
    def _make_layer(self, block, planes, blocks, stride=1):  # 用于创建一个层
        # block是一个模型中的基本单元，planes是层中的通道数，blocks是层中重复的次数，stride是步长，默认值为1
        downsample = None  # 下采样层的目的是降低输出特征图的尺寸和增加通道数，以便将输入的特征图与输出的特征图进行匹配
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # 下采样快
                nn.BatchNorm2d(planes * block.expansion),  # 归一化
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # block函数的目的是创建一个新的模块，该模块由一个卷积层和一个批归一化层组成，同时也可能包含一个下采样层
 
        return nn.Sequential(*layers)  # 构建了一个顺序网络容器，将 layers 列表中的模块组合在一起，并将其作为方法的返回值
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 特征提取和降维
 
        x = self.avgpool(x)  # 将特征降维为一维
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 全连接层进行分类
        print(x.shape)
 
        return x
 
 
def se_resnet_34(pretrained=False, **kwargs):
    model = SENet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model
 
 
def se_resnet_50(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
 
 
def se_resnet_101(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
 
 
def se_resnet_152(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
 
 
def se_resnet_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        pretrained 表示是否使用在 ImageNet 数据集上预训练的模型
    """
    # 使用了一个名为 SENet 的类来构建模型，同时也使用了 BasicBlock 类作为 ResNet 中的基本块
    # [2, 2, 2, 2] 是一个列表，表示模型中每个阶段（stage）中重复 BasicBlock 的次数
    model = SENet(BasicBlock, [2, 2, 2, 2], **kwargs)  # 创建含有SE注意机制的残差模块
    return model
 
 
if __name__ == '__main__':
    net = se_resnet_18(pretrained=False,num_classes = 10)
    print(net)
 
    input_size = torch.rand(1,3,224,224)
    o = net(input_size)         # torch.Size([1, 10])
 
    # print(o.size())