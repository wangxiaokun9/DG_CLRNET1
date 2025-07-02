net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',  #  使用ResNet作为骨干网络，并通过ResNetWrapper进行封装。
    resnet='resnet18',   # 使用ResNet-18作为基础网络
    pretrained=True,    # 使用预训练的权重
    replace_stride_with_dilation=[False, False, False], # 不使用空洞卷积来替代步长（stride）
    out_conv=False,  # 不使用额外的卷积层来输出特征
)

num_points = 72  # 定义了模型在每条车道线上采样的点数为72
max_lanes = 4   # 定义了最多检测的车道线数量为4
sample_y = range(589, 230, -20) # 定义了在y轴上的采样点，从589到230，步长为-20

heads = dict(type='CLRHead',
             num_priors=192,  # 定义了先验框的数量为192
             refine_layers=3,  # 定义了精炼层的数量为3
             fc_hidden_dim=64, # 全连接层的隐藏维度为64
             sample_points=36) #  在每条车道线上采样的点数为36

iou_loss_weight = 2.  # IoU（交并比）损失的权重为2.0
cls_loss_weight = 2.  # 分类损失的权重为2.0
xyt_loss_weight = 0.2  # xyt损失的权重为0.2（可能是位置或坐标损失）
seg_loss_weight = 1.0  # 分割损失的权重为1.0

work_dirs = "work_dirs/clr/r18_culane"

neck = dict(type='FPN_L',
            in_channels=[128, 256, 512],  # FPN 网络期望接收到 3 个输入特征图，每个特征图的通道数分别为 128、256 和 512
            out_channels=64,
            num_outs=3,  # 输出特征图的数量为3
            attention=False)  # 不使用注意力机制

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes) #置信度阈值为0.4  非极大值抑制（NMS）的阈值为50   保留的检测框数量为max_lanes（即最多4个）
 
epochs = 15
batch_size = 24

optimizer = dict(type='AdamW', lr=0.6e-3)  # 3e-4 for batchsize 8 # 用AdamW优化器，学习率为0.6e-3
total_iter = (88880 // batch_size) * epochs  # 总迭代次数为(88880 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter) # 使用余弦退火学习率调度器，T_max设置为总迭代次数

eval_ep = 1  # 每1个epoch进行一次评估。
save_ep = 5 #  每10个epoch保存一次模型

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])  #  图像归一化的均值和标准差
ori_img_w = 1640  # 原始图像的宽度和高度
ori_img_h = 590  
img_w = 800    # 输入网络的图像的宽度和高度
img_h = 320 
cut_height = 270  # 在图像中剪切的高度为270像素

train_process = [  # 训练数据的预处理流程
    dict(
        type='GenerateLaneLine', # 生成车道线数据
        transforms=[   # 一系列的数据增强操作
            dict(name='Resize', # 调整图像大小为 img_h 和 img_w
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),  # 水平翻转图像，概率为 0.5
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),  #  随机打乱图像通道，概率为 0.1
            dict(name='MultiplyAndAddToBrightness',  # 调整图像亮度，乘法因子在 0.85 到 1.15 之间，加法因子在 -10 到 10 之间，概率为 0.6
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',  # 调整图像的色调和饱和度，值在 -10 到 10 之间，概率为 0.7
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',  # 从多种变换中随机选择一种，概率为 0.2
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))), #  运动模糊，模糊核大小为 3 到 5 之间
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))  #  中值模糊，模糊核大小为 3 到 5 之间
                 ],
                 p=0.2),
            dict(name='Affine',  # 仿射变换，包括平移、旋转和缩放，概率为 0.7
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',  # 调整图像大小为 img_h 和 img_w，确保最终大小一致
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),  #  将图像、车道线和分割图转换为张量
]

val_process = [  # 验证数据的预处理流程
    dict(type='GenerateLaneLine',  # 生成车道线数据
         transforms=[   # 仅包含调整图像大小的操作，概率为 1.0
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),  #  仅将图像转换为张量
]

dataset_path = './data/CULane'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 0
log_interval = 1000  #  日志输出的间隔为 1000 个样本
# seed = 0
num_classes = 4 + 1  # 分类的类别数，这里是 5（包括背景）
ignore_label = 255  # 忽略的标签，这里是 255
bg_weight = 0.4  # 背景的权重，这里是 0.4
lr_update_by_epoch = False  # 学习率更新策略，这里是 False，表示不按 epoch 更新
