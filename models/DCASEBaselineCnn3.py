import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from typing import Optional

def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 初始化权重
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # 使用 Kaiming 正态分布初始化卷积层的权重
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        # 将偏置初始化为零
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    #将 BN 层的权重初始化为 1，偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # 使用正态分布初始化线性层的权重
        nn.init.normal_(m.weight, 0, 0.01)
        # 将线性层的偏置初始化为零
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up the inverted bottleneck block
        # 拓展卷积，使用1x1卷积进行通道扩展，增加特征维度
        exp_conv = Conv2dNormActivation(in_channels,
                                        exp_channels,
                                        kernel_size=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        # depthwise convolution with possible stride
        # 深度卷积，使用3x3卷积进行空间特征提取，保持通道数不变，且groups设置为扩展通道数
        depth_conv = Conv2dNormActivation(exp_channels,
                                          exp_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=exp_channels,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU,
                                          inplace=False
                                          )

        # 投影卷积，使用1x1卷积进行通道压缩，降低特征维度
        proj_conv = Conv2dNormActivation(exp_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=None,
                                         inplace=False
                                         )
        self.after_block_activation = nn.ReLU()

        # 判断残差连接的形式
        if in_channels == out_channels:
            # 如果输入和输出通道数相同，使用残差连接
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                # 如果步幅为1或(1, 1)，说明特征的空间大小没有变化，不需要做任何变化
                self.shortcut = nn.Sequential()
            else:
                # 如果步幅大于1，则需要使用平均池化进行下采样，保证其与主路径输出的特征图大小一致
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )

    # 前向传播
    def forward(self, x):
        if self.use_shortcut:
            x = self.block(x) + self.shortcut(x)
        else:
            x = self.block(x)
        x = self.after_block_activation(x)
        return x

                  
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']

        # stage（阶段）在卷积神经网络中通常指一组结构相似、输出特征图尺寸相同的卷积块（block）的集合。每个stage一般包含若干个block，stage之间通常通过步幅或池化实现空间下采样。
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        # 为什么每个stage的通道数需要递增？
        # 下采样后空间分辨率降低，特征图变小，为了保持信息表达能力，需要增加通道数（即特征维度）。
        # 低层stage主要提取局部、简单特征，通道数较少；高层stage提取更复杂、更抽象的特征，需要更多通道来表达丰富的信息。
        # 递增通道数是现代高效CNN（如ResNet、MobileNet等）的常见设计，能在保证计算量可控的前提下提升模型表达力和性能。
        self.total_block_count = 0

        # 定义输入部分的卷积序列，先将输入通道数降到1/4再升到第一个stage的通道数，均用3x3卷积和ReLU激活，步幅为2，实现下采样。
        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
            Conv2dNormActivation(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
        )

        # 初始化stage容器。遍历每个stage，调用_make_stage构建对应的block序列，并添加到self.stages。
        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        # 处理输出分类
        ff_list = []

        # 使用1x1卷积将最后一个stage的通道数映射到类别数，后接BatchNorm和ReLU激活，实现分类。
        ff_list += [nn.Conv2d(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]

        # 平均池化，输出特征图尺寸为1x1，融合特征，输出最终的分类结果
        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate
                    ):

        block = Block(in_channels,
                      out_channels,
                      expansion_rate,
                      stride
                      )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        # 最后两个维度为1x1，剪切掉多余的维度
        logits = x.squeeze(2).squeeze(2)
        return logits


def get_model_DCASEBaselineCnn3(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0,
              n_blocks=(2, 2, 2, 3), strides=None):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides
    }

    m = Network(model_config)
    return m
