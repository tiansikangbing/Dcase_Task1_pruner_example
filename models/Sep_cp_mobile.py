import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.ops.misc import ConvNormActivation


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
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvBnRelu(nn.Module):
    """
    Standard convolution block with Batch normalization and Relu activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias=False,
                 use_bn=True,
                 use_relu=True):
        super(ConvBnRelu, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups,
                               bias=bias)
        self._bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self._relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x) if self._bn is not None else x
        x = self._relu(x) if self._relu is not None else x
        return x
    
    def importance(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio'):
        if prune_method == 'ratio':
            # 对每个BN层，按gamma值从小到大评估 prun_ratio 的通道
            gamma = self._bn.weight.data.abs().clone()
            num_channels = gamma.size(0)
            num_prune = int(num_channels * prune_ratio)
            # 找到要剪掉的通道索引
            prune_idx = torch.argsort(gamma)[:num_prune]
            prune_idx, _ = torch.sort(prune_idx)
            keep_idx = torch.argsort(gamma)[num_prune:]
            keep_idx, _ = torch.sort(keep_idx)
                
        if prune_method == 'limit':
            # 对每个BN层，剪掉gamma小于 gamma_limit 的通道
            gamma = self._bn.weight.data.abs().clone()
            prune_idx = torch.where(gamma < gamma_limit)[0]
            keep_idx = torch.where(gamma >= gamma_limit)[0]
            num_prune = gamma.size(0) - keep_idx.size(0)
        
        return keep_idx, prune_idx
    
    def prune(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio', prune_input_channel = None):
        if prune_input_channel is not None:
            #修剪输入通道适应上一层的参数变化
            self._conv.weight.data = self._conv.weight.data[:, prune_input_channel, :, :].clone()
            self._conv.in_channels = prune_input_channel.size(0)

        if prune_ratio == 0:
            return None

        # 计算通道重要性
        keep_idx, _ = self.importance(prune_ratio, gamma_limit, prune_method)

        # 剪BN参数
        keep_weight = self._bn.weight.data[keep_idx].clone()
        keep_bias = self._bn.bias.data[keep_idx].clone()
        keep_running_mean = self._bn.running_mean[keep_idx].clone() 
        keep_running_var = self._bn.running_var[keep_idx].clone() 

        self._bn.weight.data = keep_weight
        self._bn.bias.data = keep_bias
        self._bn.running_mean = keep_running_mean
        self._bn.running_var = keep_running_var
            
        # 结构化剪枝，删除多余通道
        self._bn.num_features = keep_weight.size(0)

        # 剪conv通道
        self._conv.weight.data = self._conv.weight.data[keep_idx, :, :, :].clone()
        self._conv.out_channels = keep_weight.size(0)
        if self._conv.groups == self._conv.in_channels:
            # 深度分离卷积在剪枝时要匹配输入输出通道，保证每一个独立的卷积通道
            self._conv.in_channels = keep_idx.size(0)
            self._conv.groups = keep_idx.size(0)

        # 返回保留输出通道索引
        return keep_idx

class Residual(nn.Module):
    def __init__(self, x, y, in_channels):
        super().__init__()
        

class GRN(nn.Module):
    """
    global response normalization as introduced in https://arxiv.org/pdf/2301.00808.pdf
    全局响应归一化层
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.quant = QuantStub()
        self.dequant =  ()

    def forward(self, x):
        # # dequantize and quantize since torch.norm not implemented for quantized tensors
        # x = self.dequant(x)
        # 求特征的L2范数
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # L2范数除通道平均
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        # 深化各重要特征影响，弱化冗余特征
        x = self.gamma * (x * nx) + self.beta + x
        # return self.quant(x)
        return x

class CPMobileBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_rate,
                 stride,
                 dropout_rate
                 ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)
        half_exp_channels = exp_channels // 2

        # create the three factorized convs that make up our block
        # 拓展卷积层
        self.exp_conv = ConvBnRelu(in_channels=in_channels,
                                   out_channels=exp_channels,
                                   kernel_size=1
                                   )

        self.freq_depth_conv = ConvBnRelu(in_channels=half_exp_channels,
                                          out_channels=half_exp_channels,
                                          kernel_size=(3,1),
                                          stride=stride,
                                          padding=(1,0),
                                          groups=half_exp_channels
                                          )
        
        self.freq_avg_pooling = nn.AvgPool2d(kernel_size=(3,1), 
                                             stride=1, 
                                             padding=(1,0))
        
        self.temp_depth_conv = ConvBnRelu(in_channels=half_exp_channels,
                                          out_channels=half_exp_channels,
                                          kernel_size=(1,3),
                                          stride=stride,
                                          padding=(0,1),
                                          groups=half_exp_channels
                                          )
        
        self.temp_avg_pooling = nn.AvgPool2d(kernel_size=(1,3), 
                                             stride=1, 
                                             padding=(0,1)
                                             )
        
        self.proj_conv = ConvBnRelu(in_channels=exp_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    use_relu=False
                                    )
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.residual_in_channels = torch.arange(in_channels)
        self.residual_out_channels = torch.arange(out_channels)

        self.after_block_norm = GRN()
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

    def forward(self, x):
        residual = None
        if self.use_shortcut:
            # residual = self.shortcut(x)
            residual = x[:, self.residual_in_channels, :, :]
            residual = self.shortcut(residual)

        x = self.exp_conv(x)
        split_channels = [self.freq_depth_conv._conv.in_channels, self.temp_depth_conv._conv.in_channels]

        # frequency depthwise convolution
        x_freq, x_temp = torch.split(x, split_channels, dim=1)
        x_freq = self.freq_depth_conv(x_freq)
        x_freq = self.freq_avg_pooling(x_freq)
        x_freq = self.dropout_layer(x_freq)
        
        # temporal depthwise convolution
        x_temp = self.temp_depth_conv(x_temp)
        x_temp = self.temp_avg_pooling(x_temp)
        x_temp = self.dropout_layer(x_temp)

        # Concat the two feature maps
        x = torch.cat([x_freq, x_temp], dim=1)

        x = self.proj_conv(x)
        if residual is not None:
            x[:, self.residual_in_channels, :, :] = x[:, self.residual_out_channels, :, :] + residual
        x = self.after_block_norm(x)
        x = self.after_block_activation(x)
        return x

    def pruneInputChannel(self, prune_input_channel = None):
        # 输入通道修剪
        if prune_input_channel is not None:
            self.exp_conv.prune(prune_ratio = 0,prune_input_channel=prune_input_channel)

    def selfDepthPrune(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio'):
        # 深度卷积层剪枝
        freq_dw_input_size = self.freq_depth_conv._conv.in_channels
        freq_dw_keep_index = self.freq_depth_conv.prune(prune_ratio, gamma_limit, prune_method)
        temp_dw_keep_index = self.temp_depth_conv.prune(prune_ratio, gamma_limit, prune_method)
        dw_conv_keep_index = torch.cat([freq_dw_keep_index, temp_dw_keep_index + freq_dw_input_size], dim=0)

        # 反向剪枝exp_conv
        self.exp_conv._bn.weight.data = self.exp_conv._bn.weight.data[dw_conv_keep_index].clone()
        self.exp_conv._bn.bias.data = self.exp_conv._bn.bias.data[dw_conv_keep_index].clone()
        self.exp_conv._bn.running_mean = self.exp_conv._bn.running_mean[dw_conv_keep_index].clone()
        self.exp_conv._bn.running_var = self.exp_conv._bn.running_var[dw_conv_keep_index].clone()
        self.exp_conv._bn.num_features = dw_conv_keep_index.size(0)

        self.exp_conv._conv.weight.data = self.exp_conv._conv.weight.data[dw_conv_keep_index, :, :, :].clone()
        self.exp_conv._conv.out_channels = dw_conv_keep_index.size(0)
        
        # 剪枝proj_conv
        self.proj_conv.prune(prune_ratio = 0, prune_input_channel=dw_conv_keep_index)

    def pruneOutputChannel(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio', exp_conv_keep_index = None):
        # 输出通道修剪
        proj_conv_keep_index = self.proj_conv.prune(prune_ratio, gamma_limit, prune_method)

        shortcut_keep_input_index = []
        shortcut_keep_output_index = []
        residual_remain_index = []
        
        if self.use_shortcut:
            # 根据residual通道变化，调整shortcut
            if exp_conv_keep_index is not None:
                # 取exp_conv的保留通道与proj_conv的保留通道的交集
                for i in range(len(proj_conv_keep_index)):
                    if proj_conv_keep_index[i] in exp_conv_keep_index:
                        residual_remain_index.append(proj_conv_keep_index[i])
                # 根据residual_remain_index在exp_conv_keep_index中的位置，得到最终shortcut的输入通道索引
                for i in range(len(exp_conv_keep_index)):
                    if exp_conv_keep_index[i] in residual_remain_index:
                        shortcut_keep_input_index.append(i)

                for i in range(len(proj_conv_keep_index)):
                    if proj_conv_keep_index[i] in residual_remain_index:
                        shortcut_keep_output_index.append(i)

            # 调整shortcut
            self.residual_in_channels = shortcut_keep_input_index
            self.residual_out_channels = shortcut_keep_output_index

        return proj_conv_keep_index
        
        
            
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        dropout_rate = config['dropout_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        # 计算阶段通道
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8) for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 输入层
        self.in_c = nn.Sequential(ConvNormActivation(in_channels,
                                                     channels_per_stage[0] // 4,
                                                     kernel_size=3,
                                                     stride=2,
                                                     inplace=False
                                                     ),

                                  ConvNormActivation(channels_per_stage[0] // 4,
                                                     channels_per_stage[0],
                                                     activation_layer=torch.nn.ReLU,
                                                     kernel_size=3,
                                                     stride=2,
                                                     inplace=False
                                                     ))

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate,
                                     dropout_rate = dropout_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        # 输出层
        ff_list = []
        ff_list += [nn.Conv2d(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]
        # 自适应的池化层
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
                    expansion_rate,
                    dropout_rate):
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
                expansion_rate=expansion_rate,
                dropout_rate = dropout_rate
                )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate,
                    dropout_rate
                    ):

        block = CPMobileBlock(in_channels,
                              out_channels,
                              expansion_rate,
                              stride,
                              dropout_rate)
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        global first_RUN
        x = self.quant(x)
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        logits = x.squeeze(2).squeeze(2)
        logits = self.dequant(logits)
        return logits

    def fuse_model(self):
        for m in self.named_modules():
            module_name = m[0]
            module_instance = m[1]
            if module_name == 'in_c':
                in_conv_1 = module_instance[0]
                in_conv_2 = module_instance[1]
                torch.quantization.fuse_modules(in_conv_1, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(in_conv_2, ['0', '1', '2'], inplace=True)
            elif isinstance(module_instance, CPMobileBlock):
                exp_conv = module_instance.block[0]
                depth_conv = module_instance.block[1]
                proj_conv = module_instance.block[2]
                torch.quantization.fuse_modules(exp_conv, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(depth_conv, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(proj_conv, ['0', '1'], inplace=True)
            elif module_name == "feed_forward":
                torch.quantization.fuse_modules(module_instance, ['0', '1'], inplace=True)


def get_model_Sep_cp_mobile(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0, dropout_rate=0.1, n_blocks=(3, 2, 1), strides=None):
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
            b2=(2, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides,
        "dropout_rate": dropout_rate
    }

    m = Network(model_config)
    # print(m)
    return m
