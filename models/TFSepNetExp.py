import torch
import torch.nn as nn
from typing import Union, Tuple

class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(ConvClassifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean((-1, -2), keepdim=False)
        return x
    
    def prune(self, prune_input_channel = None):
        if prune_input_channel is not None:
            #修剪输入通道适应上一层的参数变化
            self.conv.weight.data = self.conv.weight.data[:, prune_input_channel, :, :].clone()
            self.conv.in_channels = prune_input_channel.size(0)
        
        return prune_input_channel

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

        # 返回保留输出通道索引
        return keep_idx

class ResNorm(nn.Module):
    def __init__(self, channels: int, lamb=0.1, eps=1e-5):
        super(ResNorm, self).__init__()
        self._eps = torch.full((1, channels, 1, 1), eps)
        self._lambda = torch.full((1, channels, 1, 1), lamb)

    def forward(self, x):
        self._eps = self._eps.to(x.device)
        self._lambda = self._lambda.to(x.device)

        identity = x
        fi_mean = x.mean((1, 3), keepdim=True)
        fi_var = x.var((1, 3), keepdim=True)
        fin = (x - fi_mean) / (fi_var + self._eps).sqrt()
        return self._lambda * identity + fin
    
    def prune(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio', prune_input_channel = None):
        if prune_input_channel is not None:
            #修剪输入通道适应上一层的参数变化
            self._lambda = self._lambda[:, prune_input_channel, :, :].clone()
            self._eps = self._eps[:, prune_input_channel, :, :].clone()
        
        return prune_input_channel
    

class ShuffleLayer(nn.Module):
    def __init__(self, group: int):
        super(ShuffleLayer, self).__init__()
        self._group = group

    def forward(self, x):
        b, c, f, t = x.data.size()
        # assert c % self._group == 0
        group_channels = c // self._group

        x = x.reshape(b, group_channels, self._group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x

class TimeFreqSepConvolutions(nn.Module):
    """Implementation of Time-Frequency Separable Convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_rate: float):
        super(TimeFreqSepConvolutions, self).__init__()
        assert out_channels % 2 == 0, "Out channels must be divisible by 2"
        self.in_channels = in_channels
        self.out_channels = out_channels
        exp_channels = in_channels * 2

        # create the three factorized convs that make up the inverted bottleneck block
        # 拓展卷积，使用1x1卷积进行通道扩展，增加特征维度
        self.exp_conv = ConvBnRelu(in_channels, exp_channels, kernel_size=1)

        # 时频卷积层
        self.freq_dw_conv = ConvBnRelu(self.in_channels, 
                                       self.in_channels,
                                       (kernel_size, 1),
                                       padding=((kernel_size - 1) // 2, 0),
                                       groups=self.in_channels)
        self.freq_pw_conv = ConvBnRelu(self.in_channels, self.in_channels, 1)
        self.temp_dw_conv = ConvBnRelu(self.in_channels,
                                       self.in_channels,
                                       (1, kernel_size),
                                       padding=(0, (kernel_size - 1) // 2),
                                       groups=self.in_channels)
        self.temp_pw_conv = ConvBnRelu(self.in_channels, self.in_channels, 1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        # 投影卷积，使用1x1卷积进行通道压缩，降低特征维度
        self.proj_conv = ConvBnRelu(exp_channels, out_channels, kernel_size=1)

        # 判断残差连接的形式
        if in_channels != out_channels:
            self.identity_conv = ConvBnRelu(in_channels, out_channels, 1)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            identity = x
        else:
            identity = self.identity_conv(x)
        # Expand channels
        x = self.exp_conv(x)
        # Split feature maps with half the channels
        split_channel = [self.freq_dw_conv._conv.in_channels, self.temp_dw_conv._conv.in_channels]
        x1, x2 = torch.split(x, split_channel, dim=1)
        # Frequency-wise convolution block
        x1 = self.freq_dw_conv(x1)
        x1 = x1.mean(2, keepdim=True)  # frequency average pooling
        x1 = self.freq_pw_conv(x1)
        x1 = self.dropout_layer(x1)
        # Time-wise convolution block
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout_layer(x2)
        # Concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        x = self.proj_conv(x)
        # Add identity connection
        x = x + identity
        return x
    
    def prune(self, prune_ratio: float = 0.2, gamma_limit: float = 0.2, prune_method: str = 'ratio', prune_input_channel = None):
        freq_dw_remain_index = None
        temp_dw_remain_index = None
        self.in_channels = self.freq_dw_conv._conv.in_channels
        if self.trans_conv is not None:
            if prune_input_channel is not None:
                #修剪输入通道适应上一层的参数变化
                self.trans_conv.prune(prune_ratio = 0, gamma_limit = 0, prune_method = 'ratio', prune_input_channel = prune_input_channel)
        else:
            if prune_input_channel is not None:
                # 如果没有转换卷积层，则根据后续时频分离卷积层的输入通道修剪
                
                # 将prune_input_channel中值小于self.in_channels的索引作为freq_dw_conv的保留通道
                for i in range(prune_input_channel.size(0)):
                    if prune_input_channel[i] < self.in_channels:
                        freq_dw_remain_index = prune_input_channel[:i]
                        temp_dw_remain_index = prune_input_channel[i+1:]

                # 每项减self.in_channels，得到temp_dw_conv的保留通道索引
                temp_dw_remain_index = temp_dw_remain_index - self.in_channels
            
        # 修剪时频分离卷积层
        freq_dw_keep_index = self.freq_dw_conv.prune(prune_ratio, gamma_limit, prune_method, freq_dw_remain_index)
        # 深度分离卷积在剪枝时要匹配输入输出通道，保证每一个独立的卷积通道
        self.freq_dw_conv._conv.in_channels = freq_dw_keep_index.size(0)
        self.freq_dw_conv._conv.groups = freq_dw_keep_index.size(0)

        freq_pw_keep_index = self.freq_pw_conv.prune(prune_ratio, gamma_limit, prune_method, freq_dw_keep_index)

        temp_dw_keep_index = self.temp_dw_conv.prune(prune_ratio, gamma_limit, prune_method, temp_dw_remain_index)
        # 深度分离卷积在剪枝时要匹配输入输出通道，保证每一个独立的卷积通道
        self.temp_dw_conv._conv.in_channels = temp_dw_keep_index.size(0)
        self.temp_dw_conv._conv.groups = temp_dw_keep_index.size(0)
        self.temp_dw_conv._conv.groups = temp_dw_keep_index.size(0)

        temp_pw_keep_index = self.temp_pw_conv.prune(prune_ratio, gamma_limit, prune_method, temp_dw_keep_index)

        # 由于深度分离卷积输入通道被剪枝，将剪枝通道反向传递给trans_conv剪枝输出通道
        if self.trans_conv is not None:
            trans_output_keep_index = torch.cat((freq_dw_keep_index, temp_dw_keep_index + self.in_channels), dim=0)
            # 剪BN参数
            keep_weight = self.trans_conv._bn.weight.data[trans_output_keep_index].clone()
            keep_bias = self.trans_conv._bn.bias.data[trans_output_keep_index].clone()
            keep_running_mean = self.trans_conv._bn.running_mean[trans_output_keep_index].clone() 
            keep_running_var = self.trans_conv._bn.running_var[trans_output_keep_index].clone() 

            self.trans_conv._bn.weight.data = keep_weight
            self.trans_conv._bn.bias.data = keep_bias
            self.trans_conv._bn.running_mean = keep_running_mean
            self.trans_conv._bn.running_var = keep_running_var
                
            # 结构化剪枝，删除多余通道
            self.trans_conv._bn.num_features = keep_weight.size(0)

            # 剪conv通道
            self.trans_conv._conv.weight.data = self.trans_conv._conv.weight.data[trans_output_keep_index, :, :, :].clone()
            self.trans_conv._conv.out_channels = keep_weight.size(0)
        # 计算下一层的输入通道
        next_input_channel = torch.cat((freq_pw_keep_index, temp_pw_keep_index + self.in_channels), dim=0)

        # 更新残差连接通道
        # self.identity1_use_channels = freq_pw_keep_index
        # self.identity2_use_channels = temp_pw_keep_index

        return next_input_channel


class TFSepNetExp(nn.Module):
    """
    Implementation of TF-SepNet-64, based on Time-Frequency Separate Convolutions. Check more details at:
    https://ieeexplore.ieee.org/abstract/document/10447999 and
    https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cai_61_t1.pdf

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with two options: 16 or 17. When depth = 17, an additional Max-pooling layer is inserted before the last TF-SepConvs black.
        kernel_size (int): Kernel size of each convolutional layer in TF-SepConvs blocks.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(TFSepNetExp, self).__init__()
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Two settings of the depth. ``17`` have an additional Max-pooling layer before the final block of TF-SepConvs.
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
                                         ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1,
                                                    groups=base_channels // 2))
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        # vt = width * 2
        last_out_channels = width
        for config in layer_config:
            if config == 'N':
                layers += [ResNorm(channels=last_out_channels)]
            elif config == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif config != last_out_channels:
                layers += [TimeFreqSepConvolutions(last_out_channels, config, self.kernel_size, self.dropout)]
                last_out_channels = config
            else:
                layers += [TimeFreqSepConvolutions(last_out_channels, last_out_channels, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x
    
def get_model_TFSepNetExp(in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):

    model_config = {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'base_channels': base_channels,
        'depth': depth,
        'kernel_size': kernel_size,
        'dropout': dropout
    }

    m = TFSepNetExp(**model_config)
    return m
