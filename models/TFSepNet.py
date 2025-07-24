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
        half_channels = out_channels // 2

        self.trans_conv = ConvBnRelu(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.freq_dw_conv = ConvBnRelu(half_channels, half_channels, (kernel_size, 1),
                                       padding=((kernel_size - 1) // 2, 0),
                                       groups=half_channels)
        self.freq_pw_conv = ConvBnRelu(half_channels, half_channels, 1)
        self.temp_dw_conv = ConvBnRelu(half_channels, half_channels, (1, kernel_size),
                                       padding=(0, (kernel_size - 1) // 2),
                                       groups=half_channels)
        self.temp_pw_conv = ConvBnRelu(half_channels, half_channels, 1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.shuffle_layer = ShuffleLayer(group=half_channels)

    def forward(self, x):
        # Expand or shrink channels if in_channels != out_channels
        x = self.trans_conv(x) if self.trans_conv is not None else x
        # Channel shuffle
        x = self.shuffle_layer(x)
        # Split feature maps with half the channels
        x1, x2 = torch.split(x, x.data.size(1) // 2, dim=1)
        # Copy x1, x2 for residual path
        identity1 = x1
        identity2 = x2
        # Frequency-wise convolution block
        x1 = self.freq_dw_conv(x1)
        x1 = x1.mean(2, keepdim=True)  # frequency average pooling
        x1 = self.freq_pw_conv(x1)
        x1 = self.dropout_layer(x1)
        x1 = x1 + identity1
        # Time-wise convolution block
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout_layer(x2)
        x2 = x2 + identity2
        # Concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        return x
        
class TFSepNet(nn.Module):
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
        super(TFSepNet, self).__init__()
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
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x
    
def get_model_TFSepNet(in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):

    model_config = {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'base_channels': base_channels,
        'depth': depth,
        'kernel_size': kernel_size,
        'dropout': dropout
    }

    m = TFSepNet(**model_config)
    return m
