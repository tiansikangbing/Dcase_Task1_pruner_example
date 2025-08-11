import torch
import torch.nn as nn
from Tools.plotData import plot_distribution
from models.TFSepNet import get_model_TFSepNet
from models.TFSepNet import ConvBnRelu

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = get_model_TFSepNet(
        in_channels = 1, 
        num_classes = 10,
        base_channels = 64,
        depth = 17,
        kernel_size = 3,
        dropout = 0.1
    ).to(device)    
model.load_state_dict(torch.load('model.pth'))

print(model)

# 遍历模型的每一层
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        print(module)
        # 打印每个BatchNorm层的gamma参数
        plot_distribution(module.weight.data.numpy(), bins=module.weight.data.numpy().shape[0], title=f'{name} gamma distribution', xlabel='gamma value', ylabel='frequency')   