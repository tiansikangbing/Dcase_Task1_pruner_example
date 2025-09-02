import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_distribution(data, bins=30, title="data distribution", xlabel="value", ylabel="frequency"):
    """
    Plot data distribution (histogram)

    参数:
        data: 可迭代的一组数据（如列表、numpy数组等）
        bins: 直方图的分箱数量，默认30
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_bn_param(model):
    # 打印模型结构
    print(model)

    # 遍历模型的每一层
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(module)
            # 打印每个BatchNorm层的gamma参数
            plot_distribution(module.weight.data.numpy(), bins=module.weight.data.numpy().shape[0], title=f'{name} gamma distribution', xlabel='gamma value', ylabel='frequency')   