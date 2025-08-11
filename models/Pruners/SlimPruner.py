# SlimPruner 剪枝类
import torch
import torch.nn as nn

class SlimPruner:
    def __init__(self, model: nn.Module, prun_ratio: float = 0.2, gamma_limit: float = 0.2, prun_method: str = 'ratio'):
        
        self.model = model
        self.prun_ratio = prun_ratio
        self.bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.gamma_limit = gamma_limit
        self.prun_method = prun_method

    def importance(self, module=None):
        if module is None:
            raise ValueError("Module must be provided for importance calculation")
        if not isinstance(module, nn.BatchNorm2d):
            raise ValueError("SlimPruner only supports BatchNorm2d layers")

        # 以BN层的gamma参数绝对值作为通道重要性
        importance_list = []
        gamma = module.weight.data.abs().clone()
        importance_list.append(gamma)
        return importance_list

    def prune(self):
        for bn in self.bn_layers:
            # 找到BN层之前的Conv层
            prev_conv = None
            for name, module in self.model.named_modules():
                if module is bn:
                    # 通过遍历父模块的children，找到bn之前的conv
                    parent = self.model
                    for sub_name, sub_module in parent.named_children():
                        if sub_module is bn:
                            break
                        if isinstance(sub_module, nn.Conv2d):
                            prev_conv = sub_module
                    break
            
            if self.prun_method == 'ratio':
                # 对每个BN层，按gamma值从小到大评估 prun_ratio 的通道
                gamma = bn.weight.data.abs().clone()
                num_channels = gamma.size(0)
                num_prune = int(num_channels * self.prun_ratio)
                if num_prune == 0:
                    continue
                # 找到要剪掉的通道索引
                prune_idx = torch.argsort(gamma)[:num_prune]
                prune_idx, _ = torch.sort(prune_idx)
                keep_idx = torch.argsort(gamma)[num_prune:]
                keep_idx, _ = torch.sort(keep_idx)
                
            if self.prun_method == 'limit':
                # 对每个BN层，剪掉gamma小于 gamma_limit 的通道
                gamma = bn.weight.data.abs().clone()
                prune_idx = torch.where(gamma < self.gamma_limit)[0]
                keep_idx = torch.where(gamma >= self.gamma_limit)[0]
                num_prune = gamma.size(0) - keep_idx.size(0)
                if num_prune == 0:
                    continue
            
            # 剪BN参数
            keep_weight = bn.weight.data[keep_idx].clone()
            keep_bias = bn.bias.data[keep_idx].clone()
            keep_running_mean = bn.running_mean[keep_idx].clone() 
            keep_running_var = bn.running_var[keep_idx].clone() 

            bn.weight.data = keep_weight
            bn.bias.data = keep_bias
            bn.running_mean = keep_running_mean
            bn.running_var = keep_running_var
             
            # 结构化剪枝，删除多余通道
            bn.num_features = keep_weight.size(0)
            # 剪Conv参数（输出通道）
            if prev_conv is not None:
                # Conv2d: [out_channels, in_channels, kH, kW]
                keep_conv_weight = prev_conv.weight.data[keep_idx, ...].clone()
                prev_conv.weight.data = keep_conv_weight
                prev_conv.out_channels = keep_weight.size(0)
                if prev_conv.bias is not None:
                    prev_conv.bias.data = prev_conv.bias.data[keep_idx].clone()
        return self.model