import torch
import torch.nn as nn
from models.Sep_cp_mobile import get_model_Sep_cp_mobile
from models.Sep_cp_mobile import CPMobileBlock, GRN
from train_cp_mobile import eval_acc, train_one_epoch, get_args, build_mel_pipeline, worker_init_fn
from torch.utils.data import DataLoader
from dataset.data import get_training_set, get_validation_set
from transformers import get_cosine_schedule_with_warmup
from Tools.complexity import get_model_size_bytes

def main():
    # 加载参数和数据
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    mel, mel_augment = build_mel_pipeline(args)
    mel, mel_augment = mel.to(device), mel_augment.to(device)

    validation_ds = get_validation_set(device=None)
    validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    # 加载模型
    model = get_model_Sep_cp_mobile(
        n_classes=args.n_classes,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        channels_multiplier=args.channels_multiplier,
        expansion_rate=args.expansion_rate,
        ).to(device)
    model.load_state_dict(torch.load('sep_cp_mobile_L1_limit_epoch_150.pth', map_location=device))

    # 剪枝前测试
    acc_before = eval_acc(model, validation_dl, mel, device)
    print(f"剪枝前验证集准确率: {acc_before:.4f}")

    # 计算模型参数大小
    param_bytes, num_params = get_model_size_bytes(model)
    print(f"模型参数量：{num_params} ")

    # 剪枝参数设置
    prune_config = {
        'prune_ratio': 0.6,
        'gamma_limit': 0.3,
        'prune_method': 'ratio'
    }

    # 深度通道剪枝
    for name, module in model.named_modules():
        if isinstance(module, CPMobileBlock) and module.proj_conv._conv.out_channels > 64:
            module.selfDepthPrune(**prune_config)
        
    # 计算模型参数大小
    _, num_params = get_model_size_bytes(model)
    print(f"深度剪枝后模型参数量：{num_params} ")

     # 剪枝后测试
    acc_after = eval_acc(model, validation_dl, mel, device)
    print(f"深度剪枝后验证集准确率: {acc_after:.4f}")
    print(f"深度剪枝后正确回归率: {acc_after / acc_before:.4f}")

    prune_config = {
        'prune_ratio': 0.2,
        'gamma_limit': 0.3,
        'prune_method': 'ratio'
    }
    
    prune_input_channel = None
    # 层连接剪枝
    for name, module in model.named_modules():
        if isinstance(module, CPMobileBlock) and module.proj_conv._conv.out_channels > 64:
            module.pruneInputChannel(prune_input_channel=prune_input_channel)
            prune_input_channel = module.pruneOutputChannel(**prune_config, exp_conv_keep_index=prune_input_channel)
        if isinstance(module, nn.Conv2d) and module.out_channels == args.n_classes:
            if prune_input_channel is not None:
                #修剪输入通道适应上一层的参数变化
                module.weight.data = module.weight.data[:, prune_input_channel, :, :].clone()
                module.in_channels = prune_input_channel.size(0)
    
    # 计算模型参数大小
    _, num_params = get_model_size_bytes(model)
    print(f"层连接剪枝后模型参数量：{num_params} ")

    # 剪枝后测试
    acc_after = eval_acc(model, validation_dl, mel, device)
    print(f"剪枝后验证集准确率: {acc_after:.4f}")
    print(f"剪枝后正确回归率: {acc_after / acc_before:.4f}")

    # 加载训练集
    roll_samples = int(args.orig_sample_rate * args.roll_sec)
    train_ds = get_training_set(device=None, roll=roll_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    # 剪枝后重训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.n_epochs * len(train_dl)
    )
    
    for epoch in range(1, args.retrain_epochs + 1):
        train_loss = train_one_epoch(model=model, train_batch=train_dl, args=args, optimizer=optimizer, scheduler=scheduler, mel=mel, mel_augment=mel_augment, device=device)
        acc = eval_acc(model, validation_dl, mel, device)
        acc_recover = acc / acc_before
        print(f"[retrain] Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test Acc: {acc:.4f} | Recovery Rate: {acc_recover:.4f}")

if __name__ == '__main__':
    main()