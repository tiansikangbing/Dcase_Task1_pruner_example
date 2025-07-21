import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from dataset.data import get_training_set, get_test_set
from torch.distributions.beta import Beta
from models.module import get_model

# 参数预输入
def get_args():
    parser = argparse.ArgumentParser(description='DCASE 25 simple PyTorch training')
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channels_multiplier", type=float, default=1.8)
    parser.add_argument("--expansion_rate", type=float, default=2.1)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mixstyle_p", type=float, default=0.4)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

# 定义音频预处理模型
def build_mel_pipeline(args):
    mel = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=args.window_length,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            f_min=args.f_min,
            f_max=args.f_max
        )
    )
    mel_augment = torch.nn.Sequential(
        torchaudio.transforms.FrequencyMasking(args.freqm, iid_masks=True),
        torchaudio.transforms.TimeMasking(args.timem, iid_masks=True)
    )
    return mel, mel_augment

# 音频预处理
def mel_forward(mel, mel_augment, x, training):
    """
    :param x: batch of raw audio signals (waveforms)
    :return: log mel spectrogram
    """
    x = mel(x)
    if training:
        x = mel_augment(x)
    # 防止log(0) → 加1e-5
    x = (x + 1e-5).log()
    return x

import random

def worker_init_fn(wid):
    """
    Initializes seeds for Python random, NumPy, and PyTorch for DataLoader workers.
    """
    # Generate a unique seed sequence per worker
    seed_sequence = np.random.SeedSequence([torch.initial_seed(), wid])

    # Generate independent seeds for each random generator
    torch_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]
    np_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]
    py_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]

    # Apply seeds
    torch.manual_seed(int(torch_seed))
    np.random.seed(int(np_seed))
    random.seed(int(py_seed))

def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    return x

# 训练一个epoch
def train_one_epoch(model, train_batch, args, optimizer, mel, mel_augment, device):
    model.train()
    for x, labels in train_batch:
        x, labels = x.to(device), labels.to(device)
        x = mel_forward(mel, mel_augment, x, training=True)
        if args.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, args.mixstyle_p, args.mixstyle_alpha)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

# 检验正确率
def eval_acc(model, test_batch, mel, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in test_batch:
            x, labels = x.to(device), labels.to(device)
            x = mel_forward(mel, None, x, training=False)
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 主函数
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel, mel_augment = build_mel_pipeline(args)
    mel, mel_augment = mel.to(device), mel_augment.to(device)

    # 数据集
    roll_samples = int(args.orig_sample_rate * args.roll_sec)
    train_ds = get_training_set(device=None, roll=roll_samples)
    test_ds = get_test_set(device=None)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    # 模型
    model = get_model(
        n_classes=args.n_classes,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        channels_multiplier=args.channels_multiplier,
        expansion_rate=args.expansion_rate
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_one_epoch(model=model, train_batch=train_dl, args=args, optimizer=optimizer, mel=mel, mel_augment=mel_augment, device=device)
        train_acc = eval_acc(model, train_dl, mel, device)
        val_acc = eval_acc(model, test_dl, mel, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        # 如果训练和验证准确率都达到50%，提前停止训练
        if train_acc >= 0.5 and val_acc >= 0.5:
            print(f"训练和验证准确率均达到50%，提前停止训练 (epoch={epoch})")
            break

    # 最终测试
    test_acc = eval_acc(model, test_dl, mel, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # 保存模型参数，便于后续剪枝/微调等
    torch.save(model.state_dict(), 'model.pth')
    print('模型参数已保存到 model.pth')

if __name__ == '__main__':
    main()
