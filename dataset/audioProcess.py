import torch
import torchaudio
from data import Dataset 

# 音频预处理参数（与train_base.py一致） 
orig_sample_rate = 44100  # 原始采样率
sample_rate = 32000       # 目标采样率
n_fft = 4096              # FFT窗口大小
window_length = 3072      # 窗口长度
hop_length = 500          # 帧移
n_mels = 256              # 梅尔滤波器组数
f_min = 0                 # 最小频率
f_max = None              # 最大频率（None表示采样率一半）
freqm = 48                # 频率掩蔽参数

# 构建音频预处理序列
mel_pipeline = torch.nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate),
    torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=window_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )
)
mel_augment = torch.nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True),
    # 如需时间掩蔽可加：torchaudio.transforms.TimeMasking(timem, iid_masks=True)
)

# 加载数据集
sample_dataset = Dataset()

# 选取第0个音频样本
waveform, label = sample_dataset[0]
print(f"标签: {label}")

# 1. 重采样+梅尔谱
mel_spec = mel_pipeline(waveform)
# 2. 训练时可做数据增强
mel_spec_aug = mel_augment(mel_spec)
# 3. 取对数（防止log(0)加1e-5）
log_mel_spec = (mel_spec_aug + 1e-5).log()

print(f"原始波形形状: {waveform.shape}")
print(f"梅尔谱形状: {mel_spec.shape}")
print(f"增强后梅尔谱形状: {mel_spec_aug.shape}")
print(f"对数梅尔谱形状: {log_mel_spec.shape}")

# 可视化梅尔频谱
import matplotlib.pyplot as plt
plt.imshow(log_mel_spec.squeeze().numpy(), aspect='auto', origin='lower')
plt.title('Log-Mel Spectrogram')
plt.xlabel('Frame')
plt.ylabel('Mel Bin')
plt.colorbar()
plt.show()
