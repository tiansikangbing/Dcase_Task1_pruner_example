import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import os
import torchaudio
from typing import Optional, List

dataset_dir = "D:\local_repository\TAU-urban-acoustic-scenes-2022-mobile-development"

# Dataset configuration
dataset_config = {
    "dataset_name": "TAU2022",
    "split_path": "dataset\meta_dcase_2024\split5.csv",
    "test_split_csv": "dataset\meta_dcase_2024\\test.csv",
    "validation_split_csv": "dataset\meta_dcase_2024\\valid.csv"
}

class Dataset():
    """
    Loads metadata and provides access to audio samples.
    """

    def __init__(self, meta_csv: str = dataset_config["split_path"]):
        """
        Initializes the dataset.

        Args:
            meta_csv (str): Path to the dataset metadata CSV file.
        """
        # 读取元数据CSV文件
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df["filename"].values
        # 读取数据标签，LabelEcoder()将标签转换为数字
        self.labels = torch.tensor(LabelEncoder().fit_transform(df["scene_label"]), dtype=torch.long)

    def __getitem__(self, index: int):
        """Loads an audio sample and corresponding metadata."""
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, _ = torchaudio.load(audio_path)
        return waveform,self.labels[index]

    def __len__(self) -> int:
        return len(self.files)

# 对音频数据进行“时移增强”，让模型看到同一音频的不同“起点”，增强模型的鲁棒性，防止模型只记住音频的绝对位置特征
class TimeShiftDataset(Dataset):
    """
    A dataset implementing time shifting of waveforms.
    """

    def __init__(self, dataset: Dataset, shift_range: int, axis: int = 1):
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index: int):
        waveform, label = self.dataset[index]
        shift = np.random.randint(-self.shift_range, self.shift_range + 1)
        return waveform.roll(shift, self.axis), label

    def __len__(self) -> int:
        return len(self.dataset)

# 返回训练集
def get_training_set(device: Optional[str] = None, roll: int = 0) -> Dataset:
    dataset = Dataset(dataset_config["split_path"])
    return TimeShiftDataset(dataset, shift_range=roll) if roll else dataset

# 返回测试集
def get_test_set(device: Optional[str] = None) -> Dataset:
    """Returns the test dataset."""
    return Dataset(dataset_config["test_split_csv"])

# 返回验证集
def get_validation_set(device: Optional[str] = None) -> Dataset:
    """Returns the validation dataset."""
    return Dataset(dataset_config["validation_split_csv"])
