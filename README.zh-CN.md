# DCASE_train 项目结构说明

本项目为 DCASE Task1任务的 PyTorch 实现，包含数据处理、模型训练、评估等完整流程。

## 目录结构

```
DCASE_train/
│
├── main.py                # 主训练与评估脚本
├── models/
|   └── module.py          # 包含模型结构和工具函数
├── dataset/               # 数据集与数据处理相关代码
│   ├── audioProcess.py    # 音频预处理与可视化示例
│   ├── data.py            # 数据集 Dataset 类及数据加载逻辑
│   └── meta_dcase_2024/   # 元数据（csv 文件，包含数据划分信息）
│       ├── split5.csv
│       ├── split10.csv
│       ├── split25.csv
│       ├── split50.csv
│       ├── split100.csv
│       ├── total.csv
│       ├── valid.csv      # 验证集划分
│       └── test.csv       # 测试集划分
```

## 主要文件说明

- **main.py**：项目主入口，包含训练、验证、测试流程，支持模型保存。
- **module.py**：模型结构定义或相关工具函数（具体内容请参考文件）。
- **dataset/audioProcess.py**：音频预处理示例，包括梅尔谱、掩蔽增强、可视化等。
- **dataset/data.py**：自定义 Dataset 类，负责音频数据的加载、标签处理等。
- **dataset/meta_dcase_2024/**：存放数据划分的 csv 文件，如训练、验证、测试集划分。

## 使用说明

1. 请根据实际数据路径修改 `dataset/data.py` 中的数据集路径变量。
2. 没有多余的包，整体框架仅由pytorch、numpy、pandas、matplotlib等基础包构成，准备环境只需要安装PyTorch以及一些简单的库即可。
3. 运行 `main.py` 进行模型训练与评估，训练结束后会自动保存模型参数。
4. 可参考 `audioProcess.py` 进行单个样本的音频预处理和可视化。

---
如需详细功能或代码说明，请查阅对应的 Python 文件注释，本代码做了相当一部分注释工作。
