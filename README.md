# DCASE_train Project Structure

This project is a PyTorch implementation for DCASE Task1, covering data processing, model training, and evaluation.

## Directory Structure

```
DCASE_train/
│
├── main.py                # Main training and evaluation script
├── models/
|   └── module.py          # Model architecture and utility functions
├── dataset/               # Dataset and data processing code
│   ├── audioProcess.py    # Audio preprocessing and visualization example
│   ├── data.py            # Dataset class and data loading logic
│   └── meta_dcase_2024/   # Metadata (csv files for data splits)
│       ├── split5.csv
│       ├── split10.csv
│       ├── split25.csv
│       ├── split50.csv
│       ├── split100.csv
│       ├── total.csv
│       ├── valid.csv      # Validation set split
│       └── test.csv       # Test set split
```

## Main File Descriptions

- **main.py**: Project entry point, includes training, validation, and testing processes, and supports model saving.
- **module.py**: Model architecture definitions or utility functions (see file for details).
- **dataset/audioProcess.py**: Audio preprocessing example, including Mel spectrogram, masking augmentation, and visualization.
- **dataset/data.py**: Custom Dataset class for loading audio data and label processing.
- **dataset/meta_dcase_2024/**: Stores csv files for data splits such as train, validation, and test sets.

## Usage Instructions

1. Please modify the dataset path variable in `dataset/data.py` according to your actual data location.
2. No extra packages are required; the framework only depends on basic packages such as pytorch, numpy, pandas, and matplotlib. You just need to install PyTorch and a few simple libraries to prepare the environment.
3. Run `main.py` to train and evaluate the model. Model parameters will be automatically saved after training.
4. Refer to `audioProcess.py` for single-sample audio preprocessing and visualization.

---
For detailed features or code explanations, please refer to the comments in the corresponding Python files. This codebase is well-commented for clarity.
