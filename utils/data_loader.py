from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

FEATURE_COLUMNS = ['Depth', 'AC', 'DEN', 'Ip']
TARGET_COLUMN = 'Pp'
SCALER_PATH = Path('models') / 'scaler.pkl'


class PorePressureDataset(Dataset):
    """
    用于孔隙压力预测的序列数据集，内部存储已标准化的窗口序列与标注。
    每个样本张量形状为 [sequence_length, input_size]，配套一个标量标签。
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int):
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have the same number of rows.")
        if features.shape[0] < sequence_length:
            raise ValueError("Not enough samples to create a single sequence.")

        self.sequence_length = sequence_length
        sequences, labels = self._create_sequences(features, targets)
        self.x = torch.from_numpy(sequences).float()
        self.y = torch.from_numpy(labels).float().unsqueeze(-1)

    def _create_sequences(self, data: np.ndarray, targets: np.ndarray):
        """将连续样本按滑动窗口划分，并与窗口末端的标签对齐。"""
        xs, ys = [], []
        for end_idx in range(self.sequence_length, len(data) + 1):
            xs.append(data[end_idx - self.sequence_length:end_idx])
            ys.append(targets[end_idx - 1])
        return np.stack(xs), np.array(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _persist_scaler(scaler: StandardScaler, target_path: Path = SCALER_PATH):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, target_path)


def _load_dataframe(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_file}: {missing_cols}")
    # 保持原始顺序（默认为深度或时间顺序）
    return df.reset_index(drop=True)


def get_data_loaders(csv_file, sequence_length, batch_size, train_ratio=0.8):
    """
    随机划分 sequence 级别的训练/验证 DataLoader（保持序列连续性），仅用训练序列中的特征拟合标准化器。
    """
    df = _load_dataframe(csv_file)
    num_rows = len(df)
    if num_rows < sequence_length + 1:
        raise ValueError("样本数量不足，无法同时构造至少一个训练序列和一个验证序列。")

    feature_values = df[FEATURE_COLUMNS].values
    target_values = df[TARGET_COLUMN].values

    # 所有可用的序列结束下标（包含 seq_len-1 到 num_rows-1）
    end_indices = np.arange(sequence_length - 1, num_rows)
    if end_indices.size < 2:
        raise ValueError("无法划分 train/val：可用序列数过少。")

    perm = np.random.permutation(end_indices)
    train_count = max(1, int(train_ratio * len(end_indices)))
    val_count = len(end_indices) - train_count
    if val_count < 1:
        raise ValueError("验证序列数为 0，请调整 train_ratio 或增加数据量。")

    train_end = perm[:train_count]
    val_end = perm[train_count:]

    def _collect_rows(end_idxs):
        rows = set()
        for end_idx in end_idxs:
            rows.update(range(end_idx - sequence_length + 1, end_idx + 1))
        return sorted(rows)

    # 仅使用训练序列涉及到的行来拟合 scaler
    train_rows_for_scaler = _collect_rows(train_end)
    scaler = StandardScaler()
    scaler.fit(feature_values[train_rows_for_scaler])
    scaled_all = scaler.transform(feature_values)
    _persist_scaler(scaler)

    def _build_sequences(end_idxs):
        seqs, labels = [], []
        for end_idx in end_idxs:
            start = end_idx - sequence_length + 1
            seqs.append(scaled_all[start : end_idx + 1])
            labels.append(target_values[end_idx])
        return np.stack(seqs), np.array(labels)

    train_seqs, train_labels = _build_sequences(train_end)
    val_seqs, val_labels = _build_sequences(val_end)

    train_dataset = TensorDataset(
        torch.from_numpy(train_seqs).float(),
        torch.from_numpy(train_labels).float().unsqueeze(-1),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_seqs).float(),
        torch.from_numpy(val_labels).float().unsqueeze(-1),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, scaler


if __name__ == "__main__":
    train_loader, val_loader, scaler = get_data_loaders(
        csv_file="data/raw/N873_fortrain.csv",
        sequence_length=10,
        batch_size=32,
        train_ratio=0.8,
    )
    print(train_loader)
    print(val_loader)
    print(scaler)
