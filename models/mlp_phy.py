import torch
import torch.nn as nn


class MLPPhyModel(nn.Module):
    """
    MLP-Phy 孔隙压力预测模型，仅使用序列最后一个时间步的特征进行前馈回归，
    以便与 LSTM-Phy 共用训练流程与物理约束。
    """

    def __init__(self, input_size, hidden_sizes=(128, 64), output_size=1, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, physical_params=None):
        """
        参数:
            x: 输入序列 [batch_size, seq_len, input_size]
            physical_params: 物理约束需要的参数（此处未直接使用）
        返回:
            [batch_size, 1] 的孔隙压力预测值
        """
        last_step = x[:, -1, :]
        return self.mlp(last_step)
