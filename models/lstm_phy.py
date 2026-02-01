import torch
import torch.nn as nn


class LSTMPhyModel(nn.Module):
    """
    LSTM-Phy 孔隙压力预测模型，由 LSTM 序列编码器配合全连接层完成回归。
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 编码层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 全连接映射，将隐藏状态压缩为压力预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, x, physical_params=None):
        """
        参数:
            x: 输入序列，形状 [batch_size, sequence_length, input_size]
            physical_params: 物理约束参数（此处未直接使用，接口保留）
        返回:
            形状为 [batch_size, 1] 的孔隙压力预测
        """
        # LSTM 前向传播，获取每个时间步的隐藏状态
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out 形状 [batch_size, seq_len, hidden_size]

        # 仅取最后一个时间步的隐藏状态进行回归
        last_time_step_out = lstm_out[:, -1, :]  # 形状 [batch_size, hidden_size]

        # 通过全连接层得到最终预测
        pp_pred = self.fc(last_time_step_out)  # 形状 [batch_size, 1]

        return pp_pred

    def init_hidden(self, batch_size, device):
        """初始化 LSTM 的隐状态与记忆状态。"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
