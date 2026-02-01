import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """同时考虑数据监督项与 Eaton 物理约束的复合损失。"""

    def __init__(
        self,
        data_weight=0.8,
        eaton_exponent=2,
        fluid_density=1.03,
        gravity=9.8,
        rho_rock=2.35,
        nac_a=0.0,
        nac_b=0.0,
    ):
        """
        参数:
            data_weight (float): 数据项的权重。
            eaton_exponent (float): Eaton 公式中的指数 c。
            fluid_density (float): 流体密度，单位 g/cm3。
            gravity (float): 重力加速度，单位 m/s2。
        """
        super().__init__()
        self.data_weight = data_weight
        self.eaton_exponent = eaton_exponent
        self.fluid_density = fluid_density
        self.gravity = gravity
        self.rho_rock = rho_rock
        self.nac_a = nac_a
        self.nac_b = nac_b
        self.mse_loss = nn.MSELoss()
        self.eps = 1e-6

    def forward(self, pp_pred, pp_true, physical_params):
        """
        参数:
            pp_pred: 模型输出的孔隙压力 [batch_size, 1]
            pp_true: 观测到的孔隙压力 [batch_size, 1]
            physical_params: 包含 Depth 和 AC 的字典
        """
        data_loss = self.mse_loss(pp_pred, pp_true)

        depth = physical_params['Depth']
        ac = torch.clamp(physical_params['AC'], min=self.eps)
        nac = torch.exp(self.nac_a * depth + self.nac_b)
        nac = torch.clamp(nac, min=self.eps)
        pv = self.rho_rock * self.gravity * depth / 1000.0

        ratio = torch.clamp(nac / ac, min=self.eps, max=1e6)
        pw = self.fluid_density * self.gravity * depth / 1000  # 近似换算为 MPa
        pp_phy = pv - (pv - pw) * torch.pow(ratio, self.eaton_exponent)

        physics_loss = self.mse_loss(pp_pred, pp_phy)
        total_loss = self.data_weight * data_loss + (1 - self.data_weight) * physics_loss

        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
        }
