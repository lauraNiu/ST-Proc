"""
投影头模块
用于对比学习的特征投影
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    标准投影头
    将编码器的输出投影到对比学习空间
    """

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=64):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            投影后的特征 [batch_size, output_dim]
        """
        return self.net(x)


class MultiLayerProjectionHead(nn.Module):
    """
    多层投影头
    使用更深的网络结构
    """

    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128],
                 output_dim=64, dropout=0.1):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout: Dropout概率
        """
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 最后一层不加激活函数和dropout
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            投影后的特征 [batch_size, output_dim]
        """
        return self.net(x)


class AdaptiveProjectionHead(nn.Module):
    """
    自适应投影头
    根据输入动态调整投影策略
    """

    def __init__(self, input_dim=256, output_dim=64, num_experts=3):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_experts: 专家网络数量
        """
        super().__init__()

        self.num_experts = num_experts

        # 多个专家投影头
        self.experts = nn.ModuleList([
            ProjectionHead(input_dim, input_dim, output_dim)
            for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            投影后的特征 [batch_size, output_dim]
        """
        # 计算门控权重
        gate_weights = self.gate(x)  # [B, num_experts]

        # 应用所有专家
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # [B, num_experts, output_dim]

        # 加权融合
        output = torch.einsum('be,beo->bo', gate_weights, expert_outputs)

        return output


# 测试代码
if __name__ == '__main__':
    batch_size = 4
    input_dim = 256
    output_dim = 64

    x = torch.randn(batch_size, input_dim)

    # 测试标准投影头
    proj1 = ProjectionHead(input_dim, input_dim, output_dim)
    out1 = proj1(x)
    print(f"标准投影头输出: {out1.shape}")  # [4, 64]

    # 测试多层投影头
    proj2 = MultiLayerProjectionHead(
        input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=output_dim
    )
    out2 = proj2(x)
    print(f"多层投影头输出: {out2.shape}")  # [4, 64]

    # 测试自适应投影头
    proj3 = AdaptiveProjectionHead(input_dim, output_dim, num_experts=3)
    out3 = proj3(x)
    print(f"自适应投影头输出: {out3.shape}")  # [4, 64]
