"""
投影头模块
用于对比学习的特征投影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    分层分类头：
    - shared trunk
    - fine head over all classes
    - optional coarse head whose log-prob acts as a prior on fine logits
    """
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1,
        hidden_dim: int = None,
        coarse_groups=None,
        use_hierarchical: bool = True,
        hierarchical_prior_scale: float = 0.7,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or input_dim)
        self.num_classes = int(num_classes)
        self.use_hierarchical = bool(use_hierarchical)
        self.hierarchical_prior_scale = float(hierarchical_prior_scale)
        self.coarse_groups = coarse_groups or [[i] for i in range(self.num_classes)]
        self.register_buffer(
            'fine_to_coarse',
            self._build_fine_to_coarse(self.num_classes, self.coarse_groups),
            persistent=False,
        )
        self.register_buffer(
            'logit_adjustment',
            torch.zeros(self.num_classes, dtype=torch.float32),
            persistent=False,
        )

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fine_head = nn.Linear(hidden_dim, self.num_classes)
        self.coarse_head = (
            nn.Linear(hidden_dim, len(self.coarse_groups))
            if self.use_hierarchical and len(self.coarse_groups) > 1
            else None
        )

    @staticmethod
    def _build_fine_to_coarse(num_classes: int, coarse_groups) -> torch.Tensor:
        mapping = torch.zeros(num_classes, dtype=torch.long)
        assigned = set()
        for group_idx, group in enumerate(coarse_groups):
            for cls in group:
                cls = int(cls)
                if 0 <= cls < num_classes:
                    mapping[cls] = group_idx
                    assigned.add(cls)
        for cls in range(num_classes):
            if cls not in assigned:
                mapping[cls] = min(cls, max(0, len(coarse_groups) - 1))
        return mapping

    def set_logit_adjustment(self, adjustment: torch.Tensor):
        if adjustment is None:
            self.logit_adjustment.zero_()
            return
        adj = torch.as_tensor(adjustment, dtype=torch.float32, device=self.logit_adjustment.device).view(-1)
        if adj.numel() != self.num_classes:
            raise ValueError(f'logit adjustment size mismatch: expected {self.num_classes}, got {adj.numel()}')
        self.logit_adjustment.copy_(adj)

    def forward(self, h: torch.Tensor, return_aux: bool = False):
        """
        Args:
            h: [B, input_dim]
            return_aux: whether to return fine/coarse auxiliary logits
        """
        feat = self.trunk(h)
        fine_logits = self.fine_head(feat)
        coarse_logits = None

        if self.coarse_head is not None:
            coarse_logits = self.coarse_head(feat)
            coarse_log_probs = F.log_softmax(coarse_logits, dim=1)
            fine_logits = fine_logits + self.hierarchical_prior_scale * coarse_log_probs[:, self.fine_to_coarse]

        if self.logit_adjustment.numel() == fine_logits.size(1):
            fine_logits = fine_logits + self.logit_adjustment.to(fine_logits.device)

        if not return_aux:
            return fine_logits

        return {
            'logits': fine_logits,
            'fine_logits': fine_logits,
            'coarse_logits': coarse_logits,
            'features': feat,
        }


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


if __name__ == '__main__':
    batch_size = 4
    input_dim = 256
    output_dim = 64

    x = torch.randn(batch_size, input_dim)

    # 标准投影头
    proj1 = ProjectionHead(input_dim, input_dim, output_dim)
    out1 = proj1(x)
    print(f"标准投影头输出: {out1.shape}")  # [4, 64]

    # 多层投影头
    proj2 = MultiLayerProjectionHead(
        input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=output_dim
    )
    out2 = proj2(x)
    print(f"多层投影头输出: {out2.shape}")  # [4, 64]

    # 自适应投影头
    proj3 = AdaptiveProjectionHead(input_dim, output_dim, num_experts=3)
    out3 = proj3(x)
    print(f"自适应投影头输出: {out3.shape}")  # [4, 64]
