"""
编码器模块
包含空间编码器、特征编码器和自适应融合编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class GraphAggregationLayer(nn.Module):
    """
    轨迹级图聚合层（Graph Aggregation Layer）
    在 batch 内对邻居嵌入进行加权聚合，增强样本间的结构信息。
    输入：embedding [B, D]，邻接矩阵 adj [B, B]
    输出：增强后的 embedding [B, D]
    采用 GAT 风格的注意力聚合：
      e_ij = LeakyReLU(a^T [W h_i || W h_j])
      alpha_ij = softmax over j ∈ N(i)
      h'_i = ELU(sum_j alpha_ij * W h_j)  +  residual
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   [B, D]  节点嵌入
            adj: [B, B]  0/1 邻接矩阵（不含自环）
        Returns:
            [B, D]  图聚合后的嵌入（含残差）
        """
        B, D = h.shape
        Wh = self.W(h)  # [B, D]

        # 计算注意力分数 e_ij
        Wh_i = Wh.unsqueeze(1).expand(-1, B, -1)  # [B, B, D]
        Wh_j = Wh.unsqueeze(0).expand(B, -1, -1)  # [B, B, D]
        e = self.leaky_relu(self.attn(torch.cat([Wh_i, Wh_j], dim=-1)))  # [B, B, 1]
        e = e.squeeze(-1)  # [B, B]

        # 仅对邻居节点归一化（mask 掉非邻居）
        mask = (adj == 0)
        fill_val = -65504.0 if e.dtype == torch.float16 else -1e9
        e_masked = e.masked_fill(mask, fill_val)

        # 检查是否有邻居（全为 -inf 的行用均匀权重）
        all_masked = mask.all(dim=1, keepdim=True)  # [B, 1]
        alpha = torch.softmax(e_masked, dim=1)  # [B, B]
        alpha = alpha.masked_fill(all_masked.expand_as(alpha), 0.0)  # 孤立节点权重=0
        alpha = self.dropout(alpha)

        # 聚合邻居特征
        agg = torch.mm(alpha, Wh)  # [B, D]

        # 残差连接 + LayerNorm
        out = self.norm(h + F.elu(agg))
        return out


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为序列数据添加位置信息
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# models/encoders.py

class AttentionSpatialEncoder(nn.Module):
    """空间编码器"""

    def __init__(self, coord_dim=2, hidden_dim=128, num_layers=2, dropout=0.2,num_heads=8 ):
        super().__init__()
        # 确保 hidden_dim 能被 num_attention_heads 整除
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) 必须能被 num_attention_heads ({num_heads}) 整除"

        # 坐标投影层
        self.coord_proj = nn.Linear(coord_dim, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead= num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 双向GRU
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, coords, lengths):
        """
        Args:
            coords: 轨迹坐标 [batch_size, max_len, coord_dim]
            lengths: 每个轨迹的实际长度 [batch_size]
        Returns:
            编码后的特征向量 [batch_size, hidden_dim]
        """
        # 投影坐标到高维空间
        x = self.coord_proj(coords)  # [B, L, H]

        # 添加位置编码
        x = self.pos_encoding(x)

        # 创建padding mask
        mask = self._create_padding_mask(lengths, coords.size(1)).to(coords.device)

        # Transformer编码
        trans_out = self.transformer(x, src_key_padding_mask=mask)

        attention_scores = self.attention_pool(trans_out)  # [B, L, 1]

        # 根据数据类型选择合适的填充值
        if attention_scores.dtype == torch.float16:
            fill_value = -65504.0  # FP16 的最小值
        else:
            fill_value = -1e9  # FP32 安全值

        attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), fill_value)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended = (trans_out * attention_weights).sum(dim=1)  # [B, H]

        # GRU编码
        packed = pack_padded_sequence(
            trans_out,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        gru_out = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, H]

        # 融合两种编码
        output = attended + gru_out

        return output

    def _create_padding_mask(self, lengths, max_len):
        """创建padding mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask


class FeatureEncoder(nn.Module):
    """
    特征编码器
    编码手工提取的统计特征
    """

    def __init__(self, feat_dim, hidden_dim=128, dropout=0.2):
        """
        Args:
            feat_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, feat_dim]
        Returns:
            编码后的特征 [batch_size, hidden_dim]
        """
        return self.net(x)

class AdaptiveTrajectoryEncoder(nn.Module):
    """
    自适应轨迹编码器
    融合空间信息和统计特征，使用自适应门控机制
    """

    def __init__(self, feat_dim=36, coord_dim=2, hidden_dim=256, dropout=0.3, num_heads=8, num_layers=2, encoder_mode='adaptive_gate'):
        """
        Args:
            feat_dim: 统计特征维度
            coord_dim: 坐标维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()

        self.encoder_mode = encoder_mode

        # 空间编码器
        self.spatial_encoder = AttentionSpatialEncoder(
            coord_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        )

        # 特征编码器
        self.feature_encoder = FeatureEncoder(
            feat_dim,
            hidden_dim,
            dropout=dropout
        )

        # 自适应门控机制
        if self.encoder_mode == 'adaptive_gate':
            self.adaptive_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, features, coords, lengths):
        """
        Args:
            features: 统计特征 [batch_size, feat_dim]
            coords: 轨迹坐标 [batch_size, max_len, coord_dim]
            lengths: 实际长度 [batch_size]
        Returns:
            融合后的编码 [batch_size, hidden_dim]
        """
        # Ablation 1 (ST Only)
        if self.encoder_mode == 'st_only':
            return self.spatial_encoder(coords, lengths)

        # Ablation 2 (Stats Only)
        if self.encoder_mode == 'stats_only':
            return self.feature_encoder(features)

        # 编码空间信息
        spatial_emb = self.spatial_encoder(coords, lengths)

        # 编码统计特征
        feature_emb = self.feature_encoder(features)

        # Ablation 3 (Simple Fusion)
        if self.encoder_mode == 'simple_fusion':

            combined = torch.cat([spatial_emb, feature_emb], dim=1)
            embedding = self.fusion(combined)
            return embedding

        # ST-ProC (Full) - 'adaptive_gate'
        if self.encoder_mode == 'adaptive_gate':
            combined = torch.cat([spatial_emb, feature_emb], dim=1)
            weights = self.adaptive_gate(combined)  # [B, 2]

            weighted_spatial = spatial_emb * weights[:, 0:1]
            weighted_feature = feature_emb * weights[:, 1:2]

            fused = torch.cat([weighted_spatial, weighted_feature], dim=1)
            embedding = self.fusion(fused)
            return embedding

        # 默认回退
        raise ValueError(f"未知的 encoder_mode: {self.encoder_mode}")

    def get_fusion_weights(self, features, coords, lengths):
        """
        获取融合权重（用于分析）

        Args:
            features: 统计特征
            coords: 轨迹坐标
            lengths: 实际长度
        Returns:
            融合权重 [batch_size, 2]
        """
        with torch.no_grad():
            spatial_emb = self.spatial_encoder(coords, lengths)
            feature_emb = self.feature_encoder(features)
            combined = torch.cat([spatial_emb, feature_emb], dim=1)
            weights = self.adaptive_gate(combined)
        return weights


# 测试代码
if __name__ == '__main__':
    # 测试参数
    batch_size = 4
    seq_len = 100
    feat_dim = 36
    coord_dim = 2
    hidden_dim = 256

    # 创建测试数据
    features = torch.randn(batch_size, feat_dim)
    coords = torch.randn(batch_size, seq_len, coord_dim)
    lengths = torch.tensor([100, 80, 60, 50])

    # 测试编码器
    encoder = AdaptiveTrajectoryEncoder(
        feat_dim=feat_dim,
        coord_dim=coord_dim,
        hidden_dim=hidden_dim
    )

    output = encoder(features, coords, lengths)
    print(f"输出形状: {output.shape}")  # [4, 256]

    # 测试融合权重
    weights = encoder.get_fusion_weights(features, coords, lengths)
    print(f"融合权重: {weights}")
