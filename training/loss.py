# training/loss.py
"""
Loss functions for trajectory learning.
Includes contrastive loss, prototypical loss, and consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_prototype_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.07,
    aggregation: str = 'max',
    pool_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute class logits for either single-prototype [C, D] or
    multi-prototype [C, K, D] tensors.

    aggregation:
    - max: hard nearest prototype
    - mean: average over class prototypes
    - logsumexp: soft mixture over class prototypes
    """
    z = F.normalize(embeddings, dim=1)

    if prototypes.dim() == 2:
        p = F.normalize(prototypes, dim=1)
        return torch.mm(z, p.t()) / temperature

    if prototypes.dim() != 3:
        raise ValueError(f'Unsupported prototype shape: {tuple(prototypes.shape)}')

    p = F.normalize(prototypes, dim=2)
    sims = torch.einsum('bd,ckd->bck', z, p)
    aggregation = str(aggregation).lower()

    if aggregation == 'max':
        pooled = sims.max(dim=2).values
    elif aggregation == 'mean':
        pooled = sims.mean(dim=2)
    elif aggregation == 'logsumexp':
        pool_temperature = max(float(pool_temperature), 1e-4)
        pooled = torch.logsumexp(sims / pool_temperature, dim=2) * pool_temperature
        pooled = pooled - pool_temperature * torch.log(
            torch.tensor(sims.size(2), device=sims.device, dtype=sims.dtype)
        )
    else:
        raise ValueError(f'Unknown prototype aggregation: {aggregation}')

    return pooled / temperature


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    用于无监督对比学习

    Reference:
        SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    """

    def __init__(self, temperature: float = 0.07, device: str = 'cuda'):
        """
        Args:
            temperature: Temperature parameter for scaling
            device: Device to run computations
        """
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two views.

        Args:
            z_i: Embeddings of view 1, shape [batch_size, dim]
            z_j: Embeddings of view 2, shape [batch_size, dim]

        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2*batch_size, 2*batch_size]

        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=representations.device)
        # dtype 安全的填充值
        fill_value = -65504.0 if similarity_matrix.dtype == torch.float16 else -1e9
        similarity_matrix = similarity_matrix.masked_fill(mask, fill_value)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Positive pairs: (i, batch_size+i) and (batch_size+i, i)
        positives = torch.cat([
            similarity_matrix[range(batch_size), range(batch_size, 2 * batch_size)],
            similarity_matrix[range(batch_size, 2 * batch_size), range(batch_size)]
        ], dim=0)  # [2*batch_size]

        # Compute log softmax
        nominator = torch.exp(positives)
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)

        loss = -torch.log(nominator / denominator)

        return loss.mean()


class PrototypicalLoss(nn.Module):
    """
    支持 CosFace 风格 margin 与类别权重的原型损失
    logits = <normalize(z), normalize(p)> / T
    """
    def __init__(self, temperature: float = 0.07,
                 margin: float = 0.0,
                 class_weights=None,
                 prototype_pooling: str = 'max',
                 prototype_pool_temperature: float = 1.0,
                 device: str = 'cuda'):
        super().__init__()
        self.temperature = temperature
        self.margin = float(margin)
        self.prototype_pooling = str(prototype_pooling).lower()
        self.prototype_pool_temperature = float(prototype_pool_temperature)
        self.device = device

        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                w = torch.tensor(class_weights, dtype=torch.float32)
            elif isinstance(class_weights, torch.Tensor):
                w = class_weights.float()
            else:
                raise ValueError("class_weights must be list/tuple/Tensor or None")
            self.register_buffer('full_class_weights', w)
        else:
            self.full_class_weights = None
        # ce_loss is built dynamically in forward to handle variable num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        logits = compute_prototype_logits(
            embeddings,
            prototypes,
            temperature=self.temperature,
            aggregation=self.prototype_pooling,
            pool_temperature=self.prototype_pool_temperature,
        )

        if self.margin > 0.0:
            one_hot = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1.0)
            logits = logits - one_hot * self.margin

        # 动态匹配 class_weights 到当前类别数量
        if self.full_class_weights is not None:
            n = logits.shape[1]
            w = self.full_class_weights[:n].to(embeddings.device)
            loss_fn = nn.CrossEntropyLoss(weight=w)
        else:
            loss_fn = self.ce_loss

        return loss_fn(logits, labels)


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for semi-supervised learning.
    用于教师-学生模型之间的一致性约束

    Measures the difference between teacher and student predictions.
    """

    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: Type of consistency loss ('mse', 'kl', 'cosine')
        """
        super().__init__()
        self.loss_type = loss_type.lower()

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'kl':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif self.loss_type == 'cosine':
            self.loss_fn = lambda x, y: 1 - F.cosine_similarity(x, y, dim=1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(
            self,
            student_output: torch.Tensor,
            teacher_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.

        Args:
            student_output: Student model output, shape [batch_size, dim]
            teacher_output: Teacher model output, shape [batch_size, dim]

        Returns:
            Scalar loss value
        """
        if self.loss_type == 'kl':
            # For KL divergence, apply softmax
            student_output = F.log_softmax(student_output, dim=1)
            teacher_output = F.softmax(teacher_output, dim=1)

        loss = self.loss_fn(student_output, teacher_output)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    三元组损失：拉近正样本，推远负样本
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings, shape [batch_size, dim]
            positive: Positive embeddings, shape [batch_size, dim]
            negative: Negative embeddings, shape [batch_size, dim]

        Returns:
            Scalar loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    监督对比学习损失

    Reference:
        Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    """

    def __init__(self, temperature: float = 0.07, device: str = 'cuda'):
        """
        Args:
            temperature: Temperature parameter
            device: Device for computation
        """
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(
            self,
            features: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Embeddings, shape [batch_size, dim]
            labels: Ground truth labels, shape [batch_size]

        Returns:
            Scalar loss value
        """
        batch_size = features.size(0)

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t())  # [batch_size, batch_size]

        # Create label mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(self.device)  # [batch_size, batch_size]

        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    用于处理类别不平衡问题

    Reference:
        Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions, shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]

        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CenterLoss(nn.Module):
    """
    Center Loss for deep feature learning.
    用于学习类内紧凑的特征表示

    Reference:
        A Discriminative Feature Learning Approach for Deep Face Recognition
    """

    def __init__(self, num_classes: int, feat_dim: int, device: str = 'cuda'):
        """
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            device: Device for computation
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Initialize class centers
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim).to(device)
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.

        Args:
            features: Feature embeddings, shape [batch_size, feat_dim]
            labels: Ground truth labels, shape [batch_size]

        Returns:
            Scalar loss value
        """
        batch_size = features.size(0)

        # Get centers for each sample
        centers_batch = self.centers[labels]  # [batch_size, feat_dim]

        # Compute distances
        loss = F.mse_loss(features, centers_batch)

        return loss

class GraphSmoothnessLoss(nn.Module):
    """
    归一化图平滑损失：L_graph = Tr(Z^T L_sym Z) / B
    使用对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}，
    特征值范围 [0,2]，与 batch 大小无关，保证非负。
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # z: [B, D] (L2-normalized), adj: [B, B] (float, 对称, 非负)
        deg = adj.sum(dim=1).clamp(min=1e-8)           # [B]
        d_inv_sqrt = deg.pow(-0.5)                      # [B]
        # A_norm = D^{-1/2} A D^{-1/2}
        A_norm = adj * d_inv_sqrt.unsqueeze(1) * d_inv_sqrt.unsqueeze(0)
        # L_sym = I - A_norm
        L_sym = torch.eye(z.size(0), device=z.device) - A_norm
        # Tr(Z^T L_sym Z) = ||Z||_F^2 - Tr(Z^T A_norm Z)
        # 等价但更高效：sum_ij A_norm_ij * (zi·zj)
        ztLz = torch.trace(z.t() @ L_sym @ z)
        return ztLz.clamp(min=0.0) / z.size(0)


class NeighborContrastiveLoss(nn.Module):
    """
    邻居对比损失：把 batch 内 kNN 邻居视为正例，其余（除自身）为负例。
    对每个样本 i：
      loss_i = -(1/|P_i|) * sum_{p in P_i} log( exp(sim(i,p)/t) / sum_{a!=i} exp(sim(i,a)/t) )
    要求 z 已归一化。
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # z: [B, D] (normalized), adj: [B, B] (0/1 或权重，必须对角为0)
        B = z.size(0)
        sim = torch.matmul(z, z.t()) / self.t                    # [B,B]
        mask_self = torch.eye(B, device=z.device).bool()
        sim = sim.masked_fill(mask_self, -1e4)                   # 去掉自身

        # 正例掩码（邻居）
        pos_mask = (adj > 0).float()                             # [B,B]
        pos_count = pos_mask.sum(dim=1)                          # [B]

        # log softmax
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [B,B]

        # 仅对存在正例的样本求均值
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # 对每个 i，将正例的 log_prob 相加并按 |P_i| 均值
        loss_per_i = torch.zeros(B, device=z.device)
        pos_sum = (log_prob * pos_mask).sum(dim=1)               # [B]
        loss_per_i[valid] = - pos_sum[valid] / pos_count[valid]

        return loss_per_i[valid].mean()



# Alias for backward compatibility
NTXentLoss = ContrastiveLoss
