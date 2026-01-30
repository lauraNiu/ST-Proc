# models/learners.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class PrototypicalContrastiveLearner:
    """
    原型对比学习框架

    结合了对比学习和原型学习的优势：
    - 对比学习：通过数据增强学习鲁棒的特征表示
    - 原型学习：利用类别原型进行半监督学习
    """

    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            num_classes: int = 11,
            temperature: float = 0.07,
            proto_weight: float = 0.5,
            lr: float = 5e-4,
            weight_decay: float = 1e-4,
            device: str = 'cuda'
    ):
        """
        Args:
            encoder: 轨迹编码器
            projector: 投影头
            num_classes: 类别数量
            temperature: 对比学习温度系数
            proto_weight: 原型损失权重
            lr: 学习率
            weight_decay: 权重衰减
            device: 设备
        """
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        self.proto_weight = proto_weight
        self.num_classes = num_classes
        self.device = device

        # 正确初始化类原型
        proto_dim = projector.net[-1].out_features
        prototypes = torch.randn(num_classes, proto_dim)
        nn.init.xavier_uniform_(prototypes)

        # 先移到设备，再包装为Parameter
        self.prototypes = nn.Parameter(prototypes.to(device))

        # 优化器初始化
        self.optimizer = optim.AdamW(
            [
                {'params': encoder.parameters()},
                {'params': projector.parameters()},
                {'params': [self.prototypes]}
            ],
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            dataloader: 数据加载器

        Returns:
            losses: 包含各项损失的字典
        """
        self.encoder.train()
        self.projector.train()

        total_loss = 0
        total_contrast_loss = 0
        total_proto_loss = 0

        pbar = tqdm(dataloader, desc='Training')

        for batch in pbar:
            losses = self.train_step(batch)

            total_loss += losses['total']
            total_contrast_loss += losses['contrast']
            total_proto_loss += losses['proto']

            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'contrast': f"{losses['contrast']:.4f}",
                'proto': f"{losses['proto']:.4f}"
            })

        self.scheduler.step()

        return {
            'total': total_loss / len(dataloader),
            'contrast': total_contrast_loss / len(dataloader),
            'proto': total_proto_loss / len(dataloader)
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步训练

        Args:
            batch: 批次数据

        Returns:
            losses: 各项损失
        """
        self.optimizer.zero_grad()

        coords = batch['coords'].to(self.device)
        features = batch['features'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        labels = batch['labels'].to(self.device)

        # 生成两个不同的增强视图
        view1_coords = self._augment_batch(coords, lengths)
        view2_coords = self._augment_batch(coords, lengths)

        # 编码 + 投影
        emb1 = self.encoder(features, view1_coords, lengths)
        emb2 = self.encoder(features, view2_coords, lengths)

        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        # 1. 对比学习损失（无监督）
        contrast_loss = self.nt_xent_loss(z1, z2)

        # 2. 原型对比损失（半监督）
        labeled_mask = labels >= 0
        if labeled_mask.sum() > 0:
            labeled_z = z1[labeled_mask]
            labeled_y = labels[labeled_mask]
            proto_loss = self.prototypical_loss(labeled_z, labeled_y)
        else:
            proto_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = contrast_loss + self.proto_weight * proto_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.projector.parameters()) +
            [self.prototypes],
            max_norm=1.0
        )

        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'contrast': contrast_loss.item(),
            'proto': proto_loss.item()
        }

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent 对比学习损失（SimCLR）

        Args:
            z1: 第一个视图的投影 [B, D]
            z2: 第二个视图的投影 [B, D]

        Returns:
            loss: 对比损失
        """
        batch_size = z1.size(0)

        # L2归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 拼接所有样本
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # 移除对角线（自身相似度）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 正样本对的索引
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, pos_indices)

        return loss

    def prototypical_loss(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        原型对比损失

        将样本拉向对应类别的原型，推离其他类别的原型

        Args:
            embeddings: 样本嵌入 [N, D]
            labels: 样本标签 [N]

        Returns:
            loss: 原型损失
        """
        # L2归一化
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # 计算样本到所有原型的相似度
        logits = torch.mm(embeddings, prototypes.t()) / self.temperature

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def _augment_batch(
            self,
            coords: torch.Tensor,
            lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        批量数据增强

        Args:
            coords: 坐标序列 [B, T, 2]
            lengths: 有效长度 [B]

        Returns:
            augmented_coords: 增强后的坐标
        """
        augmented_coords = coords.clone()

        for i in range(coords.size(0)):
            valid_len = lengths[i].item()

            # 随机噪声
            if np.random.rand() > 0.5:
                noise = torch.randn_like(augmented_coords[i, :valid_len]) * 0.0001
                augmented_coords[i, :valid_len] += noise

            # 随机掩码
            if np.random.rand() > 0.5:
                num_mask = max(1, int(valid_len * 0.15))
                mask_indices = torch.randperm(valid_len)[:num_mask]
                augmented_coords[i, mask_indices] = 0

        return augmented_coords

    @torch.no_grad()
    def extract_embeddings(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取所有样本的嵌入

        Args:
            dataloader: 数据加载器

        Returns:
            embeddings: 嵌入矩阵 [N, D]
            labels: 标签数组 [N]
        """
        self.encoder.eval()

        all_embeddings = []
        all_labels = []

        for batch in dataloader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].cpu().numpy()

            emb = self.encoder(features, coords, lengths)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels)

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.hstack(all_labels)

        return all_embeddings, all_labels


class SemiSupervisedPrototypicalLearner(PrototypicalContrastiveLearner):
    """
    半监督原型对比学习框架

    在原型对比学习的基础上，增加：
    1. 伪标签学习
    2. 一致性正则化（教师-学生模型）
    """

    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            num_classes: int = 11,
            temperature: float = 0.07,
            proto_weight: float = 0.5,
            pseudo_weight: float = 0.3,
            consistency_weight: float = 0.2,
            lr: float = 5e-4,
            weight_decay: float = 1e-4,
            ema_decay: float = 0.999,
            device: str = 'cuda'
    ):
        """
        Args:
            pseudo_weight: 伪标签损失权重
            consistency_weight: 一致性损失权重
            ema_decay: 教师模型的EMA衰减系数
        """
        super().__init__(
            encoder, projector, num_classes, temperature,
            proto_weight, lr, weight_decay, device
        )

        self.pseudo_weight = pseudo_weight
        self.consistency_weight = consistency_weight
        self.ema_decay = ema_decay

        # 创建教师模型（EMA）
        self.teacher_encoder = self._create_teacher_model(encoder)
        self.teacher_projector = self._create_teacher_model(projector)

    def _create_teacher_model(self, student_model: nn.Module) -> nn.Module:
        """
        创建教师模型（参数从学生模型复制）

        Args:
            student_model: 学生模型

        Returns:
            teacher_model: 教师模型（冻结参数）
        """
        import copy
        teacher = copy.deepcopy(student_model).to(self.device)

        # 冻结教师模型参数
        for param in teacher.parameters():
            param.requires_grad = False

        return teacher

    def update_teacher(self):
        """
        使用EMA更新教师模型参数

        teacher = decay * teacher + (1 - decay) * student
        """
        with torch.no_grad():
            # 更新encoder
            for teacher_param, student_param in zip(
                    self.teacher_encoder.parameters(),
                    self.encoder.parameters()
            ):
                teacher_param.data = (
                        self.ema_decay * teacher_param.data +
                        (1 - self.ema_decay) * student_param.data
                )

            # 更新projector
            for teacher_param, student_param in zip(
                    self.teacher_projector.parameters(),
                    self.projector.parameters()
            ):
                teacher_param.data = (
                        self.ema_decay * teacher_param.data +
                        (1 - self.ema_decay) * student_param.data
                )

    def train_step_semi_supervised(
            self,
            batch: Dict[str, torch.Tensor],
            pseudo_labels_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        半监督训练步骤

        Args:
            batch: 批次数据
            pseudo_labels_dict: 伪标签字典，包含 'labels' 和 'confidences'

        Returns:
            losses: 各项损失
        """
        self.optimizer.zero_grad()

        coords = batch['coords'].to(self.device)
        features = batch['features'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        labels = batch['labels'].to(self.device)
        batch_indices = batch['indices']

        # 生成增强视图
        view1_coords = self._augment_batch(coords, lengths)
        view2_coords = self._augment_batch(coords, lengths)

        # 学生模型编码
        emb1 = self.encoder(features, view1_coords, lengths)
        emb2 = self.encoder(features, view2_coords, lengths)

        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        # 1. 对比学习损失
        contrast_loss = self.nt_xent_loss(z1, z2)

        # 2. 原型损失（真实标签）
        labeled_mask = labels >= 0
        if labeled_mask.sum() > 0:
            proto_loss = self.prototypical_loss(
                z1[labeled_mask],
                labels[labeled_mask]
            )
        else:
            proto_loss = torch.tensor(0.0, device=self.device)

        # 3. 伪标签损失
        pseudo_loss = torch.tensor(0.0, device=self.device)
        if pseudo_labels_dict is not None:
            pseudo_loss = self._compute_pseudo_loss(
                z1, labels, batch_indices, pseudo_labels_dict
            )

        # 4. 一致性正则化
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(
                features, coords, view1_coords, lengths
            )

        # 总损失
        total_loss = (
                contrast_loss +
                self.proto_weight * proto_loss +
                self.pseudo_weight * pseudo_loss +
                self.consistency_weight * consistency_loss
        )

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.projector.parameters()) +
            [self.prototypes],
            max_norm=1.0
        )
        self.optimizer.step()

        # 更新教师模型
        self.update_teacher()

        return {
            'total': total_loss.item(),
            'contrast': contrast_loss.item(),
            'proto': proto_loss.item(),
            'pseudo': pseudo_loss.item(),
            'consistency': consistency_loss.item()
        }

    def _compute_pseudo_loss(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            batch_indices: torch.Tensor,
            pseudo_labels_dict: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """
        计算伪标签损失

        Args:
            embeddings: 样本嵌入
            labels: 真实标签
            batch_indices: 批次内样本的全局索引
            pseudo_labels_dict: 伪标签字典

        Returns:
            loss: 伪标签损失
        """
        try:
            # 获取当前batch的伪标签
            batch_pseudo_labels = []
            batch_pseudo_confs = []

            for idx in batch_indices:
                idx_int = int(idx)
                if idx_int < len(pseudo_labels_dict['labels']):
                    batch_pseudo_labels.append(
                        pseudo_labels_dict['labels'][idx_int]
                    )
                    batch_pseudo_confs.append(
                        pseudo_labels_dict['confidences'][idx_int]
                    )
                else:
                    batch_pseudo_labels.append(-1)
                    batch_pseudo_confs.append(0.0)

            batch_pseudo_labels = np.array(batch_pseudo_labels)
            batch_pseudo_confs = np.array(batch_pseudo_confs)

            # 找到有伪标签且无真实标签的样本
            pseudo_mask = (batch_pseudo_labels >= 0) & (labels.cpu().numpy() < 0)

            if pseudo_mask.sum() == 0:
                return torch.tensor(0.0, device=self.device)

            # 准备伪标签数据
            pseudo_labels_tensor = torch.LongTensor(
                batch_pseudo_labels[pseudo_mask]
            ).to(self.device)

            pseudo_conf_tensor = torch.FloatTensor(
                batch_pseudo_confs[pseudo_mask]
            ).to(self.device)

            pseudo_z = embeddings[pseudo_mask]

            # 计算原型损失
            pseudo_z_norm = F.normalize(pseudo_z, dim=1)
            prototypes_norm = F.normalize(self.prototypes, dim=1)

            logits = torch.mm(
                pseudo_z_norm,
                prototypes_norm.t()
            ) / self.temperature

            loss = F.cross_entropy(
                logits,
                pseudo_labels_tensor,
                reduction='none'
            )

            # 置信度加权
            loss = (loss * pseudo_conf_tensor).mean()

            return loss

        except Exception as e:
            print(f"⚠️ 伪标签损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device)

    def _compute_consistency_loss(
            self,
            features: torch.Tensor,
            coords: torch.Tensor,
            augmented_coords: torch.Tensor,
            lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性正则化损失（教师-学生）

        Args:
            features: 特征
            coords: 原始坐标
            augmented_coords: 增强坐标
            lengths: 长度

        Returns:
            loss: 一致性损失
        """
        # 教师模型预测（无梯度）
        with torch.no_grad():
            teacher_emb = self.teacher_encoder(features, coords, lengths)
            teacher_z = self.teacher_projector(teacher_emb)
            teacher_z = F.normalize(teacher_z, dim=1)

        # 学生模型预测
        student_emb = self.encoder(features, augmented_coords, lengths)
        student_z = self.projector(student_emb)
        student_z = F.normalize(student_z, dim=1)

        # MSE损失
        loss = F.mse_loss(student_z, teacher_z)

        return loss


def create_learner(
        learner_type: str,
        encoder: nn.Module,
        projector: nn.Module,
        config: dict,
        device: str = 'cuda'
):
    """
    创建学习器的工厂函数

    Args:
        learner_type: 学习器类型 ('standard' 或 'semi_supervised')
        encoder: 编码器
        projector: 投影头
        config: 配置字典
        device: 设备

    Returns:
        learner: 学习器实例
    """
    if learner_type == 'standard':
        return PrototypicalContrastiveLearner(
            encoder=encoder,
            projector=projector,
            num_classes=config.get('num_classes', 11),
            temperature=config.get('temperature', 0.07),
            proto_weight=config.get('proto_weight', 0.5),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            device=device
        )

    elif learner_type == 'semi_supervised':
        return SemiSupervisedPrototypicalLearner(
            encoder=encoder,
            projector=projector,
            num_classes=config.get('num_classes', 11),
            temperature=config.get('temperature', 0.07),
            proto_weight=config.get('proto_weight', 0.5),
            pseudo_weight=config.get('pseudo_weight', 0.3),
            consistency_weight=config.get('consistency_weight', 0.2),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            ema_decay=config.get('ema_decay', 0.999),
            device=device
        )

    else:
        raise ValueError(f"Unknown learner type: {learner_type}")
