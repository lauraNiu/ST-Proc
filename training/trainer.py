"""
Core training logic for trajectory models.
Supports both supervised and semi-supervised learning.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from data.dataset import traj_collate_fn
from utils.logger import get_logger
from .scheduler import WarmupCosineScheduler
from .loss import ContrastiveLoss, PrototypicalLoss, ConsistencyLoss
import logging
import torch.nn.functional as F
from .loss import ContrastiveLoss, PrototypicalLoss, ConsistencyLoss, GraphSmoothnessLoss, NeighborContrastiveLoss
from training.pseudo_label import graph_label_propagation, fuse_pseudo_labels


class Trainer:
    """
    Base trainer for prototypical contrastive learning.

    Features:
    - Automatic mixed precision (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint management
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: str = './experiments/exp_001',
        **kwargs
    ):
        """
        Args:
            encoder: Trajectory encoder model
            projector: Projection head
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: torch.device
            experiment_dir: Directory to save checkpoints and logs
        """
        self.encoder = encoder
        self.projector = projector
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)

        # Create directories
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.log_dir = self.experiment_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = get_logger(
            exp_name='Trainer',
            log_dir=str(self.experiment_dir.parent),
            level=logging.INFO
        )

        # Initialize prototypes
        self.num_classes = config.get('num_classes', 11)
        projection_dim = config.get('projection_dim', 64)

        # 创建原型参数
        prototypes_init = torch.randn(self.num_classes, projection_dim)
        nn.init.xavier_uniform_(prototypes_init)
        self.prototypes = nn.Parameter(prototypes_init, requires_grad=True)
        self.prototypes.data = self.prototypes.data.to(device)

        # Loss functions
        self.contrast_loss_fn = ContrastiveLoss(
            temperature=config.get('temperature', 0.07)
        )
        self.proto_loss_fn = PrototypicalLoss(
            temperature=config.get('temperature', 0.07),
            margin=float(config.get('proto_margin', 0.0)),
            class_weights=config.get('class_weights', None),
            device=str(self.device)
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # AMP scaler
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Loss weights
        self.proto_weight = config.get('proto_weight', 0.5)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'contrast_loss': [],
            'proto_loss': [],
            'lr': []
        }

        self.logger.info(f"Trainer initialized with config: {config}")
        self.logger.info(f"Model parameters: {self._count_parameters():,}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        params = list(self.encoder.parameters()) + \
                 list(self.projector.parameters()) + \
                 [self.prototypes]

        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('lr', 5e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        max_epochs = self.config.get('epochs', 100)

        if scheduler_name == 'cosine':
            scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 5),
                max_epochs=max_epochs,
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.projector.parameters() if p.requires_grad) + \
               self.prototypes.numel()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.projector.train()

        epoch_metrics = {
            'total_loss': 0.0,
            'contrast_loss': 0.0,
            'proto_loss': 0.0
        }

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config.get("epochs", 100)}'
        )

        for batch_idx, batch in enumerate(pbar):
            loss_dict = self.train_step(batch)

            # Update metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'contrast': f"{loss_dict['contrast_loss']:.4f}",
                'proto': f"{loss_dict['proto_loss']:.4f}"
            })

            self.global_step += 1

        # Compute epoch averages
        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        return epoch_metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move data to device
        coords = batch['coords'].to(self.device)
        features = batch['features'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass with AMP
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self._compute_loss(coords, features, lengths, labels)
        else:
            loss_dict = self._compute_loss(coords, features, lengths, labels)

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss_dict['total_loss_tensor']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.projector.parameters()) +
                [self.prototypes],
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict['total_loss_tensor'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.projector.parameters()) +
                [self.prototypes],
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.optimizer.step()

        # Return scalar losses
        return {
            'total_loss': loss_dict['total_loss'],
            'contrast_loss': loss_dict['contrast_loss'],
            'proto_loss': loss_dict['proto_loss']
        }

    # training/trainer.py
    def _compute_loss(
            self,
            coords: torch.Tensor,
            features: torch.Tensor,
            lengths: torch.Tensor,
            labels: torch.Tensor
    ) -> Dict[str, Any]:
        # two augmented views (coords + features)
        view1_coords = self._augment_batch(coords, lengths)
        view2_coords = self._augment_batch(coords, lengths)
        feat1 = self._augment_features(features)
        feat2 = self._augment_features(features)

        emb1 = self.encoder(feat1, view1_coords, lengths)
        emb2 = self.encoder(feat2, view2_coords, lengths)

        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        contrast_loss = self.contrast_loss_fn(z1, z2) if self.config.get('use_contrastive', True) \
            else torch.tensor(0.0, device=self.device)

        labeled_mask = labels >= 0
        if self.config.get('use_proto', True) and labeled_mask.sum() > 0:
            proto_loss = self.proto_loss_fn(z1[labeled_mask], labels[labeled_mask], self.prototypes)
        else:
            proto_loss = torch.tensor(0.0, device=self.device)

        total_loss = contrast_loss + self.proto_weight * proto_loss

        return {
            'total_loss_tensor': total_loss,
            'total_loss': total_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'proto_loss': proto_loss.item()
        }

    # training/trainer.py
    def _augment_batch(self, coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        兼容 C>=2：前2维视为位置，后续维度当作速度/其他通道。
        仅对位置做旋转/平移；速度通道只旋转/缩放，不平移。
        """
        augmented = coords.clone()
        B, L, C = coords.shape
        mask_ratio = self.config.get('mask_ratio', 0.15)
        jitter_std = self.config.get('coord_jitter_std', 1e-4)
        trans_std = self.config.get('coord_trans_std', 5e-5)

        for i in range(B):
            Li = int(lengths[i].item())
            if Li <= 1:
                continue
            x = augmented[i, :Li]  # [Li, C]
            pos = x[:, :2]
            extra = x[:, 2:] if C > 2 else None

            # small jitter（所有通道）
            if torch.rand(1).item() > 0.5:
                x = x + torch.randn_like(x) * jitter_std
                pos = x[:, :2];
                extra = x[:, 2:] if C > 2 else None

            # 邻近插值掩码（所有通道）
            if torch.rand(1).item() > 0.5:
                num_mask = max(1, int(Li * mask_ratio))
                idx = torch.randperm(Li)[:num_mask]
                rep = x.clone()
                if Li > 2:
                    prev = torch.cat([x[0:1], x[:-1]], dim=0)
                    nxt = torch.cat([x[1:], x[-1:]], dim=0)
                    rep = 0.5 * (prev + nxt)
                x[idx] = rep[idx] + torch.randn_like(x[idx]) * (jitter_std * 0.5)
                pos = x[:, :2];
                extra = x[:, 2:] if C > 2 else None

            # small rotation + scaling（pos 旋转+缩放；extra 前2维同旋转+缩放）
            if torch.rand(1).item() > 0.7:
                theta = torch.randn(1, device=x.device) * 0.15
                c, s = torch.cos(theta), torch.sin(theta)
                R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).squeeze(2)  # [2,2]
                scale = 1.0 + (torch.randn(1, device=x.device) * 0.05)
                pos = (pos @ R.T) * scale
                if extra is not None and extra.size(1) >= 2:
                    extra[:, :2] = (extra[:, :2] @ R.T) * scale

            # small global translation
            if torch.rand(1).item() > 0.7:
                shift = torch.randn(1, 2, device=x.device) * trans_std
                pos = pos + shift

            x = torch.cat([pos, extra], dim=1) if extra is not None else pos
            augmented[i, :Li] = x

        return augmented

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Light feature jitter to reduce modality mismatch.
        """
        std = self.config.get('feature_jitter_std', 0.01)  # relative scale
        if std and std > 0:
            return features + torch.randn_like(features) * std
        return features

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.encoder.eval();
        self.projector.eval()
        val_metrics = {'total_loss': 0.0, 'contrast_loss': 0.0, 'proto_loss': 0.0}

        for batch in tqdm(self.val_loader, desc='Validation'):
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].to(self.device)

            emb = self.encoder(features, coords, lengths)
            z = self.projector(emb)

            # 轻量验证：不再生成 view2，避免强增广噪音
            contrast_loss = torch.tensor(0.0, device=self.device)

            proto_loss = torch.tensor(0.0, device=self.device)
            labeled_mask = labels >= 0
            if labeled_mask.sum() > 0:
                proto_loss = self.proto_loss_fn(
                    z[labeled_mask], labels[labeled_mask], self.prototypes
                )

            total_loss = contrast_loss + self.proto_weight * proto_loss

            val_metrics['total_loss'] += total_loss.item()
            val_metrics['contrast_loss'] += contrast_loss.item()
            val_metrics['proto_loss'] += proto_loss.item()

        # Compute averages
        num_batches = len(self.val_loader)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        return val_metrics

    def fit(self) -> Dict[str, list]:
        """Main training loop."""
        max_epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 30)
        min_delta = self.config.get('min_delta', 1e-4)

        self.logger.info(f"Starting training for {max_epochs} epochs...")
        start_time = time.time()

        try:
            for epoch in range(max_epochs):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step()

                # Record history
                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['val_loss'].append(val_metrics['total_loss'])
                self.history['contrast_loss'].append(train_metrics['contrast_loss'])
                self.history['proto_loss'].append(train_metrics['proto_loss'])
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                # Log metrics
                self.logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

                # Save checkpoint
                is_best = val_metrics['total_loss'] < (self.best_val_loss - min_delta)
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    self.logger.info(f"✅ New best model saved! Val Loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                # Regular checkpoint
                if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

                # Early stopping
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")

            # Save final checkpoint
            self.save_checkpoint('final_model.pth')

            # Save training history
            self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'prototypes': self.prototypes.data,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.prototypes.data = checkpoint['prototypes'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)

        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        self.logger.info(f"Training history saved: {history_path}")


class SemiSupervisedTrainer(Trainer):
    """
    Trainer for semi-supervised learning with pseudo-labeling.

    Additional features:
    - Pseudo-label generation
    - Teacher-student consistency
    - Progressive pseudo-labeling
    """

    def __init__(self, *args,pseudo_label_generator=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional loss weights
        self.pseudo_weight = self.config.get('pseudo_weight', 0.3)
        self.consistency_weight = self.config.get('consistency_weight', 0.2)

        # Consistency loss
        self.consistency_loss_fn = ConsistencyLoss()

        # Teacher model (EMA)
        self.teacher_encoder = self._create_teacher_model()
        self.ema_decay = self.config.get('ema_decay', 0.999)

        import copy
        self.teacher_projector = copy.deepcopy(self.projector).to(self.device)
        for p in self.teacher_projector.parameters():
            p.requires_grad = False

        # Pseudo-label state
        self.pseudo_labels_dict = None
        self.pseudo_label_generator = pseudo_label_generator
        self.prototypes_initialized = False

        # Update history
        self.history.update({
            'pseudo_loss': [],
            'consistency_loss': []
        })

        # Graph settings
        self.graph_k = int(self.config.get('graph_k', 10))
        self.lambda_graph_smooth = float(self.config.get('lambda_graph_smooth', 0.0))
        self.lambda_graph_contrast = float(self.config.get('lambda_graph_contrast', 0.0))
        self.graph_build_interval = int(self.config.get('graph_build_interval', 1))

        self.graph_smooth_loss = GraphSmoothnessLoss()
        self.neighbor_contrast_loss = NeighborContrastiveLoss(
            temperature=self.config.get('temperature', 0.1)
        )

        # 全局 kNN 图缓存
        self.global_knn_indices = None  # np.ndarray [N, k]
        self.memory_z = None  # np.ndarray [N, D] (normalized)

        self.logger.info("Semi-supervised trainer initialized")

    def _create_teacher_model(self, AdaptiveTrajectoryEncoder=None) -> nn.Module:
        from models.encoders import AdaptiveTrajectoryEncoder
        teacher = AdaptiveTrajectoryEncoder(
            feat_dim=self.config.get('feat_dim', 36),
            coord_dim=self.config.get('coord_dim', 2),
            hidden_dim=self.config.get('hidden_dim', 256),
            dropout=self.config.get('dropout', 0.3),
            num_heads=self.config.get('num_attention_heads', 8),
            encoder_mode=self.config.get('encoder_mode', 'adaptive_gate')
        ).to(self.device)
        teacher.load_state_dict(self.encoder.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    # 同步 encoder + projector
    @torch.no_grad()
    def update_teacher(self):
        for tp, sp in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * sp.data
        for tp, sp in zip(self.teacher_projector.parameters(), self.projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * sp.data

    # training/trainer.py
    def _compute_loss(
            self,
            coords: torch.Tensor,
            features: torch.Tensor,
            lengths: torch.Tensor,
            labels: torch.Tensor,
            batch_indices: torch.Tensor = None  # 新增：用于图损失取 batch 邻接
    ) -> Dict[str, Any]:
        view1_coords = self._augment_batch(coords, lengths)
        view2_coords = self._augment_batch(coords, lengths)
        feat1 = self._augment_features(features)
        feat2 = self._augment_features(features)

        emb1 = self.encoder(feat1, view1_coords, lengths)
        emb2 = self.encoder(feat2, view2_coords, lengths)

        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        # 对比损失（视图）
        contrast_loss = self.contrast_loss_fn(z1, z2)

        # 原型损失（真实标签）
        proto_loss = torch.tensor(0.0, device=self.device)
        labeled_mask = labels >= 0
        if labeled_mask.sum() > 0:
            proto_loss = self.proto_loss_fn(z1[labeled_mask], labels[labeled_mask], self.prototypes)

        # 伪标签损失
        pseudo_loss = torch.tensor(0.0, device=self.device)
        if (self.pseudo_labels_dict is not None and
                'labels' in self.pseudo_labels_dict and
                'confidences' in self.pseudo_labels_dict and
                'indices' in self.pseudo_labels_dict):
            batch_idx = self.pseudo_labels_dict['indices']
            pseudo_loss = self._compute_pseudo_loss(z1, labels, batch_idx)

        # 一致性（teacher-student）
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_weight > 0:
            with torch.no_grad():
                teacher_emb = self.teacher_encoder(features, coords, lengths)
                teacher_z = self.projector(teacher_emb)
                teacher_z = F.normalize(teacher_z, dim=1)
            student_z = F.normalize(z1, dim=1)
            consistency_loss = self.consistency_loss_fn(student_z, teacher_z)

        # ===== 图损失（邻接由全局 kNN 图在 batch 裁剪或 fallback 到 batch 内 kNN）=====
        graph_smooth = torch.tensor(0.0, device=self.device)
        graph_contrast = torch.tensor(0.0, device=self.device)
        if (self.lambda_graph_smooth > 0 or self.lambda_graph_contrast > 0):
            z1_norm = F.normalize(z1, dim=1)
            adj_b = None
            if batch_indices is not None:
                adj_b = self._build_batch_adj_from_global(batch_indices, device=self.device)
            if adj_b is None or adj_b.sum() == 0:
                adj_b = self._build_batch_knn_adj(z1_norm, self.graph_k)
            if self.lambda_graph_smooth > 0:
                graph_smooth = self.graph_smooth_loss(z1_norm, adj_b)
            if self.lambda_graph_contrast > 0:
                graph_contrast = self.neighbor_contrast_loss(z1_norm, adj_b)

        total_loss = (
                contrast_loss +
                self.proto_weight * proto_loss +
                self._current_pseudo_weight() * pseudo_loss +
                self._current_consistency_weight() * consistency_loss +
                self.lambda_graph_smooth * graph_smooth +
                self.lambda_graph_contrast * graph_contrast
        )

        return {
            'total_loss_tensor': total_loss,
            'total_loss': total_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'proto_loss': proto_loss.item(),
            'pseudo_loss': pseudo_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'graph_smooth_loss': graph_smooth.item(),
            'graph_contrast_loss': graph_contrast.item()
        }

    def _current_consistency_weight(self) -> float:
        target = self.config.get('consistency_weight', 0.05)
        warmup = self.config.get('pseudo_warmup_epochs', 10)
        ramp = self.config.get('consistency_ramp_epochs', 50)
        if self.current_epoch < warmup:
            return 0.0
        t = min(1.0, (self.current_epoch - warmup + 1) / max(1, ramp))
        return target * t

    @torch.no_grad()
    def _extract_all_projected_embeddings(self) -> np.ndarray:
        """提取训练集所有样本的 z（已归一化），用于构图/传播。"""
        self.encoder.eval(); self.projector.eval()
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )
        all_z = []
        for batch in loader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            emb = self.encoder(features, coords, lengths)
            z = self.projector(emb)
            z = F.normalize(z, dim=1)
            all_z.append(z.cpu().numpy())
        self.encoder.train(); self.projector.train()
        return np.vstack(all_z)

    def _rebuild_global_knn_graph(self):
        """重建全局 kNN 图（基于投影空间 z 的余弦相似）。"""
        import numpy as np

        self.memory_z = self._extract_all_projected_embeddings()     # [N,D], normalized
        N = self.memory_z.shape[0]
        # 余弦相似 (Z Z^T)
        sims = self.memory_z @ self.memory_z.T                       # [N,N]
        np.fill_diagonal(sims, -1.0)                                 # 排除自身
        k = min(self.graph_k, max(1, N - 1))
        # top-k neighbors
        knn_idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]     # 未排序 top-k
        # 为稳定起见，可再按相似度排序
        row_arange = np.arange(N)[:, None]
        knn_sims = sims[row_arange, knn_idx]
        order = np.argsort(-knn_sims, axis=1)
        self.global_knn_indices = knn_idx[row_arange, order]         # [N,k]

    def _build_batch_adj_from_global(self, batch_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        基于全局 kNN 图构建 batch 内邻接矩阵。
        - batch_indices：数据集中样本的全局索引（collate_fn 已返回），shape [B]
        返回：adj_b [B,B] （0/1 对称）
        """
        if self.global_knn_indices is None:
            return None
        idx_np = batch_indices.detach().cpu().numpy()                # [B]
        index_pos = {int(gidx): i for i, gidx in enumerate(idx_np)}  # 全局 idx -> batch 内位置
        B = len(idx_np)
        adj = torch.zeros(B, B, device=device)

        for i, g in enumerate(idx_np):
            neigh = self.global_knn_indices[g]                       # [k]
            # 仅保留邻居也在本 batch 的
            for n in neigh:
                j = index_pos.get(int(n), None)
                if j is not None and j != i:
                    adj[i, j] = 1.0

        # 对称化
        adj = torch.maximum(adj, adj.t())
        return adj

    def _build_batch_knn_adj(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """
        仅用当前 batch 的 z 构建 kNN 邻接（余弦相似）。
        返回 adj [B,B]（0/1，对称）
        """
        B = z.size(0)
        sim = torch.matmul(z, z.t())             # [B,B], z 已归一化
        sim.fill_diagonal_(0.0)
        k = min(k, max(1, B - 1))
        topk = sim.topk(k, dim=1).indices        # [B,k]
        adj = torch.zeros(B, B, device=z.device)
        adj.scatter_(1, topk, torch.ones_like(topk, dtype=adj.dtype))
        adj = torch.maximum(adj, adj.t())
        return adj


    # training/trainer.py (class SemiSupervisedTrainer)
    import torch.nn.functional as F

    def _compute_pseudo_loss(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            batch_indices: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.detach().cpu().numpy()

        batch_pseudo_labels = self.pseudo_labels_dict['labels'][batch_indices]
        batch_pseudo_confs = self.pseudo_labels_dict['confidences'][batch_indices]

        pseudo_mask = (batch_pseudo_labels >= 0) & (labels.cpu().numpy() < 0)
        if pseudo_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        pseudo_labels_tensor = torch.LongTensor(batch_pseudo_labels[pseudo_mask]).to(self.device)
        pseudo_conf_tensor = torch.FloatTensor(batch_pseudo_confs[pseudo_mask]).to(self.device)
        pseudo_z = embeddings[pseudo_mask]

        # 逐样本 logits
        pseudo_z = F.normalize(pseudo_z, dim=1)
        protos = F.normalize(self.prototypes, dim=1)
        logits = torch.matmul(pseudo_z, protos.t()) / self.config.get('temperature', 0.07)

        per_sample_loss = F.cross_entropy(logits, pseudo_labels_tensor, reduction='none')
        loss = (per_sample_loss * pseudo_conf_tensor).mean()
        return loss

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (semi-supervised, with extra metrics)."""
        self.encoder.train()
        self.projector.train()

        epoch_metrics = {
            'total_loss': 0.0,
            'contrast_loss': 0.0,
            'proto_loss': 0.0,
            'pseudo_loss': 0.0,
            'consistency_loss': 0.0,
            'graph_smooth_loss': 0.0,
            'graph_contrast_loss': 0.0
        }

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config.get("epochs", 100)}'
        )

        for batch_idx, batch in enumerate(pbar):
            loss_dict = self.train_step(batch)

            # Only sum keys we track
            for k in epoch_metrics.keys():
                if k in loss_dict:
                    epoch_metrics[k] += loss_dict[k]

            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'ctr': f"{loss_dict.get('contrast_loss', 0.0):.4f}",
                'proto': f"{loss_dict.get('proto_loss', 0.0):.4f}",
                'pseudo': f"{loss_dict.get('pseudo_loss', 0.0):.4f}",
                'cons': f"{loss_dict.get('consistency_loss', 0.0):.4f}",
            })
            self.global_step += 1

        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # 仅当 pseudo_labels_dict 完整时更新批索引
        if 'indices' in batch and self.pseudo_labels_dict is not None:
            if all(k in self.pseudo_labels_dict for k in ['labels', 'confidences']):
                self.pseudo_labels_dict['indices'] = batch['indices']

        coords = batch['coords'].to(self.device)
        features = batch['features'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        labels = batch['labels'].to(self.device)
        batch_indices = batch['indices'] if 'indices' in batch else None

        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self._compute_loss(coords, features, lengths, labels, batch_indices)
        else:
            loss_dict = self._compute_loss(coords, features, lengths, labels, batch_indices)

        # 反传与优化
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss_dict['total_loss_tensor']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.projector.parameters()) +
                [self.prototypes],
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict['total_loss_tensor'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.projector.parameters()) +
                [self.prototypes],
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.optimizer.step()

        # EMA teacher 更新
        self.update_teacher()

        out = {
            'total_loss': loss_dict['total_loss'],
            'contrast_loss': loss_dict.get('contrast_loss', 0.0),
            'proto_loss': loss_dict.get('proto_loss', 0.0),
            'pseudo_loss': loss_dict.get('pseudo_loss', 0.0),
            'consistency_loss': loss_dict.get('consistency_loss', 0.0),
            'graph_smooth_loss': loss_dict.get('graph_smooth_loss', 0.0),
            'graph_contrast_loss': loss_dict.get('graph_contrast_loss', 0.0)
        }
        return out

    from data.dataset import traj_collate_fn


    @torch.no_grad()
    def _bootstrap_prototypes_from_labeled(self):
        """用当前有标签样本的投影均值初始化原型。"""
        self.encoder.eval();
        self.projector.eval()
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )
        C = self.num_classes
        D = self.prototypes.shape[1]
        sum_z = torch.zeros(C, D, device=self.device)
        cnt = torch.zeros(C, device=self.device)

        for batch in loader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].to(self.device)

            labeled_mask = labels >= 0
            if labeled_mask.sum() == 0:
                continue
            emb = self.encoder(features, coords, lengths)
            z = self.projector(emb)
            z = F.normalize(z, dim=1)

            for c in range(C):
                m = (labels == c) & labeled_mask
                if m.any():
                    sum_z[c] += z[m].sum(dim=0)
                    cnt[c] += m.sum()

        mask = cnt > 0
        if mask.any():
            self.prototypes.data[mask] = F.normalize(sum_z[mask] / cnt[mask].unsqueeze(1), dim=1)
        self.encoder.train();
        self.projector.train()

    def fit(self) -> Dict[str, list]:
        """
        半监督训练主循环：
        - 按间隔刷新伪标签
        - 记录 pseudo/consistency 指标
        - 早停与保存 checkpoint
        """
        import time
        max_epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('pseudo_warmup_epochs', 10)
        pseudo_interval = self.config.get('pseudo_label_interval', 10)
        if max_epochs <= pseudo_interval:  # 少轮数/调试兜底
            pseudo_interval = 1
        patience = self.config.get('patience', 30)
        min_delta = self.config.get('min_delta', 1e-4)
        # 确保历史项存在
        if 'pseudo_loss' not in self.history:
            self.history['pseudo_loss'] = []
        if 'consistency_loss' not in self.history:
            self.history['consistency_loss'] = []

        self.logger.info(
            f"Starting semi-supervised training for {max_epochs} epochs "
            f"(pseudo_label_interval={pseudo_interval})..."
        )
        start_time = time.time()

        try:
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                # ===== Graph build (每 graph_build_interval 个 epoch) =====
                if (self.lambda_graph_smooth > 0 or self.lambda_graph_contrast > 0):
                    if epoch % max(1, self.graph_build_interval) == 0:
                        try:
                            self.logger.info(f"Rebuilding global kNN graph (k={self.graph_k}) ...")
                            self._rebuild_global_knn_graph()
                            self.logger.info("Global kNN graph built.")
                        except Exception as e:
                            self.logger.warning(f"Global graph build failed: {e}")
                            self.global_knn_indices = None

                # 在第一个epoch用有标签数据初始化原型
                if epoch == 0:
                    self._initialize_prototypes_from_labeled()

                # 在warmup后开始更新伪标签
                if epoch >= warmup_epochs and epoch % self.config.get('pseudo_label_interval', 5) == 0:
                    if self.pseudo_label_generator is not None:
                        self.update_pseudo_labels(self.pseudo_label_generator)

                # 训练和验证...
                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                # 学习率调度
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step()

                # 记录历史
                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['val_loss'].append(val_metrics['total_loss'])
                self.history['contrast_loss'].append(train_metrics.get('contrast_loss', 0.0))
                self.history['proto_loss'].append(train_metrics.get('proto_loss', 0.0))
                self.history['pseudo_loss'].append(train_metrics.get('pseudo_loss', 0.0))
                self.history['consistency_loss'].append(train_metrics.get('consistency_loss', 0.0))
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                # 日志
                self.logger.info(
                    "Epoch {}/{} - Train: {:.4f} [ctr {:.4f} | proto {:.4f} | pseudo {:.4f} | cons {:.4f}] "
                    "| Val: {:.4f} | LR: {:.6f}".format(
                        epoch + 1, max_epochs,
                        train_metrics['total_loss'],
                        train_metrics.get('contrast_loss', 0.0),
                        train_metrics.get('proto_loss', 0.0),
                        train_metrics.get('pseudo_loss', 0.0),
                        train_metrics.get('consistency_loss', 0.0),
                        val_metrics['total_loss'],
                        self.optimizer.param_groups[0]['lr']
                    )
                )

                # 保存最佳
                is_best = val_metrics['total_loss'] < (self.best_val_loss - min_delta)
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    self.logger.info(f"✅ New best model saved! Val Loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                # 周期性保存
                if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

                # 早停
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break


        except KeyboardInterrupt:

            self.logger.info("Training interrupted")

        finally:

            elapsed_time = time.time() - start_time

            self.logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")

            self.save_checkpoint('final_model.pth')

            self.save_history()

        return self.history

    @torch.no_grad()
    def _initialize_prototypes_from_labeled(self):
        """用有标签数据初始化原型"""
        self.logger.info("🔧 Initializing prototypes from labeled samples...")

        self.encoder.eval()
        self.projector.eval()

        # 收集有标签样本的embeddings
        from data.dataset import traj_collate_fn
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        class_embeddings = {i: [] for i in range(self.num_classes)}

        for batch in loader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels']

            # 只处理有标签样本
            labeled_mask = labels >= 0
            if labeled_mask.sum() == 0:
                continue

            # 提取embeddings
            emb = self.encoder(
                features[labeled_mask],
                coords[labeled_mask],
                lengths[labeled_mask]
            )
            z = self.projector(emb)
            z = F.normalize(z, dim=1)

            # 按类别分组
            for i, label in enumerate(labels[labeled_mask].numpy()):
                class_embeddings[label].append(z[i].cpu())

        # 计算每个类的原型（均值）
        for class_id in range(self.num_classes):
            if len(class_embeddings[class_id]) > 0:
                class_z = torch.stack(class_embeddings[class_id])
                proto = class_z.mean(dim=0)
                proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
                self.prototypes.data[class_id] = proto.to(self.device)
                self.logger.info(
                    f"   Class {class_id} ({self.config.get('label_names', {}).get(class_id, class_id)}): "
                    f"{len(class_embeddings[class_id])} samples"
                )
            else:
                self.logger.warning(f"   Class {class_id}: No labeled samples!")

        self.prototypes_initialized = True
        self.encoder.train()
        self.projector.train()

    def update_pseudo_labels(self, pseudo_label_generator):
        self.logger.info("Updating pseudo-labels...")

        self.encoder.eval();
        self.projector.eval()
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        all_z, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Extracting z (aligned)'):
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels']

                emb = self.encoder(features, coords, lengths)
                z = self.projector(emb)
                z = F.normalize(z, dim=1)

                all_z.append(z.cpu().numpy())
                all_labels.append(labels.numpy())

        all_z = np.vstack(all_z)
        all_labels = np.hstack(all_labels)

        # 新接口：直接传 z
        new_labels, confidences = pseudo_label_generator.generate_pseudo_labels(
            projected_embeddings=all_z,
            labels=all_labels,
            prototypes=self.prototypes.data,
            epoch=self.current_epoch
        )
        # ====== 图式伪标签传播（LP） + 融合 ======
        try:
            glp_labels, glp_conf = graph_label_propagation(
                projected_embeddings=all_z,   # np.ndarray (N,D)，是你上面 already normalized 的 z_t.cpu().numpy() 也可
                labels=all_labels,
                k=self.graph_k,
                alpha=0.9,
                iters=20
            )
            # 融合（阈值可用训练配置或生成器当前阈值）
            thr = float(self.config.get('pseudo_label_threshold', 0.8))
            fused_labels, fused_conf = fuse_pseudo_labels(
                base_labels=new_labels,
                base_conf=confidences,
                glp_labels=glp_labels,
                glp_conf=glp_conf,
                observed_labels=all_labels,
                thr=max(0.5, min(0.95, thr))   # 稍作裁剪
            )
            # 覆盖为融合结果
            new_labels = fused_labels
            confidences = fused_conf
            self.logger.info("Pseudo labels fused with Graph LP.")
        except Exception as e:
            self.logger.warning(f"Graph LP failed, use base pseudo labels only: {e}")

        # 统计采纳
        unlabeled_mask = all_labels < 0
        adopted_mask = (new_labels >= 0) & unlabeled_mask
        adopted = int(adopted_mask.sum())
        self.logger.info(f"✅ 伪标签更新: 采纳 {adopted}/{int(unlabeled_mask.sum())} 个 (unlabeled)")

        # 保存供 batch 内部索引用（不写回 dataset）
        self.pseudo_labels_dict = {
            'labels': new_labels,
            'confidences': confidences,
            'indices': None
        }

        # ====== 原型 EMA 更新 (labeled + 高置信伪标签) ======
        try:
            ema = float(self.config.get('proto_ema', 0.9))
            conf_thr = float(self.config.get('proto_ema_conf_thr', 0.9))

            z_t = torch.from_numpy(all_z).to(self.device)
            C, D = self.num_classes, self.prototypes.shape[1]
            sum_z = torch.zeros(C, D, device=self.device)
            cnt = torch.zeros(C, device=self.device)

            # 真实有标签
            labeled_mask = all_labels >= 0
            if labeled_mask.any():
                y_l = torch.from_numpy(all_labels[labeled_mask]).long().to(self.device)
                z_l = z_t[labeled_mask]
                for c in range(C):
                    m = (y_l == c)
                    if m.any():
                        sum_z[c] += z_l[m].sum(dim=0)
                        cnt[c] += m.sum()

            # 高置信伪标签
            pseudo_high_mask = (unlabeled_mask) & (new_labels >= 0) & (confidences >= conf_thr)
            if pseudo_high_mask.any():
                y_p = torch.from_numpy(new_labels[pseudo_high_mask]).long().to(self.device)
                z_p = z_t[pseudo_high_mask]
                for c in range(C):
                    m = (y_p == c)
                    if m.any():
                        sum_z[c] += z_p[m].sum(dim=0)
                        cnt[c] += m.sum()

            mask = cnt > 0
            if mask.any():
                new_means = F.normalize(sum_z[mask] / cnt[mask].unsqueeze(1), dim=1)
                old = self.prototypes.data[mask]
                self.prototypes.data[mask] = F.normalize(ema * old + (1.0 - ema) * new_means, dim=1)
                self.logger.info(f"Prototypes EMA updated (classes: {(mask.cpu().numpy()).sum()})")
        except Exception as e:
            self.logger.warning(f"Prototype EMA update skipped: {e}")

        self.encoder.train();
        self.projector.train()

        # # 兜底：一次都采不到时，临时降低阈值重试
        # if pseudo_count == 0 and hasattr(pseudo_label_generator, 'confidence_threshold'):
        #     old_thr = float(pseudo_label_generator.confidence_threshold)
        #     new_thr = max(0.5, old_thr - 0.1)
        #     if new_thr < old_thr:
        #         pseudo_label_generator.confidence_threshold = new_thr
        #         self.logger.warning(f"伪标签=0，临时降阈值 {old_thr:.2f}->{new_thr:.2f} 重试一次")
        #         new_labels, confidences = pseudo_label_generator.generate_pseudo_labels(
        #             all_embeddings, all_labels, self.prototypes.data,
        #             projector=self.projector, epoch=self.current_epoch
        #         )
        #         protected_labels = all_labels.copy()
        #         protected_labels[unlabeled_mask] = new_labels[unlabeled_mask]
        #         self.train_loader.dataset.update_labels(protected_labels)
        #         self.pseudo_labels_dict = {'labels': protected_labels, 'confidences': confidences, 'indices': None}
        #         pseudo_count = (protected_labels >= 0).sum() - (all_labels >= 0).sum()
        #         self.logger.info(f"二次尝试: +{pseudo_count} 个新伪标签")

    def _current_pseudo_weight(self) -> float:
        target = self.config.get('pseudo_weight', 0.3)
        warmup = self.config.get('pseudo_warmup_epochs', 10)
        ramp = self.config.get('pseudo_ramp_epochs', 50)
        if self.current_epoch < warmup:
            return 0.0
        t = min(1.0, (self.current_epoch - warmup + 1) / max(1, ramp))
        return target * t


