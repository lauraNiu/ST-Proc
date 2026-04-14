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
from typing import Dict, Optional, Tuple, Any, List
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import logging
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from data.dataset import traj_collate_fn
from utils.logger import get_logger
from .scheduler import WarmupCosineScheduler
from .loss import (
    ContrastiveLoss,
    PrototypicalLoss,
    ConsistencyLoss,
    GraphSmoothnessLoss,
    NeighborContrastiveLoss,
    compute_prototype_logits,
)
from training.pseudo_label import (
    PSEUDO_SOURCE_ABSTAIN,
    PSEUDO_SOURCE_AGREE,
    PSEUDO_SOURCE_BASE_CONFLICT,
    PSEUDO_SOURCE_BASE_ONLY,
    PSEUDO_SOURCE_LP_CONFLICT,
    PSEUDO_SOURCE_LP_ONLY,
    PSEUDO_SOURCE_OBSERVED,
    apply_pseudo_label_acceptance_cap,
    class_balanced_pseudo_cap,
    fuse_pseudo_labels,
    graph_label_propagation,
)


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
        self.base_prototypes_per_class = max(1, int(config.get('prototypes_per_class', 1)))
        proto_map_cfg = config.get('prototype_per_class_map', {}) or {}
        self.prototype_per_class_map = {int(k): max(1, int(v)) for k, v in proto_map_cfg.items()}
        self.class_prototype_counts = [
            max(1, int(self.prototype_per_class_map.get(cls, self.base_prototypes_per_class)))
            for cls in range(self.num_classes)
        ]
        self.prototypes_per_class = max(self.class_prototype_counts)
        projection_dim = config.get('projection_dim', 64)

        prototype_shape = (self.num_classes, projection_dim)
        if self.prototypes_per_class > 1:
            prototype_shape = (self.num_classes, self.prototypes_per_class, projection_dim)

        prototypes_init = torch.randn(*prototype_shape)
        nn.init.xavier_uniform_(prototypes_init.view(-1, projection_dim))
        self.prototypes = prototypes_init.to(device)

        # Loss functions
        self.contrast_loss_fn = ContrastiveLoss(
            temperature=config.get('temperature', 0.07)
        )
        self.prototype_pooling = str(config.get('prototype_pooling', 'max')).lower()
        self.prototype_pool_temperature = float(config.get('prototype_pool_temperature', 1.0))
        self.proto_loss_fn = PrototypicalLoss(
            temperature=config.get('temperature', 0.07),
            margin=float(config.get('proto_margin', 0.0)),
            class_weights=config.get('class_weights', None),
            prototype_pooling=self.prototype_pooling,
            prototype_pool_temperature=self.prototype_pool_temperature,
            device=str(self.device)
        )

        # Optimization config
        self.classifier = getattr(self, 'classifier', None)
        self.classifier_weight = float(config.get('classifier_weight', 1.0))
        self.selection_metric = config.get('selection_metric', 'macro_f1')
        self.selection_mode = 'min' if self.selection_metric in {'total_loss', 'val_loss', 'loss'} else 'max'

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
        self.best_val_score = float('inf') if self.selection_mode == 'min' else float('-inf')
        self.patience_counter = 0

        # Loss weights
        self.contrast_weight = float(config.get('contrast_weight', 1.0))
        self.proto_weight = float(config.get('proto_weight', 0.5))
        self.classifier_weight_final = float(config.get('classifier_weight_final', self.classifier_weight))
        self.contrast_weight_final = float(config.get('contrast_weight_final', self.contrast_weight))
        self.proto_weight_final = float(config.get('proto_weight_final', self.proto_weight))
        self.graph_weight_final = float(config.get('graph_weight_final', 1.0))
        self.use_stagewise_loss_schedule = bool(config.get('use_stagewise_loss_schedule', False))
        self.classification_stage_start = int(config.get('classification_stage_start', 0))
        self.classification_stage_ramp = max(1, int(config.get('classification_stage_ramp', 1)))
        cw = config.get('class_weights', None)
        self.class_weight_tensor = None
        if cw is not None:
            self.class_weight_tensor = torch.tensor(cw, dtype=torch.float32, device=self.device)

        hard_pairs_cfg = config.get('hard_negative_pairs', []) or []
        self.hard_negative_pairs = []
        for pair in hard_pairs_cfg:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                self.hard_negative_pairs.append((int(pair[0]), int(pair[1])))
        self.hard_negative_margin = float(config.get('hard_negative_margin', 0.25))
        self.hard_negative_weight = float(config.get('hard_negative_weight', 0.0))
        coarse_groups_cfg = config.get('coarse_groups', [[0, 1], [2, 3], [4]]) or []
        self.coarse_groups = [tuple(int(v) for v in group) for group in coarse_groups_cfg if group]
        self.coarse_aux_weight = float(config.get('coarse_aux_weight', 0.0))
        self.fine_to_coarse = {}
        for group_idx, group in enumerate(self.coarse_groups):
            for cls in group:
                self.fine_to_coarse[int(cls)] = group_idx
        hard_classes = set()
        for pair in self.hard_negative_pairs:
            hard_classes.update(pair)
        hard_classes.update(int(k) for k in (config.get('sampler_hard_class_boost', {}) or {}).keys())
        hard_classes.update(int(k) for k in (config.get('pseudo_hard_class_extra_strictness', {}) or {}).keys())
        self.hard_class_ids = hard_classes

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'contrast_loss': [],
            'proto_loss': [],
            'ce_loss': [],
            'hard_negative_loss': [],
            'coarse_loss': [],
            'lr': [],
            'metrics': [],
            'pseudo_monitor': []
        }

        self.logger.info(f"Trainer initialized with config: {config}")
        self.logger.info(f"Model parameters: {self._count_parameters():,}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        params = self._trainable_parameters()

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

    def _trainable_parameters(self):
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        if getattr(self, 'classifier', None) is not None:
            params += list(self.classifier.parameters())
        if getattr(self, 'graph_agg', None) is not None:
            params += list(self.graph_agg.parameters())
        return params

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self._trainable_parameters() if p.requires_grad)

    def _get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        return float(val_metrics.get(self.selection_metric, val_metrics.get('total_loss', 0.0)))

    def _is_improved(self, score: float, min_delta: float) -> bool:
        if self.selection_mode == 'min':
            return score < (self.best_val_score - min_delta)
        return score > (self.best_val_score + min_delta)

    def _compute_eval_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        if len(labels) == 0:
            return {'accuracy': 0.0, 'macro_f1': 0.0, 'balanced_accuracy': 0.0}
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, preds))
        }

    def _stage_interp(self, start: float, final: float) -> float:
        if not self.use_stagewise_loss_schedule:
            return float(start)
        if self.current_epoch < self.classification_stage_start:
            return float(start)
        progress = min(
            1.0,
            (self.current_epoch - self.classification_stage_start + 1) / max(1, self.classification_stage_ramp)
        )
        return float(start) + (float(final) - float(start)) * progress

    def _current_classifier_weight(self) -> float:
        return self._stage_interp(self.classifier_weight, self.classifier_weight_final)

    def _current_contrast_weight(self) -> float:
        return self._stage_interp(self.contrast_weight, self.contrast_weight_final)

    def _current_proto_weight(self) -> float:
        return self._stage_interp(self.proto_weight, self.proto_weight_final)

    def _current_graph_weight(self, base_weight: float) -> float:
        return self._stage_interp(base_weight, base_weight * self.graph_weight_final)

    def _mean_pseudo_quality(self) -> Optional[float]:
        if not self.pseudo_class_precision_ema:
            return None
        hard_vals = [
            float(self.pseudo_class_precision_ema[c])
            for c in sorted(self.hard_class_ids)
            if c in self.pseudo_class_precision_ema
        ]
        if hard_vals:
            return float(np.mean(hard_vals))
        vals = [float(v) for v in self.pseudo_class_precision_ema.values()]
        return float(np.mean(vals)) if vals else None

    def _current_pseudo_cap_rate(self) -> float:
        base_rate = float(self.config.get('pseudo_max_adoption_rate', 1.0))
        target_rate = float(self.config.get('pseudo_target_adoption_rate', base_rate))
        ratio = float(self.config.get('labeled_ratio', 1.0))
        low_cutoff = float(self.config.get('low_ratio_cutoff', 0.10))
        if ratio > low_cutoff:
            return target_rate
        ramp_epochs = max(1, int(self.config.get('pseudo_cap_ramp_epochs', 1)))
        warmup = int(self.config.get('pseudo_warmup_epochs', 0))
        quality = self._mean_pseudo_quality()
        q_min = float(self.config.get('pseudo_cap_ramp_min_quality', 0.88))
        q_max = float(self.config.get('pseudo_cap_ramp_max_quality', 0.94))
        quality_scale = 0.0 if quality is None else np.clip((quality - q_min) / max(1e-6, q_max - q_min), 0.0, 1.0)
        time_scale = np.clip((self.current_epoch - warmup + 1) / ramp_epochs, 0.0, 1.0)
        scale = min(float(time_scale), float(quality_scale))
        return float(base_rate + (target_rate - base_rate) * scale)

    def _current_hard_negative_weight(self) -> float:
        base_weight = float(self.hard_negative_weight)
        ratio = float(self.config.get('labeled_ratio', 1.0))
        low_cutoff = float(self.config.get('low_ratio_cutoff', 0.10))
        if ratio > low_cutoff:
            return base_weight
        start_scale = float(self.config.get('hard_negative_weight_scale_low_ratio', 1.0))
        ramp_epochs = max(1, int(self.config.get('low_ratio_aux_ramp_epochs', 35)))
        progress = np.clip((self.current_epoch + 1) / ramp_epochs, 0.0, 1.0)
        return float(base_weight * (start_scale + (1.0 - start_scale) * progress))

    def _current_coarse_aux_weight(self) -> float:
        base_weight = float(self.coarse_aux_weight)
        ratio = float(self.config.get('labeled_ratio', 1.0))
        low_cutoff = float(self.config.get('low_ratio_cutoff', 0.10))
        if ratio > low_cutoff:
            return base_weight
        start_scale = float(self.config.get('coarse_aux_weight_scale_low_ratio', 1.0))
        ramp_epochs = max(1, int(self.config.get('low_ratio_aux_ramp_epochs', 35)))
        progress = np.clip((self.current_epoch + 1) / ramp_epochs, 0.0, 1.0)
        return float(base_weight * (start_scale + (1.0 - start_scale) * progress))

    def _get_class_sample_weights(self, labels: torch.Tensor) -> torch.Tensor:
        if self.class_weight_tensor is None or labels.numel() == 0:
            return torch.ones_like(labels, dtype=torch.float32, device=self.device)
        w = self.class_weight_tensor.to(self.device)
        labels = labels.clamp_min(0)
        labels = labels.clamp_max(w.numel() - 1)
        sample_w = w[labels]
        return sample_w / sample_w.mean().clamp_min(1e-6)

    def _prototype_count_for_class(self, class_id: int) -> int:
        class_id = int(class_id)
        if not (0 <= class_id < len(self.class_prototype_counts)):
            return int(self.base_prototypes_per_class)
        target_count = int(self.class_prototype_counts[class_id])
        ratio = float(self.config.get('labeled_ratio', 1.0))
        low_cutoff = float(self.config.get('prototype_stage_low_ratio_cutoff', 0.10))
        if ratio > low_cutoff:
            return target_count
        low_map_cfg = self.config.get('prototype_per_class_map_low_ratio', {}) or {}
        low_map = {int(k): max(1, int(v)) for k, v in low_map_cfg.items()}
        start_count = min(target_count, int(low_map.get(class_id, target_count)))
        if start_count >= target_count:
            return target_count
        expand_epoch = int(self.config.get('prototype_expand_epoch', 0))
        quality_thr = float(self.config.get('prototype_expand_quality_thr', 0.92))
        quality = self.pseudo_class_precision_ema.get(class_id)
        if self.current_epoch < expand_epoch:
            return start_count
        if quality is None or quality < quality_thr:
            return start_count
        return target_count

    def _compute_hard_negative_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.hard_negative_weight <= 0.0 or logits.numel() == 0 or not self.hard_negative_pairs:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for cls_a, cls_b in self.hard_negative_pairs:
            pair_mask = (labels == cls_a) | (labels == cls_b)
            if pair_mask.sum() == 0:
                continue
            pair_logits = logits[pair_mask]
            pair_labels = labels[pair_mask]
            sample_w = self._get_class_sample_weights(pair_labels)
            row_idx = torch.arange(pair_labels.size(0), device=pair_labels.device)
            true_logits = pair_logits[row_idx, pair_labels]
            other_labels = torch.where(
                pair_labels == cls_a,
                torch.full_like(pair_labels, cls_b),
                torch.full_like(pair_labels, cls_a),
            )
            other_logits = pair_logits[row_idx, other_labels]
            margin_gap = true_logits - other_logits
            losses.append(F.relu(self.hard_negative_margin - margin_gap) * sample_w)

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(losses).mean()

    def _compute_coarse_aux_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.coarse_aux_weight <= 0.0 or logits.numel() == 0 or not self.coarse_groups:
            return torch.tensor(0.0, device=self.device)
        coarse_logits = []
        for group in self.coarse_groups:
            idx = torch.tensor(group, device=logits.device, dtype=torch.long)
            coarse_logits.append(torch.logsumexp(logits[:, idx], dim=1))
        coarse_logits = torch.stack(coarse_logits, dim=1)
        coarse_targets = torch.tensor([self.fine_to_coarse[int(v)] for v in labels.detach().cpu().tolist()], device=labels.device)
        return F.cross_entropy(coarse_logits, coarse_targets)

    def _update_pseudo_quality_ema(self, per_class_monitor: Dict[int, Dict[str, float]]):
        momentum = float(self.config.get('pseudo_quality_momentum', 0.7))
        for cls, stats in per_class_monitor.items():
            acc = stats.get('hidden_acc')
            if acc is None:
                continue
            cls = int(cls)
            prev = self.pseudo_class_precision_ema.get(cls)
            if prev is None:
                self.pseudo_class_precision_ema[cls] = float(acc)
            else:
                self.pseudo_class_precision_ema[cls] = float(momentum * prev + (1.0 - momentum) * acc)

    def _build_pseudo_monitor(self, observed_labels, pseudo_labels, confidences, pseudo_sources) -> Dict[int, Dict[str, float]]:
        label_names = {0: 'walk', 1: 'bike', 2: 'bus', 3: 'car', 4: 'subway'}
        unlabeled_mask = observed_labels < 0
        adopted_mask = unlabeled_mask & (pseudo_labels >= 0)
        hidden_mask = None
        hidden_true = None
        if self.hidden_true_labels is not None:
            hidden_mask = unlabeled_mask & (self.hidden_true_labels >= 0)
            hidden_true = self.hidden_true_labels

        monitor = {}
        accepted_hidden_mask = None
        true_hidden_accepted = None
        pred_hidden_accepted = None
        if hidden_mask is not None:
            accepted_hidden_mask = hidden_mask & (pseudo_labels >= 0)
            true_hidden_accepted = hidden_true[accepted_hidden_mask]
            pred_hidden_accepted = pseudo_labels[accepted_hidden_mask]

        for cls in range(self.num_classes):
            cls_adopted_mask = adopted_mask & (pseudo_labels == cls)
            cls_stats = {
                'name': label_names.get(cls, str(cls)),
                'adopted': int(cls_adopted_mask.sum()),
                'mean_conf': float(confidences[cls_adopted_mask].mean()) if cls_adopted_mask.any() else None,
                'source_agree': int(((pseudo_sources == PSEUDO_SOURCE_AGREE) & cls_adopted_mask).sum()),
                'source_base': int(((pseudo_sources == PSEUDO_SOURCE_BASE_ONLY) & cls_adopted_mask).sum()),
                'source_lp': int(((pseudo_sources == PSEUDO_SOURCE_LP_ONLY) & cls_adopted_mask).sum()),
                'source_base_conflict': int(((pseudo_sources == PSEUDO_SOURCE_BASE_CONFLICT) & cls_adopted_mask).sum()),
                'source_lp_conflict': int(((pseudo_sources == PSEUDO_SOURCE_LP_CONFLICT) & cls_adopted_mask).sum()),
                'hidden_acc': None,
                'hidden_f1': None,
                'hidden_support': 0,
            }
            if accepted_hidden_mask is not None and accepted_hidden_mask.any():
                cls_pred_mask = pred_hidden_accepted == cls
                cls_true_binary = (true_hidden_accepted == cls).astype(np.int32)
                cls_pred_binary = (pred_hidden_accepted == cls).astype(np.int32)
                if cls_true_binary.sum() > 0 or cls_pred_binary.sum() > 0:
                    cls_stats['hidden_f1'] = float(f1_score(cls_true_binary, cls_pred_binary, zero_division=0))
                if cls_pred_mask.any():
                    cls_stats['hidden_acc'] = float((true_hidden_accepted[cls_pred_mask] == cls).mean())
                    cls_stats['hidden_support'] = int(cls_pred_mask.sum())
            monitor[cls] = cls_stats
        return monitor

    def _current_allow_lp_only(self) -> bool:
        if bool(self.config.get('pseudo_allow_lp_only', False)):
            return True
        lp_warmup = int(self.config.get('pseudo_lp_only_warmup_epochs', 0))
        return self.current_epoch >= lp_warmup

    def _apply_lp_source_cap(
            self,
            labels: np.ndarray,
            confidences: np.ndarray,
            observed_labels: np.ndarray,
            pseudo_sources: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_lp_rate = float(self.config.get('pseudo_lp_max_adoption_rate', 0.0))
        if max_lp_rate <= 0.0:
            return labels, confidences, pseudo_sources
        unlabeled_mask = observed_labels < 0
        lp_mask = ((pseudo_sources == PSEUDO_SOURCE_LP_ONLY) | (pseudo_sources == PSEUDO_SOURCE_LP_CONFLICT))
        lp_idx = np.where(lp_mask & unlabeled_mask & (labels >= 0))[0]
        if len(lp_idx) == 0:
            return labels, confidences, pseudo_sources
        cap = max(1, int(unlabeled_mask.sum() * max_lp_rate))
        if len(lp_idx) <= cap:
            return labels, confidences, pseudo_sources
        order = lp_idx[np.argsort(-confidences[lp_idx])]
        drop_idx = order[cap:]
        labels = labels.copy()
        confidences = confidences.copy()
        pseudo_sources = pseudo_sources.copy()
        labels[drop_idx] = -1
        confidences[drop_idx] = 0.0
        pseudo_sources[drop_idx] = PSEUDO_SOURCE_ABSTAIN
        return labels, confidences, pseudo_sources

    def _update_generator_class_thresholds(self, pseudo_label_generator, observed_labels: np.ndarray):
        if not bool(self.config.get('pseudo_adaptive_per_class', False)):
            return
        if not hasattr(pseudo_label_generator, 'per_class_thr'):
            return
        labeled = observed_labels[observed_labels >= 0]
        if labeled.size == 0:
            return
        counts = np.bincount(labeled.astype(np.int64), minlength=self.num_classes)
        positive = counts[counts > 0]
        if positive.size == 0:
            return

        max_count = max(float(positive.max()), 1.0)
        rare_gamma = float(self.config.get('pseudo_rare_class_gamma', 0.0))
        margin_gamma = float(self.config.get('pseudo_margin_relax_gamma', 0.0))
        hard_extra = {int(k): float(v) for k, v in (self.config.get('pseudo_hard_class_extra_strictness', {}) or {}).items()}
        quality_relax_thr = float(self.config.get('pseudo_quality_relax_threshold', 0.93))
        quality_strict_thr = float(self.config.get('pseudo_quality_strict_threshold', 0.85))
        quality_strict_scale = float(self.config.get('pseudo_quality_strict_scale', 0.20))
        base_thr = float(getattr(pseudo_label_generator, 'confidence_threshold', self.config.get('pseudo_label_threshold', 0.75)))
        min_thr = float(getattr(pseudo_label_generator, 'min_threshold', self.config.get('pseudo_threshold_min', base_thr)))
        base_margin = float(getattr(pseudo_label_generator, 'margin_threshold', self.config.get('proto_margin', 0.1)))
        hard_boost = self.config.get('sampler_hard_class_boost', {}) or {}
        per_class_thr = {}
        per_class_margin = {}

        for cls in range(self.num_classes):
            count = int(counts[cls])
            if count <= 0:
                continue

            quality = self.pseudo_class_precision_ema.get(cls)
            thr = base_thr + float(hard_extra.get(cls, 0.0))
            margin = base_margin
            allow_rare_relax = True

            if cls in self.hard_class_ids and (quality is None or quality < quality_relax_thr):
                allow_rare_relax = False

            if quality is not None and quality < quality_strict_thr:
                quality_gap = quality_strict_thr - quality
                thr += quality_strict_scale * quality_gap
                margin += max(margin_gamma, 0.02) * quality_gap
                allow_rare_relax = False

            if allow_rare_relax:
                rarity = 1.0 - np.sqrt(count / max_count)
                boost = float(hard_boost.get(cls, 1.0)) - 1.0
                relax = rare_gamma * max(0.0, rarity + 0.25 * boost)
                margin_relax = margin_gamma * max(0.0, rarity + 0.25 * boost)
                thr -= relax
                margin -= margin_relax

            per_class_thr[cls] = max(min_thr, min(0.995, float(thr)))
            per_class_margin[cls] = max(0.02, min(0.35, float(margin)))

        pseudo_label_generator.per_class_thr = per_class_thr
        pseudo_label_generator.per_class_margin = per_class_margin

        if hasattr(pseudo_label_generator, 'target_distribution'):
            target_distribution = counts.astype(np.float32)
            target_distribution = target_distribution / max(float(target_distribution.sum()), 1.0)
            pseudo_label_generator.target_distribution = target_distribution

        if hasattr(pseudo_label_generator, 'class_reliability'):
            reliability = {}
            gate_enabled = bool(self.config.get('pseudo_reliability_gate', False))
            gate_warmup = int(self.config.get('pseudo_reliability_warmup_epochs', 0))
            for cls in range(self.num_classes):
                quality = self.pseudo_class_precision_ema.get(cls)
                if not gate_enabled or self.current_epoch < gate_warmup or quality is None:
                    reliability[cls] = 1.0
                else:
                    reliability[cls] = max(
                        float(self.config.get('pseudo_reliability_floor', 0.35)),
                        min(1.0, float(quality))
                    )
            pseudo_label_generator.class_reliability = reliability

        label_names = {0: 'walk', 1: 'bike', 2: 'bus', 3: 'car', 4: 'subway'}
        thr_str = ', '.join(
            f"{label_names.get(cls, cls)}:{per_class_thr[cls]:.3f}/{per_class_margin[cls]:.3f}"
            for cls in range(self.num_classes) if cls in per_class_thr
        )
        if thr_str:
            self.logger.info(f"Pseudo per-class threshold/margin: [{thr_str}]")

    def _empty_prototype_bank(self) -> torch.Tensor:
        dim = self.prototypes.shape[-1]
        if self.prototypes_per_class > 1:
            return torch.zeros(self.num_classes, self.prototypes_per_class, dim, device=self.device)
        return torch.zeros(self.num_classes, dim, device=self.device)

    def _fit_class_prototypes(self, class_embeddings: np.ndarray, class_id: Optional[int] = None) -> Optional[torch.Tensor]:
        if len(class_embeddings) == 0:
            return None

        z = torch.from_numpy(class_embeddings).float().to(self.device)
        z = F.normalize(z, dim=1)

        if self.prototypes_per_class == 1:
            return F.normalize(z.mean(dim=0, keepdim=True), dim=1).squeeze(0)

        effective_k = self._prototype_count_for_class(class_id if class_id is not None else -1)
        effective_k = min(max(1, effective_k), self.prototypes_per_class)
        if effective_k == 1:
            centers = F.normalize(z.mean(dim=0, keepdim=True), dim=1)
        elif z.size(0) >= effective_k:
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=effective_k, random_state=self.config.get('random_seed', 42), n_init=10)
                centers = torch.from_numpy(km.fit(class_embeddings).cluster_centers_).float().to(self.device)
                centers = F.normalize(centers, dim=1)
            except Exception:
                idx = torch.arange(effective_k, device=self.device) % z.size(0)
                centers = F.normalize(z[idx], dim=1)
        elif z.size(0) == 1:
            centers = z.repeat(effective_k, 1)
        else:
            idx = torch.arange(effective_k, device=self.device) % z.size(0)
            centers = F.normalize(z[idx], dim=1)

        if centers.size(0) < self.prototypes_per_class:
            pad = centers[-1:].repeat(self.prototypes_per_class - centers.size(0), 1)
            centers = torch.cat([centers, pad], dim=0)
        return F.normalize(centers, dim=1)

    def _estimate_prototypes(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        prototype_bank = self._empty_prototype_bank()
        class_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)

        for class_id in range(self.num_classes):
            class_embeddings = embeddings[labels == class_id]
            class_proto = self._fit_class_prototypes(class_embeddings, class_id=class_id)
            if class_proto is None:
                continue
            prototype_bank[class_id] = class_proto
            class_mask[class_id] = True

        return prototype_bank, class_mask

    def _assign_prototypes(self, prototypes: torch.Tensor):
        self.prototypes = prototypes.to(self.device)

    def _build_graph_adj(
            self,
            embeddings: torch.Tensor,
            batch_indices: Optional[torch.Tensor] = None,
            use_global_graph: bool = True,
    ) -> Optional[torch.Tensor]:
        if embeddings.size(0) <= 1:
            return torch.zeros(
                embeddings.size(0),
                embeddings.size(0),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        adj = None
        if (
                use_global_graph
                and batch_indices is not None
                and getattr(self, 'global_knn_indices', None) is not None
        ):
            adj = self._build_batch_adj_from_global(batch_indices, embeddings.device)

        if adj is None or adj.sum() == 0:
            emb_norm = F.normalize(embeddings.detach(), dim=1)
            graph_k = int(self.config.get('graph_k', 10))
            adj = self._build_batch_knn_adj(emb_norm, graph_k)

        return adj

    def _maybe_graph_aggregate(
            self,
            embeddings: torch.Tensor,
            batch_indices: Optional[torch.Tensor] = None,
            use_global_graph: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if getattr(self, 'graph_agg', None) is None:
            return embeddings, None

        with torch.no_grad():
            adj = self._build_graph_adj(
                embeddings,
                batch_indices=batch_indices,
                use_global_graph=use_global_graph,
            )

        if adj is None:
            return embeddings, None

        return self.graph_agg(embeddings, adj), adj

    def _ema_update_prototypes(self, embeddings: np.ndarray, labels: np.ndarray, ema: float) -> int:
        new_prototypes, class_mask = self._estimate_prototypes(embeddings, labels)
        if not class_mask.any():
            return 0

        if self.prototypes_per_class == 1:
            blended = ema * self.prototypes[class_mask] + (1.0 - ema) * new_prototypes[class_mask]
            self.prototypes[class_mask] = F.normalize(blended, dim=1)
        else:
            blended = ema * self.prototypes[class_mask] + (1.0 - ema) * new_prototypes[class_mask]
            self.prototypes[class_mask] = F.normalize(blended, dim=2)

        return int(class_mask.sum().item())

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.projector.train()
        if getattr(self, 'graph_agg', None) is not None:
            self.graph_agg.train()

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
                if key in epoch_metrics:
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
                self._trainable_parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict['total_loss_tensor'].backward()
            torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.optimizer.step()

        # Return scalar losses
        return {
            'total_loss': loss_dict['total_loss'],
            'contrast_loss': loss_dict['contrast_loss'],
            'proto_loss': loss_dict['proto_loss']
        }

    @torch.no_grad()
    def _sync_teacher_from_student(self):
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_projector.load_state_dict(self.projector.state_dict())
        if self.teacher_classifier is not None and self.classifier is not None:
            self.teacher_classifier.load_state_dict(self.classifier.state_dict())

    def run_self_supervised_pretrain(self):
        if self.pretrain_completed:
            return
        if not bool(self.config.get('use_ssl_pretrain', False)):
            self.pretrain_completed = True
            return
        pretrain_epochs = int(self.config.get('pretrain_epochs', 0))
        if pretrain_epochs <= 0:
            self.pretrain_completed = True
            return

        self.logger.info(f"Starting train-split SSL pretraining for {pretrain_epochs} epochs...")
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        if getattr(self, 'graph_agg', None) is not None:
            params += list(self.graph_agg.parameters())
        optimizer = optim.AdamW(
            params,
            lr=float(self.config.get('pretrain_lr', self.config.get('lr', 3e-4))),
            weight_decay=float(self.config.get('pretrain_weight_decay', self.config.get('weight_decay', 1e-4))),
        )
        smooth_w = float(self.config.get('pretrain_graph_smooth_weight', 0.0))
        contrast_w = float(self.config.get('pretrain_graph_contrast_weight', 0.0))

        self.encoder.train()
        self.projector.train()
        if getattr(self, 'graph_agg', None) is not None:
            self.graph_agg.train()

        pretrain_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            collate_fn=traj_collate_fn,
            num_workers=0,
        )

        for epoch in range(pretrain_epochs):
            total = 0.0
            nb = 0
            pbar = tqdm(pretrain_loader, desc=f'Pretrain {epoch + 1}/{pretrain_epochs}')
            for batch in pbar:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                batch_indices = batch['indices'] if 'indices' in batch else None

                view1_coords = self._augment_batch(coords, lengths)
                view2_coords = self._augment_batch(coords, lengths)
                feat1 = self._augment_features(features)
                feat2 = self._augment_features(features)

                emb1 = self.encoder(feat1, view1_coords, lengths)
                emb2 = self.encoder(feat2, view2_coords, lengths)
                emb1, graph_adj = self._maybe_graph_aggregate(emb1, batch_indices=batch_indices, use_global_graph=False)
                emb2, _ = self._maybe_graph_aggregate(emb2, batch_indices=batch_indices, use_global_graph=False)
                z1 = self.projector(emb1)
                z2 = self.projector(emb2)
                loss = self.contrast_loss_fn(z1, z2)

                if smooth_w > 0.0 or contrast_w > 0.0:
                    z1_norm = F.normalize(z1, dim=1)
                    adj = graph_adj if graph_adj is not None else self._build_graph_adj(z1_norm, batch_indices=batch_indices, use_global_graph=False)
                    if smooth_w > 0.0:
                        loss = loss + smooth_w * self.graph_smooth_loss(z1_norm, adj)
                    if contrast_w > 0.0:
                        loss = loss + contrast_w * self.neighbor_contrast_loss(z1_norm, adj)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.config.get('max_grad_norm', 1.0))
                optimizer.step()

                total += float(loss.item())
                nb += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            self.logger.info(f"Pretrain epoch {epoch + 1}/{pretrain_epochs} | loss={total / max(nb, 1):.4f}")

        self._sync_teacher_from_student()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.pretrain_completed = True
        self.logger.info("SSL pretraining finished; optimizer and scheduler reset for finetuning.")

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
        self.encoder.eval()
        self.projector.eval()
        if getattr(self, 'classifier', None) is not None:
            self.classifier.eval()
        if getattr(self, 'graph_agg', None) is not None:
            self.graph_agg.eval()

        val_metrics = {
            'total_loss': 0.0,
            'contrast_loss': 0.0,
            'proto_loss': 0.0,
            'ce_loss': 0.0,
        }
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_indices = batch['indices'].to(self.device) if 'indices' in batch else None

            emb = self.encoder(features, coords, lengths)
            emb, _ = self._maybe_graph_aggregate(
                emb,
                batch_indices=batch_indices,
                use_global_graph=False,
            )
            z = self.projector(emb)

            contrast_loss = torch.tensor(0.0, device=self.device)
            proto_loss = torch.tensor(0.0, device=self.device)
            ce_loss = torch.tensor(0.0, device=self.device)
            labeled_mask = labels >= 0

            if labeled_mask.sum() > 0:
                proto_loss = self.proto_loss_fn(
                    z[labeled_mask], labels[labeled_mask], self.prototypes
                )

                if getattr(self, 'classifier', None) is not None and hasattr(self, 'ce_loss_fn'):
                    logits = self.classifier(emb[labeled_mask])
                    ce_loss = self.ce_loss_fn(logits, labels[labeled_mask])
                else:
                    logits = compute_prototype_logits(
                        z[labeled_mask],
                        self.prototypes,
                        temperature=self.config.get('temperature', 0.07),
                        aggregation=self.prototype_pooling,
                        pool_temperature=self.prototype_pool_temperature,
                    )

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels[labeled_mask].cpu().numpy())

            total_loss = (
                self._current_contrast_weight() * contrast_loss +
                self._current_proto_weight() * proto_loss +
                self._current_classifier_weight() * ce_loss
            )

            val_metrics['total_loss'] += total_loss.item()
            val_metrics['contrast_loss'] += contrast_loss.item()
            val_metrics['proto_loss'] += proto_loss.item()
            val_metrics['ce_loss'] += ce_loss.item()

        num_batches = len(self.val_loader)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        if all_labels:
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            val_metrics.update(self._compute_eval_metrics(y_true, y_pred))
        else:
            val_metrics.update(self._compute_eval_metrics(np.array([]), np.array([])))

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

                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                selection_score = self._get_selection_score(val_metrics)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step()

                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['val_loss'].append(val_metrics['total_loss'])
                self.history['contrast_loss'].append(train_metrics.get('contrast_loss', 0.0))
                self.history['proto_loss'].append(train_metrics.get('proto_loss', 0.0))
                self.history['ce_loss'].append(train_metrics.get('ce_loss', 0.0))
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
                self.history['metrics'].append({
                    'accuracy': val_metrics.get('accuracy', 0.0),
                    'macro_f1': val_metrics.get('macro_f1', 0.0),
                    'balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0),
                    'selection_metric': self.selection_metric,
                    'pseudo_adoption_rate': self.last_pseudo_monitor.get('adoption_rate') if isinstance(self.last_pseudo_monitor, dict) else None,
                    'selection_score': selection_score,
                })

                self.logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"Acc: {val_metrics.get('accuracy', 0.0):.4f}, "
                    f"Macro-F1: {val_metrics.get('macro_f1', 0.0):.4f}, "
                    f"BalAcc: {val_metrics.get('balanced_accuracy', 0.0):.4f}, "
                    f"{self.selection_metric}: {selection_score:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

                is_best = self._is_improved(selection_score, min_delta)
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    self.best_val_score = selection_score
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    self.logger.info(
                        f"✅ New best model saved! {self.selection_metric}: {self.best_val_score:.4f}"
                    )
                else:
                    self.patience_counter += 1

                if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")
            self.save_checkpoint('final_model.pth')
            self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'prototypes': self.prototypes,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_score': self.best_val_score,
            'selection_metric': self.selection_metric,
            'config': self.config,
            'history': self.history
        }

        # 保存分类头（如果存在）
        if hasattr(self, 'classifier') and self.classifier is not None:
            checkpoint['classifier_state_dict'] = self.classifier.state_dict()
        if getattr(self, 'graph_agg', None) is not None:
            checkpoint['graph_agg_state_dict'] = self.graph_agg.state_dict()

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
        if getattr(self, 'classifier', None) is not None and 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if getattr(self, 'graph_agg', None) is not None and 'graph_agg_state_dict' in checkpoint:
            self.graph_agg.load_state_dict(checkpoint['graph_agg_state_dict'])
        self.prototypes = checkpoint['prototypes'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
        self.best_val_score = checkpoint.get('best_val_score', self.best_val_loss)
        self.selection_metric = checkpoint.get('selection_metric', self.selection_metric)
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

    def __init__(self, *args, pseudo_label_generator=None, classifier=None, graph_agg=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = classifier
        self.classifier_weight = float(self.config.get('classifier_weight', 1.0))
        self.classifier_weight_final = float(self.config.get('classifier_weight_final', self.classifier_weight))
        cw = self.config.get('class_weights', None)
        cw_t = torch.tensor(cw, dtype=torch.float32).to(self.device) if cw is not None else None
        label_smoothing = float(self.config.get('classifier_label_smoothing', 0.0))
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=label_smoothing)

        use_gnn = self.config.get('use_gnn_aggregation', True)
        if use_gnn:
            if graph_agg is None:
                from models.encoders import GraphAggregationLayer
                hidden_dim = self.config.get('hidden_dim', 256)
                graph_agg = GraphAggregationLayer(hidden_dim, dropout=0.1).to(self.device)
            self.graph_agg = graph_agg.to(self.device)
            self.logger.info("GNN 聚合层已启用")
        else:
            self.graph_agg = None

        # 重新构建 optimizer/scheduler，确保 classifier / graph_agg 参数被正确纳入
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Additional loss weights
        self.pseudo_weight = self.config.get('pseudo_weight', 1.0)
        self.consistency_weight = self.config.get('consistency_weight', 0.1)

        # Consistency loss
        self.consistency_loss_fn = ConsistencyLoss()

        # Teacher model (EMA)
        self.teacher_encoder = self._create_teacher_model()
        self.ema_decay = self.config.get('ema_decay', 0.999)

        import copy
        self.teacher_projector = copy.deepcopy(self.projector).to(self.device)
        for p in self.teacher_projector.parameters():
            p.requires_grad = False

        # Teacher classifier EMA
        if classifier is not None:
            self.teacher_classifier = copy.deepcopy(classifier).to(self.device)
            for p in self.teacher_classifier.parameters():
                p.requires_grad = False
        else:
            self.teacher_classifier = None

        # Pseudo-label state
        self.pseudo_labels_dict = None
        # Online hidden monitoring
        self.hidden_true_labels = None
        self.pseudo_active_epoch = None
        self.best_after_pseudo_score = -1.0
        self.pseudo_class_precision_ema = {}
        self.last_pseudo_monitor = {}
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
        self.memory_h = None  # np.ndarray [N, H]
        self.memory_z = None  # np.ndarray [N, D] (normalized)
        self.pretrain_completed = False

        self.logger.info("Semi-supervised trainer initialized")

    def _create_teacher_model(self, AdaptiveTrajectoryEncoder=None) -> nn.Module:
        from models.encoders import AdaptiveTrajectoryEncoder
        teacher = AdaptiveTrajectoryEncoder(
            feat_dim=self.config.get('feat_dim', 36),
            coord_dim=self.config.get('coord_dim', 4),
            hidden_dim=self.config.get('hidden_dim', 256),
            dropout=self.config.get('dropout', 0.2),
            num_heads=self.config.get('num_attention_heads', 8),
            num_layers=self.config.get('num_encoder_layers', 3),
            encoder_mode=self.config.get('encoder_mode', 'adaptive_gate')
        ).to(self.device)
        teacher.load_state_dict(self.encoder.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    # 同步 encoder + projector + classifier
    @torch.no_grad()
    def update_teacher(self):
        for tp, sp in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * sp.data
        for tp, sp in zip(self.teacher_projector.parameters(), self.projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * sp.data
        if self.teacher_classifier is not None and self.classifier is not None:
            for tp, sp in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
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

        raw_emb1 = self.encoder(feat1, view1_coords, lengths)
        raw_emb2 = self.encoder(feat2, view2_coords, lengths)

        raw_z1 = self.projector(raw_emb1)
        raw_z1 = F.normalize(raw_z1, dim=1)

        emb1, graph_adj = self._maybe_graph_aggregate(
            raw_emb1,
            batch_indices=batch_indices,
            use_global_graph=True,
        )
        emb2, _ = self._maybe_graph_aggregate(
            raw_emb2,
            batch_indices=batch_indices,
            use_global_graph=True,
        )

        z1 = self.projector(emb1)
        z2 = self.projector(emb2)

        # 对比损失（视图）
        contrast_loss = self.contrast_loss_fn(z1, z2)
        labeled_mask = labels >= 0
        proto_loss = torch.tensor(0.0, device=self.device)
        if labeled_mask.sum() > 0:
            proto_loss = self.proto_loss_fn(z1[labeled_mask], labels[labeled_mask], self.prototypes)

        # === 分类头 CE 损失（主要监督信号，直接在 h 上分类）===
        ce_loss = torch.tensor(0.0, device=self.device)
        hard_negative_loss = torch.tensor(0.0, device=self.device)
        coarse_loss = torch.tensor(0.0, device=self.device)
        if self.classifier is not None and labeled_mask.sum() > 0:
            logits = self.classifier(emb1[labeled_mask])
            ce_loss = self.ce_loss_fn(logits, labels[labeled_mask])
            hard_negative_loss = self._compute_hard_negative_loss(logits, labels[labeled_mask])
            coarse_loss = self._compute_coarse_aux_loss(logits, labels[labeled_mask])

        # 伪标签损失
        pseudo_loss = torch.tensor(0.0, device=self.device)
        if (self.pseudo_labels_dict is not None and
                'labels' in self.pseudo_labels_dict and
                'confidences' in self.pseudo_labels_dict and
                'indices' in self.pseudo_labels_dict):
            batch_idx = self.pseudo_labels_dict['indices']
            pseudo_loss = self._compute_pseudo_loss(z1, labels, batch_idx, h=emb1)

        # 一致性（teacher-student）仅约束原始 encoder/projector 表示，避免 teacher/student 图路径不对齐
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_weight > 0:
            with torch.no_grad():
                teacher_emb = self.teacher_encoder(features, coords, lengths)
                teacher_z = self.teacher_projector(teacher_emb)
                teacher_z = F.normalize(teacher_z, dim=1)
            consistency_loss = self.consistency_loss_fn(raw_z1, teacher_z)

        # ===== 图损失（优先使用全局 kNN 图，退化时用 batch 内 kNN 图）=====
        graph_smooth = torch.tensor(0.0, device=self.device)
        graph_contrast = torch.tensor(0.0, device=self.device)
        if (self.lambda_graph_smooth > 0 or self.lambda_graph_contrast > 0):
            z1_norm = F.normalize(z1, dim=1)
            adj_b = graph_adj
            if adj_b is None:
                adj_b = self._build_graph_adj(
                    z1_norm,
                    batch_indices=batch_indices,
                    use_global_graph=True,
                )
            if self.lambda_graph_smooth > 0:
                graph_smooth = self.graph_smooth_loss(z1_norm, adj_b)
            if self.lambda_graph_contrast > 0:
                graph_contrast = self.neighbor_contrast_loss(z1_norm, adj_b)

        total_loss = (
                self._current_classifier_weight() * ce_loss +
                self._current_hard_negative_weight() * hard_negative_loss +
                self._current_coarse_aux_weight() * coarse_loss +
                self._current_contrast_weight() * contrast_loss +
                self._current_proto_weight() * proto_loss +
                self._current_pseudo_weight() * pseudo_loss +
                self._current_consistency_weight() * consistency_loss +
                self._current_graph_weight(self.lambda_graph_smooth) * graph_smooth +
                self._current_graph_weight(self.lambda_graph_contrast) * graph_contrast
        )

        return {
            'total_loss_tensor': total_loss,
            'total_loss': total_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'proto_loss': proto_loss.item(),
            'pseudo_loss': pseudo_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'graph_smooth_loss': graph_smooth.item(),
            'graph_contrast_loss': graph_contrast.item(),
            'ce_loss': ce_loss.item(),
            'hard_negative_loss': hard_negative_loss.item(),
            'coarse_loss': coarse_loss.item(),
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
    def _extract_all_projected_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """提取训练集所有样本的 h/z，用于构图、传播与全局伪标签监督。"""
        self.encoder.eval(); self.projector.eval()
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )
        all_h = []
        all_z = []
        for batch in loader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            batch_indices = batch['indices'].to(self.device) if 'indices' in batch else None

            emb = self.encoder(features, coords, lengths)
            emb, _ = self._maybe_graph_aggregate(
                emb,
                batch_indices=batch_indices,
                use_global_graph=False,
            )
            z = self.projector(emb)
            z = F.normalize(z, dim=1)
            all_h.append(emb.cpu().numpy())
            all_z.append(z.cpu().numpy())
        self.encoder.train(); self.projector.train()
        return np.vstack(all_h), np.vstack(all_z)

    def _rebuild_global_knn_graph(self):
        """重建全局 kNN 图（分块余弦相似，避免 O(N^2) 内存峰值）。"""
        import numpy as np

        self.memory_h, self.memory_z = self._extract_all_projected_embeddings()  # [N,H], [N,D], normalized
        N = self.memory_z.shape[0]
        k = min(self.graph_k, max(1, N - 1))
        z = torch.from_numpy(self.memory_z).float()

        chunk = self.config.get('knn_chunk_size', 512)
        knn_idx = np.empty((N, k), dtype=np.int64)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            sim_block = torch.mm(z[start:end], z.t())  # [chunk, N]
            sim_block[:, start:end].fill_diagonal_(-1.0)  # 排除自身（对角块）
            topk = sim_block.topk(k, dim=1).indices.numpy()
            knn_idx[start:end] = topk

        self.global_knn_indices = knn_idx  # [N, k]

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
        if B <= 1:
            return torch.zeros(B, B, device=z.device, dtype=z.dtype)
        sim = torch.matmul(z, z.t())             # [B,B], z 已归一化
        sim.fill_diagonal_(0.0)
        k = min(k, B - 1)
        topk = sim.topk(k, dim=1).indices        # [B,k]
        adj = torch.zeros(B, B, device=z.device)
        adj.scatter_(1, topk, torch.ones_like(topk, dtype=adj.dtype))
        adj = torch.maximum(adj, adj.t())
        return adj


    # training/trainer.py (class SemiSupervisedTrainer)
    import torch.nn.functional as F

    def _compute_pseudo_loss(
            self,
            embeddings: torch.Tensor,   # z (projector 输出)，用于 prototype loss
            labels: torch.Tensor,
            batch_indices: torch.Tensor,
            h: torch.Tensor = None      # encoder 输出 h，用于 classifier CE loss
    ) -> torch.Tensor:
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.detach().cpu().numpy()

        batch_pseudo_labels = self.pseudo_labels_dict['labels'][batch_indices]
        batch_pseudo_confs = self.pseudo_labels_dict['confidences'][batch_indices]

        pseudo_mask = (batch_pseudo_labels >= 0) & (labels.cpu().numpy() < 0)
        if pseudo_mask.sum() == 0:
            return self._compute_global_pseudo_loss()

        pseudo_mask_t = torch.from_numpy(pseudo_mask).to(self.device).bool()
        pseudo_labels_tensor = torch.LongTensor(batch_pseudo_labels[pseudo_mask]).to(self.device)
        pseudo_conf_tensor = torch.FloatTensor(batch_pseudo_confs[pseudo_mask]).to(self.device)
        pseudo_z = embeddings[pseudo_mask_t]

        # Prototype 头 pseudo loss
        logits = compute_prototype_logits(
            pseudo_z,
            self.prototypes,
            temperature=self.config.get('temperature', 0.07),
            aggregation=self.prototype_pooling,
            pool_temperature=self.prototype_pool_temperature,
        )
        sample_class_w = self._get_class_sample_weights(pseudo_labels_tensor)
        per_sample_loss = F.cross_entropy(logits, pseudo_labels_tensor, reduction='none')
        proto_pseudo_loss = (per_sample_loss * pseudo_conf_tensor * sample_class_w).mean()

        # Classifier 头 pseudo loss（同时监督主推理头）
        clf_pseudo_loss = torch.tensor(0.0, device=self.device)
        if self.classifier is not None and h is not None:
            pseudo_h = h[pseudo_mask_t]
            clf_logits = self.classifier(pseudo_h)
            clf_per_sample = F.cross_entropy(clf_logits, pseudo_labels_tensor, reduction='none')
            clf_pseudo_loss = (clf_per_sample * pseudo_conf_tensor * sample_class_w).mean()

        return proto_pseudo_loss + clf_pseudo_loss

    def _compute_global_pseudo_loss(self, n_sample: int = 16) -> torch.Tensor:
        """当 batch 内无伪标签时，从全局伪标签池随机采样计算 pseudo loss（解决零梯度问题）。"""
        if self.memory_z is None or self.pseudo_labels_dict is None:
            return torch.tensor(0.0, device=self.device)
        pl = self.pseudo_labels_dict['labels']
        obs_labels = self.pseudo_labels_dict.get('observed_labels', None)
        if obs_labels is None:
            return torch.tensor(0.0, device=self.device)
        pool = np.where((pl >= 0) & (obs_labels < 0))[0]
        if len(pool) == 0:
            return torch.tensor(0.0, device=self.device)
        idx = np.random.choice(pool, size=min(n_sample, len(pool)), replace=False)
        z_pseudo = torch.from_numpy(self.memory_z[idx]).float().to(self.device)
        pseudo_labels_t = torch.LongTensor(pl[idx]).to(self.device)
        pseudo_conf_t = torch.FloatTensor(self.pseudo_labels_dict['confidences'][idx]).to(self.device)
        logits = compute_prototype_logits(
            z_pseudo,
            self.prototypes,
            temperature=self.config.get('temperature', 0.07),
            aggregation=self.prototype_pooling,
            pool_temperature=self.prototype_pool_temperature,
        )
        sample_class_w = self._get_class_sample_weights(pseudo_labels_t)
        per_sample = F.cross_entropy(logits, pseudo_labels_t, reduction='none')
        proto_loss = (per_sample * pseudo_conf_t * sample_class_w).mean()

        clf_loss = torch.tensor(0.0, device=self.device)
        if self.classifier is not None and self.memory_h is not None:
            h_pseudo = torch.from_numpy(self.memory_h[idx]).float().to(self.device)
            clf_logits = self.classifier(h_pseudo)
            clf_per_sample = F.cross_entropy(clf_logits, pseudo_labels_t, reduction='none')
            clf_loss = (clf_per_sample * pseudo_conf_t * sample_class_w).mean()

        return proto_loss + clf_loss

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (semi-supervised, with extra metrics)."""
        self.encoder.train()
        self.projector.train()
        if getattr(self, 'classifier', None) is not None:
            self.classifier.train()
        if getattr(self, 'graph_agg', None) is not None:
            self.graph_agg.train()

        epoch_metrics = {
            'total_loss': 0.0,
            'contrast_loss': 0.0,
            'proto_loss': 0.0,
            'ce_loss': 0.0,
            'hard_negative_loss': 0.0,
            'coarse_loss': 0.0,
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

            for k in epoch_metrics.keys():
                if k in loss_dict:
                    epoch_metrics[k] += loss_dict[k]

            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'ctr': f"{loss_dict.get('contrast_loss', 0.0):.4f}",
                'proto': f"{loss_dict.get('proto_loss', 0.0):.4f}",
                'ce': f"{loss_dict.get('ce_loss', 0.0):.4e}",
                'pseudo': f"{loss_dict.get('pseudo_loss', 0.0):.4e}",
                'cons': f"{loss_dict.get('consistency_loss', 0.0):.4e}",
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
                self._trainable_parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict['total_loss_tensor'].backward()
            torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters(),
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
            'graph_contrast_loss': loss_dict.get('graph_contrast_loss', 0.0),
            'ce_loss': loss_dict.get('ce_loss', 0.0),
            'hard_negative_loss': loss_dict.get('hard_negative_loss', 0.0),
            'coarse_loss': loss_dict.get('coarse_loss', 0.0)
        }
        return out

    from data.dataset import traj_collate_fn


    @torch.no_grad()
    def _bootstrap_prototypes_from_labeled(self):
        """Backward-compatible alias for prototype initialization."""
        self._initialize_prototypes_from_labeled()

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
        if max_epochs <= pseudo_interval:
            pseudo_interval = 1
        patience = self.config.get('patience', 30)
        min_delta = self.config.get('min_delta', 1e-4)
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
            self.run_self_supervised_pretrain()
            for epoch in range(max_epochs):
                self.current_epoch = epoch

                if epoch == 0:
                    self._initialize_prototypes_from_labeled()

                current_interval = int(self.config.get('pseudo_label_interval', pseudo_interval))
                if epoch >= warmup_epochs and current_interval > 0 and epoch % current_interval == 0:
                    if self.pseudo_label_generator is not None:
                        self.update_pseudo_labels(self.pseudo_label_generator)

                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                selection_score = self._get_selection_score(val_metrics)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step()

                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['val_loss'].append(val_metrics['total_loss'])
                self.history['contrast_loss'].append(train_metrics.get('contrast_loss', 0.0))
                self.history['proto_loss'].append(train_metrics.get('proto_loss', 0.0))
                self.history['ce_loss'].append(train_metrics.get('ce_loss', 0.0))
                self.history['hard_negative_loss'].append(train_metrics.get('hard_negative_loss', 0.0))
                self.history['coarse_loss'].append(train_metrics.get('coarse_loss', 0.0))
                self.history['pseudo_loss'].append(train_metrics.get('pseudo_loss', 0.0))
                self.history['consistency_loss'].append(train_metrics.get('consistency_loss', 0.0))
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
                self.history['metrics'].append({
                    'accuracy': val_metrics.get('accuracy', 0.0),
                    'macro_f1': val_metrics.get('macro_f1', 0.0),
                    'balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0),
                    'selection_metric': self.selection_metric,
                    'selection_score': selection_score,
                })

                self.logger.info(
                    "Epoch {}/{} - Train: {:.4f} [ctr {:.4f} | proto {:.4f} | ce {:.4e} | hard {:.4e} | coarse {:.4e} | pseudo {:.4e} | cons {:.4e}] "
                    "| Val: {:.4f} | Acc: {:.4f} | Macro-F1: {:.4f} | BalAcc: {:.4f} | {}: {:.4f} | LR: {:.6f}".format(
                        epoch + 1, max_epochs,
                        train_metrics['total_loss'],
                        train_metrics.get('contrast_loss', 0.0),
                        train_metrics.get('proto_loss', 0.0),
                        train_metrics.get('ce_loss', 0.0),
                        train_metrics.get('hard_negative_loss', 0.0),
                        train_metrics.get('coarse_loss', 0.0),
                        train_metrics.get('pseudo_loss', 0.0),
                        train_metrics.get('consistency_loss', 0.0),
                        val_metrics['total_loss'],
                        val_metrics.get('accuracy', 0.0),
                        val_metrics.get('macro_f1', 0.0),
                        val_metrics.get('balanced_accuracy', 0.0),
                        self.selection_metric,
                        selection_score,
                        self.optimizer.param_groups[0]['lr']
                    )
                )

                is_best = self._is_improved(selection_score, min_delta)
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    self.best_val_score = selection_score
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    self.logger.info(
                        f"✅ New best model saved! {self.selection_metric}: {self.best_val_score:.4f}"
                    )
                else:
                    self.patience_counter += 1

                # 伪标签激活后额外保存 best_after_pseudo
                if self.pseudo_active_epoch is not None:
                    if selection_score > self.best_after_pseudo_score:
                        self.best_after_pseudo_score = selection_score
                        self.save_checkpoint('best_after_pseudo.pth', is_best=False)
                        self.logger.info(
                            f"💡 best_after_pseudo updated: {self.selection_metric}={selection_score:.4f}"
                        )
                    # 早停保护：伪标签激活后至少再跑 patience_after_pseudo 个 epoch
                    patience_after_pseudo = self.config.get('patience_after_pseudo', 15)
                    if (self.current_epoch - self.pseudo_active_epoch < patience_after_pseudo
                            and self.patience_counter >= patience):
                        self.patience_counter = patience - 1

                if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

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
        """用有标签数据初始化原型。"""
        self.logger.info("🔧 Initializing prototypes from labeled samples...")

        self.encoder.eval()
        self.projector.eval()

        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        all_z, all_labels = [], []
        for batch in loader:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].cpu().numpy()

            labeled_mask = labels >= 0
            if not labeled_mask.any():
                continue

            emb = self.encoder(features[labeled_mask], coords[labeled_mask], lengths[labeled_mask])
            z = self.projector(emb)
            z = F.normalize(z, dim=1)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels[labeled_mask])

        if all_z:
            prototype_bank, class_mask = self._estimate_prototypes(
                np.vstack(all_z),
                np.hstack(all_labels)
            )
            self._assign_prototypes(prototype_bank)
            for class_id in range(self.num_classes):
                if class_mask[class_id]:
                    count = int((np.hstack(all_labels) == class_id).sum())
                    self.logger.info(f"   Class {class_id}: {count} samples")
                else:
                    self.logger.warning(f"   Class {class_id}: No labeled samples!")
        else:
            self.logger.warning("No labeled embeddings found during prototype initialization.")

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

        all_h, all_z, all_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Extracting z (aligned)'):
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels']
                batch_indices = batch['indices'].to(self.device) if 'indices' in batch else None

                emb = self.encoder(features, coords, lengths)
                emb, _ = self._maybe_graph_aggregate(
                    emb,
                    batch_indices=batch_indices,
                    use_global_graph=False,
                )
                z = self.projector(emb)
                z = F.normalize(z, dim=1)

                all_h.append(emb.cpu().numpy())
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.numpy())

        all_h = np.vstack(all_h)
        all_z = np.vstack(all_z)
        all_labels = np.hstack(all_labels)
        self.memory_h = all_h
        self.memory_z = all_z  # 供 _compute_global_pseudo_loss 使用（用于伪标签生成）
        all_teacher_clf_logits = None
        if self.teacher_classifier is not None and self.config.get('use_teacher_clf_pseudo', False):
            all_h_teacher = []
            with torch.no_grad():
                for batch in loader:
                    coords = batch['coords'].to(self.device)
                    features = batch['features'].to(self.device)
                    lengths = batch['lengths'].to(self.device)
                    h_t = self.teacher_encoder(features, coords, lengths)
                    logits_t = self.teacher_classifier(h_t)
                    all_h_teacher.append(logits_t.cpu().numpy())
            all_teacher_clf_logits = np.vstack(all_h_teacher)

        self._update_generator_class_thresholds(pseudo_label_generator, all_labels)

        # 新接口：直接传 z
        new_labels, confidences = pseudo_label_generator.generate_pseudo_labels(
            projected_embeddings=all_z,
            labels=all_labels,
            prototypes=self.prototypes,
            epoch=self.current_epoch,
            teacher_clf_logits=all_teacher_clf_logits
        )
        # ====== 图式伪标签传播（LP） + 融合 ======
        # 全局 kNN 图仅供 LP 使用，在此按 interval 重建
        try:
            if self.current_epoch % max(1, self.graph_build_interval) == 0:
                self._rebuild_global_knn_graph()
        except Exception as e:
            self.logger.warning(f"Global graph build failed: {e}")
            self.global_knn_indices = None
        try:
            glp_labels, glp_conf = graph_label_propagation(
                projected_embeddings=all_z,
                labels=all_labels,
                k=self.graph_k,
                alpha=self.config.get('lp_alpha', 0.9),
                iters=self.config.get('lp_iters', 20),
                min_support=self.config.get('lp_min_support', 0.60),
                conf_power=self.config.get('pseudo_lp_conf_power', 0.75),
                min_purity=self.config.get('pseudo_lp_min_purity', 0.0),
            )
            static_thr = float(self.config.get('pseudo_label_threshold', 0.8))
            if hasattr(pseudo_label_generator, '_get_dynamic_threshold'):
                thr = pseudo_label_generator._get_dynamic_threshold(self.current_epoch)
            else:
                thr = static_thr
            candidate_base = int(((all_labels < 0) & (new_labels >= 0)).sum())
            candidate_lp = int(((all_labels < 0) & (glp_labels >= 0)).sum())
            candidate_agree = int(((all_labels < 0) & (new_labels >= 0) & (glp_labels >= 0) & (new_labels == glp_labels)).sum())
            self.logger.info(
                f"Pseudo candidates before fuse: base={candidate_base}, lp={candidate_lp}, agree={candidate_agree}, "
                f"lp_mean_conf={float(glp_conf[all_labels < 0].mean()) if (all_labels < 0).any() else 0.0:.3f}"
            )
            fused_labels, fused_conf, pseudo_sources = fuse_pseudo_labels(
                base_labels=new_labels,
                base_conf=confidences,
                glp_labels=glp_labels,
                glp_conf=glp_conf,
                observed_labels=all_labels,
                thr=max(0.5, min(0.95, thr)),
                lp_thr=self.config.get('pseudo_lp_threshold', 0.92),
                conflict_thr=self.config.get('pseudo_conflict_threshold', 0.95),
                conflict_margin=self.config.get('pseudo_conflict_margin', 0.15),
                allow_lp_only=self._current_allow_lp_only(),
                lp_agree_bonus=self.config.get('pseudo_lp_agree_bonus', 0.0),
                lp_agree_threshold_offset=self.config.get('pseudo_lp_agree_threshold_offset', 0.03),
                return_sources=True,
            )
            new_labels = fused_labels
            confidences = fused_conf
            self.logger.info("Pseudo labels fused with Graph LP.")
        except Exception as e:
            self.logger.warning(f"Graph LP failed, use base pseudo labels only: {e}")

        # 伪标签采纳上限：阻止后期确认偏差雪崩
        max_rate = self._current_pseudo_cap_rate()
        max_count = int(self.config.get('pseudo_max_adoption_count', 0))
        new_labels, confidences, cap_stats = apply_pseudo_label_acceptance_cap(
            new_labels,
            confidences,
            observed_labels=all_labels,
            max_rate=max_rate,
            max_count=max_count,
            class_balance=bool(self.config.get('pseudo_adaptive_per_class', False)),
            class_balance_power=float(self.config.get('pseudo_class_balance_power', 0.5)),
            min_per_class=int(self.config.get('pseudo_class_balance_min_count', 0)),
        )

        # 每类配额限制（可选）
        quota = int(self.config.get('pseudo_class_quota_per_update', 0))
        if quota > 0:
            new_labels, confidences = class_balanced_pseudo_cap(
                new_labels, confidences, all_labels, quota_per_class=quota
            )

        if 'pseudo_sources' not in locals():
            pseudo_sources = np.full(len(new_labels), PSEUDO_SOURCE_OBSERVED, dtype=np.int64)
            pseudo_sources[all_labels < 0] = PSEUDO_SOURCE_ABSTAIN
            pseudo_sources[(all_labels < 0) & (new_labels >= 0)] = PSEUDO_SOURCE_BASE_ONLY

        new_labels, confidences, pseudo_sources = self._apply_lp_source_cap(
            new_labels, confidences, all_labels, pseudo_sources
        )
        pseudo_sources[(all_labels < 0) & (new_labels < 0)] = PSEUDO_SOURCE_ABSTAIN

        # 统计采纳（详细监控）
        unlabeled_mask = all_labels < 0
        adopted_mask = (new_labels >= 0) & unlabeled_mask
        adopted = int(adopted_mask.sum())
        total_unlabeled = int(unlabeled_mask.sum())
        adoption_rate = adopted / max(total_unlabeled, 1)

        label_names = {0: 'walk', 1: 'bike', 2: 'bus', 3: 'car', 4: 'subway'}
        per_class_str = ', '.join([
            f"{label_names.get(c, c)}:{int((new_labels[adopted_mask] == c).sum())}"
            for c in range(self.num_classes)
        ])

        if adopted > 0:
            adopted_confs = confidences[adopted_mask]
            conf_info = f"conf(mean={adopted_confs.mean():.3f}, min={adopted_confs.min():.3f}, max={adopted_confs.max():.3f})"
        else:
            conf_info = "conf(N/A)"

        raw_before = int(cap_stats.get('before', adopted))
        if raw_before != adopted:
            cap_info = f" | cap {raw_before}->{adopted}"
        else:
            cap_info = ""

        source_counts = {
            'agree': int(((pseudo_sources == PSEUDO_SOURCE_AGREE) & adopted_mask).sum()),
            'base_only': int(((pseudo_sources == PSEUDO_SOURCE_BASE_ONLY) & adopted_mask).sum()),
            'lp_only': int(((pseudo_sources == PSEUDO_SOURCE_LP_ONLY) & adopted_mask).sum()),
            'base_conflict': int(((pseudo_sources == PSEUDO_SOURCE_BASE_CONFLICT) & adopted_mask).sum()),
            'lp_conflict': int(((pseudo_sources == PSEUDO_SOURCE_LP_CONFLICT) & adopted_mask).sum()),
        }
        source_info = (
            f"sources[agree={source_counts['agree']}, base={source_counts['base_only']}, "
            f"lp={source_counts['lp_only']}, base_conflict={source_counts['base_conflict']}, "
            f"lp_conflict={source_counts['lp_conflict']}]"
        )

        self.logger.info(
            f"✅ 伪标签更新: 采纳 {adopted}/{total_unlabeled} 个 "
            f"({adoption_rate:.1%}){cap_info} | {conf_info} | {source_info} | per-class: [{per_class_str}]"
        )

        pseudo_monitor = self._build_pseudo_monitor(all_labels, new_labels, confidences, pseudo_sources)
        self.last_pseudo_monitor = {
            'adoption_rate': float(adoption_rate),
            'per_class': pseudo_monitor,
        }
        self.history.setdefault('pseudo_monitor', []).append(self.last_pseudo_monitor)

        label_names = {0: 'walk', 1: 'bike', 2: 'bus', 3: 'car', 4: 'subway'}
        monitor_parts = []
        for cls in range(self.num_classes):
            stats = pseudo_monitor.get(cls, {})
            conf_str = 'N/A' if stats.get('mean_conf') is None else f"{stats['mean_conf']:.3f}"
            acc_str = 'N/A' if stats.get('hidden_acc') is None else f"{stats['hidden_acc']:.3f}"
            f1_str = 'N/A' if stats.get('hidden_f1') is None else f"{stats['hidden_f1']:.3f}"
            monitor_parts.append(
                f"{label_names.get(cls, cls)}:n={stats.get('adopted', 0)},conf={conf_str},acc={acc_str},f1={f1_str},"
                f"src(a/b/l)={stats.get('source_agree', 0)}/{stats.get('source_base', 0)}/{stats.get('source_lp', 0)}"
            )
        self.logger.info(f"📌 Pseudo per-class: [{' | '.join(monitor_parts)}]")
        self._update_pseudo_quality_ema(pseudo_monitor)

        self.pseudo_labels_dict = {
            'labels': new_labels,
            'confidences': confidences,
            'sources': pseudo_sources,
            'indices': None,
            'observed_labels': all_labels.copy(),
            'monitor': self.last_pseudo_monitor,
            'class_precision_ema': self.pseudo_class_precision_ema.copy(),
        }

        # 在线 hidden 监控
        if self.hidden_true_labels is not None:
            hidden_mask = (all_labels < 0) & (self.hidden_true_labels >= 0)
            if hidden_mask.sum() > 0:
                hidden_preds = new_labels[hidden_mask]
                hidden_true = self.hidden_true_labels[hidden_mask]
                accepted = hidden_preds >= 0
                if accepted.sum() > 0:
                    h_acc = accuracy_score(hidden_true[accepted], hidden_preds[accepted])
                    h_f1 = f1_score(hidden_true[accepted], hidden_preds[accepted], average='macro', zero_division=0)
                    self.logger.info(
                        f"📊 Hidden监控: 采纳={accepted.sum()}/{hidden_mask.sum()} "
                        f"({accepted.mean():.1%}) | acc={h_acc:.4f} | F1={h_f1:.4f}"
                    )
                    if self.pseudo_active_epoch is None:
                        self.pseudo_active_epoch = self.current_epoch

        # ====== 原型 EMA 更新：默认只用真标签，避免伪标签确认偏差反哺原型 ======
        try:
            ema = float(self.config.get('proto_ema', 0.9))
            conf_thr = float(self.config.get('proto_ema_conf_thr', 0.9))
            use_pseudo_for_proto = bool(self.config.get('use_pseudo_for_proto_ema', False))
            agreement_only = bool(self.config.get('pseudo_proto_ema_agreement_only', True))

            prototype_labels = all_labels.copy()
            if use_pseudo_for_proto:
                pseudo_high_mask = (unlabeled_mask) & (new_labels >= 0) & (confidences >= conf_thr)
                if agreement_only:
                    pseudo_high_mask &= (pseudo_sources == PSEUDO_SOURCE_AGREE)
                prototype_labels[pseudo_high_mask] = new_labels[pseudo_high_mask]

            known_mask = prototype_labels >= 0
            updated = self._ema_update_prototypes(all_z[known_mask], prototype_labels[known_mask], ema=ema)
            if updated > 0:
                suffix = '' if use_pseudo_for_proto else ' (labeled only)'
                self.logger.info(f"Prototypes EMA updated (classes: {updated}){suffix}")
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


