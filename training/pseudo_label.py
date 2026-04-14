import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import logging
from .loss import compute_prototype_logits


logger = logging.getLogger(__name__)

PSEUDO_SOURCE_ABSTAIN = 0
PSEUDO_SOURCE_OBSERVED = 1
PSEUDO_SOURCE_AGREE = 2
PSEUDO_SOURCE_BASE_ONLY = 3
PSEUDO_SOURCE_LP_ONLY = 4
PSEUDO_SOURCE_BASE_CONFLICT = 5
PSEUDO_SOURCE_LP_CONFLICT = 6


class PseudoLabelGenerator:
    """基础伪标签生成器 - 高置信度策略"""

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        pseudo_temperature: float = 0.30,
        prototype_pooling: str = 'max',
        prototype_pool_temperature: float = 1.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.pseudo_temperature = float(pseudo_temperature)
        self.prototype_pooling = str(prototype_pooling).lower()
        self.prototype_pool_temperature = float(prototype_pool_temperature)
        self.last_outputs: Dict[str, np.ndarray] = {}

    def generate(
        self,
        projected_embeddings: np.ndarray,  # 已投影到原型空间的 z
        labels: np.ndarray,
        prototypes: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            projected_embeddings: z (shape: [N, dim])，必须已是 projector 输出
            labels: 当前标签 (N,), -1 表示无标签
            prototypes: [C, dim]
            epoch: 仅为接口兼容，不使用
        Returns:
            (new_labels, confidences)；confidences 为长度 N 的数组（有标签样本置 1.0）
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        device = prototypes.device
        unlabeled_mask = labels < 0
        N = len(labels)

        all_confidences = np.zeros(N, dtype=np.float32)
        all_confidences[~unlabeled_mask] = 1.0
        num_classes = prototypes.size(0) if prototypes.dim() >= 2 else 0
        all_probs = np.zeros((N, num_classes), dtype=np.float32)
        if num_classes > 0 and (~unlabeled_mask).any():
            all_probs[np.where(~unlabeled_mask)[0], labels[~unlabeled_mask].astype(np.int64)] = 1.0

        if unlabeled_mask.sum() == 0:
            self.last_outputs = {
                'probabilities': all_probs,
                'confidences': all_confidences,
                'predictions': labels.copy(),
            }
            return labels, all_confidences

        z_u = torch.from_numpy(projected_embeddings[unlabeled_mask]).float().to(device)
        logits = compute_prototype_logits(
            z_u,
            prototypes,
            temperature=self.pseudo_temperature,
            aggregation=self.prototype_pooling,
            pool_temperature=self.prototype_pool_temperature,
        )
        probs = F.softmax(logits, dim=1)  # 转为概率，与阈值量纲一致
        confs, preds = probs.max(dim=1)

        high_mask = confs > self.confidence_threshold
        new_labels = labels.copy()
        u_idx = np.where(unlabeled_mask)[0]
        high_idx = u_idx[high_mask.cpu().numpy()]

        new_labels[high_idx] = preds[high_mask].cpu().numpy()
        all_confidences[u_idx] = confs.cpu().numpy()
        all_probs[u_idx] = probs.detach().cpu().numpy()
        pred_full = labels.copy()
        pred_full[u_idx] = preds.detach().cpu().numpy()
        self.last_outputs = {
            'probabilities': all_probs,
            'confidences': all_confidences.copy(),
            'predictions': pred_full,
        }

        return new_labels, all_confidences

    def generate_pseudo_labels(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def get_last_outputs(self) -> Dict[str, np.ndarray]:
        return self.last_outputs



class AdvancedPseudoLabelGenerator:
    """伪标签生成器：支持 per-class 阈值/ margin，一致性检查，可渐进阈值，teacher classifier 融合"""
    def __init__(self,
                 confidence_threshold: float = 0.85,
                 progressive_threshold: bool = True,
                 consistency_check: bool = True,
                 top_k_consistency: int = 3,
                 margin_threshold: float = 0.1,
                 per_class_thresholds: Optional[Dict[int, float]] = None,
                 per_class_margin: Optional[Dict[int, float]] = None,
                 pseudo_temperature: float = 0.30,
                 min_threshold: float = 0.65,
                 threshold_decay: float = 0.02,
                 threshold_decay_interval: int = 10,
                 low_margin_penalty: float = 0.90,
                 teacher_clf_weight: float = 0.0,
                 proto_weight: float = 1.0,
                 prototype_pooling: str = 'max',
                 prototype_pool_temperature: float = 1.0,
                 confidence_floor: float = 0.0,
                 distribution_alignment: bool = False,
                 distribution_momentum: float = 0.90,
                 distribution_min_prob: float = 1e-3,
                 teacher_temperature: float = 1.0,
                 proto_temperature: float = 1.0,
                 reliability_gate: bool = False,
                 reliability_power: float = 1.0,
                 reliability_floor: float = 0.35):
        self.confidence_threshold = confidence_threshold
        self.progressive_threshold = progressive_threshold
        self.consistency_check = consistency_check
        self.top_k = top_k_consistency
        self.margin_threshold = margin_threshold
        self.per_class_thr = per_class_thresholds or {}
        self.per_class_margin = per_class_margin or {}
        self.pseudo_temperature = float(pseudo_temperature)
        self.min_threshold = float(min_threshold)
        self.threshold_decay = float(threshold_decay)
        self.threshold_decay_interval = max(1, int(threshold_decay_interval))
        self.low_margin_penalty = float(low_margin_penalty)
        self.teacher_clf_weight = float(teacher_clf_weight)
        self.proto_weight = float(proto_weight)
        self.prototype_pooling = str(prototype_pooling).lower()
        self.prototype_pool_temperature = float(prototype_pool_temperature)
        self.confidence_floor = float(confidence_floor)
        self.distribution_alignment = bool(distribution_alignment)
        self.distribution_momentum = float(distribution_momentum)
        self.distribution_min_prob = float(distribution_min_prob)
        self.teacher_temperature = max(1e-4, float(teacher_temperature))
        self.proto_temperature = max(1e-4, float(proto_temperature))
        self.reliability_gate = bool(reliability_gate)
        self.reliability_power = max(0.0, float(reliability_power))
        self.reliability_floor = float(reliability_floor)
        self.target_distribution: Optional[np.ndarray] = None
        self.unlabeled_prior_ema: Optional[torch.Tensor] = None
        self.class_reliability: Dict[int, float] = {}
        self.history: List[Dict] = []
        self.last_outputs: Dict[str, np.ndarray] = {}

    def _get_dynamic_threshold(self, epoch: Optional[int]) -> float:
        if not self.progressive_threshold or epoch is None:
            return self.confidence_threshold
        steps = max(0, int(epoch) // self.threshold_decay_interval)
        decay = steps * self.threshold_decay
        return max(self.min_threshold, self.confidence_threshold - decay)

    def _apply_distribution_alignment(self, probs: torch.Tensor) -> torch.Tensor:
        if not self.distribution_alignment or probs.numel() == 0:
            return probs
        batch_prior = probs.mean(dim=0)
        if self.unlabeled_prior_ema is None or self.unlabeled_prior_ema.numel() != batch_prior.numel():
            self.unlabeled_prior_ema = batch_prior.detach()
        else:
            self.unlabeled_prior_ema = (
                self.distribution_momentum * self.unlabeled_prior_ema
                + (1.0 - self.distribution_momentum) * batch_prior.detach()
            )
        if self.target_distribution is None:
            return probs
        target = torch.as_tensor(self.target_distribution, dtype=probs.dtype, device=probs.device)
        if target.numel() != probs.size(1):
            return probs
        target = target.clamp_min(self.distribution_min_prob)
        model_prior = self.unlabeled_prior_ema.to(probs.device).clamp_min(self.distribution_min_prob)
        aligned = probs * (target / model_prior).unsqueeze(0)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return aligned

    def _apply_reliability_gate(self, conf: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if not self.reliability_gate or conf.numel() == 0:
            return conf
        if not self.class_reliability:
            return conf
        reliability = torch.ones_like(conf)
        for cls, quality in self.class_reliability.items():
            mask = pred == int(cls)
            if mask.any():
                q = max(self.reliability_floor, min(1.0, float(quality)))
                reliability[mask] = q
        if self.reliability_power <= 0.0:
            return conf
        return conf * torch.pow(reliability, self.reliability_power)

    def generate(self,
                 projected_embeddings: np.ndarray,
                 labels: np.ndarray,
                 prototypes: torch.Tensor,
                 epoch: Optional[int] = None,
                 teacher_clf_logits: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        device = prototypes.device
        unlabeled_mask = labels < 0
        N = len(labels)

        all_conf = np.zeros(N, dtype=np.float32)
        all_conf[~unlabeled_mask] = 1.0
        num_classes = prototypes.size(0) if prototypes.dim() >= 2 else 0
        all_probs = np.zeros((N, num_classes), dtype=np.float32)
        if num_classes > 0 and (~unlabeled_mask).any():
            all_probs[np.where(~unlabeled_mask)[0], labels[~unlabeled_mask].astype(np.int64)] = 1.0
        if unlabeled_mask.sum() == 0:
            self.last_outputs = {
                'probabilities': all_probs,
                'confidences': all_conf,
                'predictions': labels.copy(),
            }
            return labels, all_conf

        z_u = torch.from_numpy(projected_embeddings[unlabeled_mask]).float().to(device)
        logits = compute_prototype_logits(
            z_u,
            prototypes,
            temperature=self.pseudo_temperature,
            aggregation=self.prototype_pooling,
            pool_temperature=self.prototype_pool_temperature,
        )
        proto_probs = F.softmax(logits / self.proto_temperature, dim=1)

        if teacher_clf_logits is not None and self.teacher_clf_weight > 0.0:
            clf_logits_u = torch.from_numpy(teacher_clf_logits[unlabeled_mask]).float().to(device)
            clf_probs = F.softmax(clf_logits_u / self.teacher_temperature, dim=1)
            w_clf = self.teacher_clf_weight
            w_proto = self.proto_weight
            total_w = w_clf + w_proto
            probs = (w_clf * clf_probs + w_proto * proto_probs) / max(total_w, 1e-8)
        else:
            probs = proto_probs

        probs = self._apply_distribution_alignment(probs)

        top_k = min(self.top_k, probs.shape[1])
        top_vals, top_idx = probs.topk(top_k, dim=1)
        pred = top_idx[:, 0]
        conf = top_vals[:, 0]
        margin = top_vals[:, 0] - top_vals[:, 1] if top_k > 1 else torch.ones_like(conf)

        base_thr = self._get_dynamic_threshold(epoch)
        class_thr = torch.full_like(conf, fill_value=base_thr)
        class_mar = torch.full_like(conf, fill_value=self.margin_threshold)
        for c, t in self.per_class_thr.items():
            m = (pred == int(c))
            if m.any():
                class_thr[m] = torch.clamp(
                    torch.tensor(t, device=conf.device, dtype=conf.dtype),
                    min=0.0,
                    max=0.999,
                )
        for c, mval in self.per_class_margin.items():
            m = (pred == int(c))
            if m.any():
                class_mar[m] = torch.tensor(mval, device=conf.device, dtype=conf.dtype)

        if self.consistency_check and top_k > 1 and 0.0 < self.low_margin_penalty < 1.0:
            conf = torch.where(margin > class_mar, conf, conf * self.low_margin_penalty)

        conf = self._apply_reliability_gate(conf, pred)

        self.history.append({
            'epoch': int(epoch) if epoch is not None else None,
            'base_threshold': float(base_thr),
            'avg_conf': float(conf.mean().item()),
            'num_pred': int(conf.numel())
        })

        new_labels = labels.copy()
        u_idx = np.where(unlabeled_mask)[0]
        if self.confidence_floor > 0.0:
            conf = torch.where(conf >= self.confidence_floor, conf, torch.zeros_like(conf))
        conf_np = conf.detach().cpu().numpy()
        mar_np = margin.detach().cpu().numpy() if self.top_k > 1 else np.ones_like(conf_np)
        pred_np = pred.detach().cpu().numpy()
        class_thr_np = class_thr.detach().cpu().numpy()
        class_mar_np = class_mar.detach().cpu().numpy()
        for j, (cval, mval, pr) in enumerate(zip(conf_np, mar_np, pred_np)):
            if cval >= class_thr_np[j] and (top_k == 1 or mval >= class_mar_np[j]):
                new_labels[u_idx[j]] = int(pr)
        all_conf[u_idx] = conf_np
        all_probs[u_idx] = probs.detach().cpu().numpy()
        pred_full = labels.copy()
        pred_full[u_idx] = pred_np
        self.last_outputs = {
            'probabilities': all_probs,
            'confidences': all_conf.copy(),
            'predictions': pred_full,
        }
        return new_labels, all_conf

    def generate_pseudo_labels(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def get_last_outputs(self) -> Dict[str, np.ndarray]:
        return self.last_outputs

    def _consistency_based_labeling(
        self,
        similarities: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于Top-K一致性的标签生成
        
        Args:
            similarities: 相似度矩阵 [N, num_classes]
            threshold: 置信度阈值
            
        Returns:
            (pseudo_labels, confidences)
        """
        # 获取Top-K预测
        top_k_values, top_k_indices = similarities.topk(self.top_k, dim=1)
        
        # 检查Top-1和Top-2的差距（margin）
        margin = top_k_values[:, 0] - top_k_values[:, 1]
        
        # 只有当margin足够大时才认为是高置信度
        high_margin_mask = margin > self.margin_threshold
        
        confidences = top_k_values[:, 0]
        pseudo_labels = top_k_indices[:, 0]
        
        # 降低低margin样本的置信度
        penalty = self.low_margin_penalty if 0.0 < self.low_margin_penalty < 1.0 else 1.0
        confidences = torch.where(
            high_margin_mask,
            confidences,
            confidences * penalty
        )
        
        return pseudo_labels, confidences
        
    def _update_history(
        self,
        epoch: Optional[int],
        threshold: float,
        confidences: torch.Tensor
    ):
        """更新历史记录"""
        self.history.append({
            'epoch': epoch,
            'threshold': threshold,
            'num_labeled': (confidences > threshold).sum().item(),
            'avg_confidence': confidences.mean().item(),
            'max_confidence': confidences.max().item(),
            'min_confidence': confidences.min().item()
        })
        
    def get_statistics(self) -> Dict:
        """获取伪标签生成的统计信息"""
        if not self.history:
            return {}
            
        recent = self.history[-1]
        return {
            'total_iterations': len(self.history),
            'latest_num_labeled': recent['num_labeled'],
            'latest_avg_confidence': recent['avg_confidence'],
            'confidence_trend': [h['avg_confidence'] for h in self.history]
        }

class KNNPseudoLabelGenerator:
    """基于KNN的伪标签生成器"""
    
    def __init__(
        self,
        n_neighbors: int = 10,
        confidence_threshold: float = 0.6,
        weight_mode: str = 'distance'  # 'uniform' or 'distance'
    ):
        """
        Args:
            n_neighbors: KNN的邻居数量
            confidence_threshold: 置信度阈值
            weight_mode: 权重模式（uniform或distance）
        """
        self.n_neighbors = n_neighbors
        self.confidence_threshold = confidence_threshold
        self.weight_mode = weight_mode
        
    def generate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        使用KNN投票生成伪标签
        
        Args:
            embeddings: 特征embeddings
            labels: 当前标签
            
        Returns:
            (new_labels, confidences): 更新后的标签和置信度字典
        """
        from sklearn.neighbors import KNeighborsClassifier
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        unlabeled_indices = np.where(labels < 0)[0]
        labeled_indices = np.where(labels >= 0)[0]
        
        if len(labeled_indices) == 0:
            logger.warning("没有标注样本，无法使用KNN生成伪标签")
            return labels, {}
            
        # 训练KNN分类器
        knn = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(labeled_indices)),
            weights=self.weight_mode
        )
        
        labeled_embeddings = embeddings[labeled_indices]
        labeled_y = labels[labeled_indices]
        knn.fit(labeled_embeddings, labeled_y)
        
        # 预测无标签样本
        unlabeled_embeddings = embeddings[unlabeled_indices]
        probas = knn.predict_proba(unlabeled_embeddings)
        predictions = knn.predict(unlabeled_embeddings)
        
        # 使用最高概率作为置信度
        max_probas = probas.max(axis=1)
        
        # 更新标签
        new_labels = labels.copy()
        confidences = {}
        
        num_generated = 0
        for idx, (pred, conf) in enumerate(zip(predictions, max_probas)):
            if conf >= self.confidence_threshold:
                original_idx = unlabeled_indices[idx]
                new_labels[original_idx] = pred
                confidences[original_idx] = float(conf)
                num_generated += 1
                
        logger.info(
            f"KNN生成 {num_generated} 个伪标签 "
            f"(阈值={self.confidence_threshold:.2f})"
        )
        
        return new_labels, confidences


class EnsemblePseudoLabelGenerator:
    """集成多种策略的伪标签生成器"""
    
    def __init__(
        self,
        generators: List[PseudoLabelGenerator],
        voting_strategy: str = 'soft',  # 'hard' or 'soft'
        agreement_threshold: float = 0.7
    ):
        """
        Args:
            generators: 伪标签生成器列表
            voting_strategy: 投票策略（硬投票或软投票）
            agreement_threshold: 一致性阈值
        """
        self.generators = generators
        self.voting_strategy = voting_strategy
        self.agreement_threshold = agreement_threshold
        
    def generate(
        self,
        projected_embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_predictions = []
        all_confidences = []
        for generator in self.generators:
            pred_labels, confs = generator.generate(
                projected_embeddings, labels, prototypes, epoch=epoch
            )
            all_predictions.append(pred_labels)
            all_confidences.append(confs)
            
        all_predictions = np.array(all_predictions)  # [num_generators, N]
        all_confidences = np.array(all_confidences)  # [num_generators, N]
        
        # 执行投票
        if self.voting_strategy == 'hard':
            final_labels, final_confidences = self._hard_voting(
                all_predictions, all_confidences, labels
            )
        else:
            final_labels, final_confidences = self._soft_voting(
                all_predictions, all_confidences, labels
            )
            
        return final_labels, final_confidences
        
    def _hard_voting(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        original_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """硬投票：少数服从多数"""
        from scipy.stats import mode
        
        final_labels = original_labels.copy()
        final_confidences = np.zeros(len(original_labels))
        
        unlabeled_mask = original_labels < 0
        
        for idx in np.where(unlabeled_mask)[0]:
            # 获取所有生成器对该样本的预测
            sample_preds = predictions[:, idx]
            sample_confs = confidences[:, idx]
            
            # 只考虑有效预测（>=0）
            valid_mask = sample_preds >= 0
            if valid_mask.sum() == 0:
                continue
                
            valid_preds = sample_preds[valid_mask]
            valid_confs = sample_confs[valid_mask]
            
            # 多数投票
            mode_result = mode(valid_preds, keepdims=True)
            majority_label = mode_result.mode[0]
            vote_count = mode_result.count[0]
            
            # 检查一致性
            agreement = vote_count / len(valid_preds)
            
            if agreement >= self.agreement_threshold:
                final_labels[idx] = majority_label
                # 置信度为投票该标签的生成器的平均置信度
                label_mask = valid_preds == majority_label
                final_confidences[idx] = valid_confs[label_mask].mean()
                
        return final_labels, final_confidences
        
    def _soft_voting(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        original_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """软投票：加权平均置信度"""
        final_labels = original_labels.copy()
        final_confidences = np.zeros(len(original_labels))
        
        unlabeled_mask = original_labels < 0
        
        for idx in np.where(unlabeled_mask)[0]:
            sample_preds = predictions[:, idx]
            sample_confs = confidences[:, idx]
            
            valid_mask = sample_preds >= 0
            if valid_mask.sum() == 0:
                continue
                
            valid_preds = sample_preds[valid_mask]
            valid_confs = sample_confs[valid_mask]
            
            # 统计每个类别的加权得分
            class_scores = defaultdict(float)
            for pred, conf in zip(valid_preds, valid_confs):
                class_scores[pred] += conf
                
            # 选择得分最高的类别
            best_class = max(class_scores, key=class_scores.get)
            total_score = sum(class_scores.values())
            best_score = class_scores[best_class] / total_score
            
            if best_score >= self.agreement_threshold:
                final_labels[idx] = best_class
                final_confidences[idx] = best_score
                
        return final_labels, final_confidences


class PseudoLabelManager:
    """伪标签管理器 - 统一管理伪标签生成流程"""
    
    def __init__(
        self,
        generator: PseudoLabelGenerator,
        update_interval: int = 5,
        warmup_epochs: int = 10
    ):
        """
        Args:
            generator: 伪标签生成器
            update_interval: 更新间隔（每N个epoch更新一次）
            warmup_epochs: 预热轮次（前N个epoch不生成伪标签）
        """
        self.generator = generator
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.pseudo_labels_dict: Optional[Dict] = None
        
    def should_update(self, epoch: int) -> bool:
        """判断是否应该更新伪标签"""
        if epoch < self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.update_interval == 0
        
    def update(
        self,
        projected_embeddings: np.ndarray,
        current_labels: np.ndarray,
        prototypes: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Dict:
        new_labels, confidences = self.generator.generate(
            projected_embeddings, current_labels, prototypes, epoch=epoch
        )
        self.pseudo_labels_dict = {'labels': new_labels, 'confidences': confidences}
        return self.pseudo_labels_dict
        
    def get_pseudo_labels(self) -> Optional[Dict]:
        """获取当前的伪标签字典"""
        return self.pseudo_labels_dict
        
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.pseudo_labels_dict is None:
            return {}
            
        labels = self.pseudo_labels_dict['labels']
        confidences = self.pseudo_labels_dict['confidences']
        
        labeled_mask = labels >= 0
        
        stats = {
            'total_samples': len(labels),
            'labeled_samples': labeled_mask.sum(),
            'labeling_rate': labeled_mask.sum() / len(labels),
            'avg_confidence': confidences[labeled_mask].mean() if labeled_mask.sum() > 0 else 0.0,
            'label_distribution': dict(zip(*np.unique(labels[labeled_mask], return_counts=True)))
        }
        
        return stats

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def graph_label_propagation(
    projected_embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    alpha: float = 0.9,
    iters: int = 20,
    min_support: float = 0.0,
    conf_power: float = 0.75,
    min_purity: float = 0.0,
    seed_probabilities: np.ndarray = None,
    graph_temperature: float = 0.20,
    mutual_knn: bool = True,
    seed_weight: float = 0.35,
    entropy_weight: float = 0.20,
    neighbor_agreement_weight: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft-seeded LP with temperature-scaled kNN graph.
    - labeled nodes use one-hot seeds
    - unlabeled nodes can inject base/teacher probabilities as weak seeds
    - confidence is calibrated by purity + neighborhood agreement + low entropy
    """
    z = torch.from_numpy(projected_embeddings).float()
    z = F.normalize(z, dim=1)
    N, _ = z.shape
    unlabeled_mask_np = labels < 0

    if N <= 1:
        return np.full(N, -1, dtype=np.int64), np.zeros(N, dtype=np.float32)

    sim = torch.mm(z, z.t())
    sim.fill_diagonal_(-1.0)
    k = max(1, min(int(k), max(1, N - 1)))
    topk_vals, topk_idx = sim.topk(k, dim=1)
    temp = max(float(graph_temperature), 1e-4)
    topk_w = torch.softmax(topk_vals / temp, dim=1)

    adj = torch.zeros_like(sim)
    adj.scatter_(1, topk_idx, topk_w)
    if mutual_knn:
        adj = 0.5 * (adj + adj.t())
    else:
        adj = torch.maximum(adj, adj.t())
    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
    S = adj / row_sum

    C = int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
    if C == 0:
        return np.full(N, -1, dtype=np.int64), np.zeros(N, dtype=np.float32)

    Y0 = torch.zeros(N, C, dtype=torch.float32)
    labeled_idx = np.where(labels >= 0)[0]
    if len(labeled_idx) > 0:
        Y0[labeled_idx, labels[labeled_idx]] = 1.0

    if seed_probabilities is not None:
        seed_probs = torch.as_tensor(seed_probabilities, dtype=torch.float32)
        if seed_probs.ndim == 2 and seed_probs.shape[0] == N and seed_probs.shape[1] == C:
            seed_probs = seed_probs / seed_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weak_seed = max(0.0, min(1.0, float(seed_weight)))
            if weak_seed > 0.0:
                Y0[unlabeled_mask_np] = weak_seed * seed_probs[unlabeled_mask_np]

    Y = Y0.clone()
    labeled_mask_t = torch.from_numpy(labels >= 0)
    for _ in range(iters):
        Y = alpha * (S @ Y) + (1.0 - alpha) * Y0
        if labeled_mask_t.any():
            Y[labeled_mask_t] = Y0[labeled_mask_t]

    probs = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    purity_conf, pred = probs.max(dim=1)
    neighbor_class_probs = probs[:, pred].transpose(0, 1)
    neighbor_support = torch.sum(S * neighbor_class_probs, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1) / np.log(max(C, 2))
    entropy_conf = 1.0 - entropy

    purity_w = max(0.0, 1.0 - float(neighbor_agreement_weight) - float(entropy_weight))
    conf = (
        purity_w * purity_conf
        + float(neighbor_agreement_weight) * neighbor_support
        + float(entropy_weight) * entropy_conf
    )
    conf = torch.pow(conf.clamp(0.0, 1.0), max(float(conf_power), 1e-4))

    unlabeled_mask = torch.from_numpy(unlabeled_mask_np)
    if min_support > 0:
        low_support = unlabeled_mask & (neighbor_support < float(min_support))
        conf = conf.masked_fill(low_support, 0.0)
        pred = pred.masked_fill(low_support, -1)
    if min_purity > 0:
        low_purity = unlabeled_mask & (purity_conf < float(min_purity))
        conf = conf.masked_fill(low_purity, 0.0)
        pred = pred.masked_fill(low_purity, -1)

    return pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)

def apply_pseudo_label_acceptance_cap(
    labels: np.ndarray,
    confidences: np.ndarray,
    observed_labels: np.ndarray,
    max_rate: float = 1.0,
    max_count: int = 0,
    class_balance: bool = False,
    class_balance_power: float = 0.5,
    min_per_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Cap pseudo-label adoption to avoid late-stage confirmation-bias cascades."""
    capped_labels = labels.copy()
    capped_conf = confidences.copy()

    unlabeled_mask = observed_labels < 0
    accepted_idx = np.where((capped_labels >= 0) & unlabeled_mask)[0]
    before = int(len(accepted_idx))
    total_unlabeled = int(unlabeled_mask.sum())

    if before == 0:
        return capped_labels, capped_conf, {
            'before': 0,
            'after': 0,
            'dropped': 0,
            'cap': 0,
            'total_unlabeled': total_unlabeled,
        }

    cap = before
    if max_rate is not None and 0.0 < float(max_rate) < 1.0:
        cap = min(cap, max(1, int(total_unlabeled * float(max_rate))))
    if max_count is not None and int(max_count) > 0:
        cap = min(cap, int(max_count))

    if cap < before:
        if class_balance:
            observed_known = observed_labels[observed_labels >= 0]
            obs_counts = np.bincount(observed_known.astype(np.int64)) if len(observed_known) > 0 else np.array([], dtype=np.int64)
            keep_idx = []
            remaining_slots = cap
            by_class = {}
            class_weights = {}
            for c in np.unique(capped_labels[accepted_idx]):
                cls_idx = accepted_idx[capped_labels[accepted_idx] == c]
                cls_idx = cls_idx[np.argsort(-capped_conf[cls_idx])]
                by_class[int(c)] = cls_idx
                obs_count = int(obs_counts[int(c)]) if int(c) < len(obs_counts) else 0
                class_weights[int(c)] = 1.0 / max(obs_count, 1) ** float(class_balance_power)
            total_weight = sum(class_weights.values()) if class_weights else 0.0
            quotas = {}
            if total_weight > 0.0:
                for c, cls_idx in by_class.items():
                    target = int(round(cap * class_weights[c] / total_weight))
                    if min_per_class > 0:
                        target = max(min_per_class, target)
                    quotas[c] = min(len(cls_idx), target)
            else:
                quotas = {c: min(len(cls_idx), min_per_class if min_per_class > 0 else len(cls_idx)) for c, cls_idx in by_class.items()}
            for c, cls_idx in by_class.items():
                take = min(len(cls_idx), quotas.get(c, 0), remaining_slots)
                if take > 0:
                    keep_idx.extend(cls_idx[:take].tolist())
                    remaining_slots -= take
            if remaining_slots > 0:
                residual = np.setdiff1d(accepted_idx, np.array(keep_idx, dtype=np.int64), assume_unique=False)
                residual = residual[np.argsort(-capped_conf[residual])]
                keep_idx.extend(residual[:remaining_slots].tolist())
            keep_idx = np.array(sorted(set(keep_idx)), dtype=np.int64)
            drop_idx = np.setdiff1d(accepted_idx, keep_idx, assume_unique=False)
        else:
            order = accepted_idx[np.argsort(-capped_conf[accepted_idx])]
            drop_idx = order[cap:]
        capped_labels[drop_idx] = -1
        capped_conf[drop_idx] = 0.0

    after = int(((capped_labels >= 0) & unlabeled_mask).sum())
    return capped_labels, capped_conf, {
        'before': before,
        'after': after,
        'dropped': before - after,
        'cap': int(cap),
        'total_unlabeled': total_unlabeled,
    }


def class_balanced_pseudo_cap(
    labels: np.ndarray,
    confidences: np.ndarray,
    observed_labels: np.ndarray,
    quota_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """对每个类别独立按置信度排序，各取 top-quota 个高置信样本。
    quota_per_class <= 0 时直接返回原始结果（不限制）。
    """
    if quota_per_class <= 0:
        return labels.copy(), confidences.copy()

    capped_labels = labels.copy()
    capped_conf = confidences.copy()
    unlabeled_mask = observed_labels < 0

    classes = np.unique(labels[(labels >= 0) & unlabeled_mask])
    for c in classes:
        cls_mask = (labels == int(c)) & unlabeled_mask
        cls_idx = np.where(cls_mask)[0]
        if len(cls_idx) <= quota_per_class:
            continue
        # 按置信度降序，保留 top-quota，其余 abstain
        order = cls_idx[np.argsort(-confidences[cls_idx])]
        drop_idx = order[quota_per_class:]
        capped_labels[drop_idx] = -1
        capped_conf[drop_idx] = 0.0

    return capped_labels, capped_conf



def fuse_pseudo_labels(
    base_labels: np.ndarray,
    base_conf: np.ndarray,
    glp_labels: np.ndarray,
    glp_conf: np.ndarray,
    observed_labels: np.ndarray,
    thr: float = 0.6,
    lp_thr: Optional[float] = None,
    conflict_thr: Optional[float] = None,
    conflict_margin: float = 0.10,
    allow_lp_only: bool = True,
    lp_agree_bonus: float = 0.0,
    lp_agree_threshold_offset: float = 0.03,
    return_sources: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    融合高级伪标签生成器与图传播标签：
    - agreement 时允许略低于 base 主阈值，给图一致性一个可验证的入口。
    - LP-only 分支默认更严格，必要时可直接关闭。
    - 双方冲突时默认 abstain，只有一方显著更强才放行。
    """
    agree_thr = max(0.50, float(thr) - max(0.0, float(lp_agree_threshold_offset)))
    agree_min_side = max(0.45, agree_thr - 0.08)
    base_thr = float(thr)
    lp_thr = max(float(lp_thr) if lp_thr is not None else 0.0, min(0.99, float(thr) + 0.08))
    conflict_thr = max(float(conflict_thr) if conflict_thr is not None else 0.0, min(0.99, float(thr) + 0.15))
    conflict_margin = max(0.10, float(conflict_margin))

    N = len(base_labels)
    fused_labels = -1 * np.ones(N, dtype=np.int64)
    fused_conf = np.zeros(N, dtype=np.float32)
    fused_sources = np.full(N, PSEUDO_SOURCE_ABSTAIN, dtype=np.int64)

    labeled_mask = observed_labels >= 0
    fused_labels[labeled_mask] = observed_labels[labeled_mask]
    fused_conf[labeled_mask] = 1.0
    fused_sources[labeled_mask] = PSEUDO_SOURCE_OBSERVED

    U = np.where(~labeled_mask)[0]
    for i in U:
        bl, bc = int(base_labels[i]), float(base_conf[i])
        gl, gc = int(glp_labels[i]), float(glp_conf[i])

        if bl >= 0 and gl >= 0 and bl == gl:
            agree_conf = min(0.999, max(bc, gc) + max(0.0, float(lp_agree_bonus)))
            if agree_conf >= agree_thr and min(bc, gc) >= agree_min_side:
                fused_labels[i] = bl
                fused_conf[i] = agree_conf
                fused_sources[i] = PSEUDO_SOURCE_AGREE
                continue

        if bl >= 0 and gl < 0 and bc >= base_thr:
            fused_labels[i] = bl
            fused_conf[i] = bc
            fused_sources[i] = PSEUDO_SOURCE_BASE_ONLY
            continue

        if allow_lp_only and gl >= 0 and bl < 0 and gc >= lp_thr:
            fused_labels[i] = gl
            fused_conf[i] = gc
            fused_sources[i] = PSEUDO_SOURCE_LP_ONLY
            continue

        if bl >= 0 and gl >= 0 and bl != gl:
            if bc >= conflict_thr and (bc - gc) >= conflict_margin:
                fused_labels[i] = bl
                fused_conf[i] = bc
                fused_sources[i] = PSEUDO_SOURCE_BASE_CONFLICT
            elif allow_lp_only and gc >= max(conflict_thr, lp_thr) and (gc - bc) >= conflict_margin:
                fused_labels[i] = gl
                fused_conf[i] = gc
                fused_sources[i] = PSEUDO_SOURCE_LP_CONFLICT
            else:
                fused_labels[i] = -1
                fused_conf[i] = 0.0
                fused_sources[i] = PSEUDO_SOURCE_ABSTAIN

    if return_sources:
        return fused_labels, fused_conf, fused_sources
    return fused_labels, fused_conf
