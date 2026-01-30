import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """基础伪标签生成器 - 高置信度策略"""

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

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

        if unlabeled_mask.sum() == 0:
            return labels, all_confidences

        z_u = torch.from_numpy(projected_embeddings[unlabeled_mask]).float().to(device)
        z_u = F.normalize(z_u, dim=1)
        protos = F.normalize(prototypes, dim=1)

        sims = z_u @ protos.t()  # 余弦相似度
        confs, preds = sims.max(dim=1)

        high_mask = confs > self.confidence_threshold
        new_labels = labels.copy()
        u_idx = np.where(unlabeled_mask)[0]
        high_idx = u_idx[high_mask.cpu().numpy()]

        new_labels[high_idx] = preds[high_mask].cpu().numpy()
        all_confidences[u_idx] = confs.cpu().numpy()

        return new_labels, all_confidences

    def generate_pseudo_labels(self, *args, **kwargs):
        return self.generate(*args, **kwargs)



class AdvancedPseudoLabelGenerator:
    """伪标签生成器：支持 per-class 阈值/ margin，一致性检查，可渐进阈值"""
    def __init__(self,
                 confidence_threshold: float = 0.85,
                 progressive_threshold: bool = True,
                 consistency_check: bool = True,
                 top_k_consistency: int = 3,
                 margin_threshold: float = 0.1,
                 per_class_thresholds: Optional[Dict[int, float]] = None,
                 per_class_margin: Optional[Dict[int, float]] = None):
        self.confidence_threshold = confidence_threshold
        self.progressive_threshold = progressive_threshold
        self.consistency_check = consistency_check
        self.top_k = top_k_consistency
        self.margin_threshold = margin_threshold
        self.per_class_thr = per_class_thresholds or {}
        self.per_class_margin = per_class_margin or {}
        self.history: List[Dict] = []

    def _get_dynamic_threshold(self, epoch: Optional[int]) -> float:
        if not self.progressive_threshold or epoch is None:
            return self.confidence_threshold
        decay = (epoch // 10) * 0.05
        return max(0.5, self.confidence_threshold - decay)

    def generate(self,
                 projected_embeddings: np.ndarray,
                 labels: np.ndarray,
                 prototypes: torch.Tensor,
                 epoch: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        device = prototypes.device
        unlabeled_mask = labels < 0
        N = len(labels)

        all_conf = np.zeros(N, dtype=np.float32)
        all_conf[~unlabeled_mask] = 1.0
        if unlabeled_mask.sum() == 0:
            return labels, all_conf

        z_u = torch.from_numpy(projected_embeddings[unlabeled_mask]).float().to(device)
        z_u = F.normalize(z_u, dim=1)
        protos = F.normalize(prototypes, dim=1)
        sims = z_u @ protos.t()  # [Nu, C]

        # Top-K / margin
        top_vals, top_idx = sims.topk(self.top_k, dim=1)
        pred = top_idx[:, 0]
        conf = top_vals[:, 0]
        margin = top_vals[:, 0] - top_vals[:, 1] if self.top_k > 1 else torch.ones_like(conf)

        # per-class 阈值/ margin 应用
        base_thr = self._get_dynamic_threshold(epoch)
        class_thr = torch.full_like(conf, fill_value=base_thr)
        class_mar = torch.full_like(conf, fill_value=self.margin_threshold)
        for c, t in self.per_class_thr.items():
            m = (pred == int(c))
            if m.any():
                class_thr[m] = torch.clamp(torch.tensor(t, device=conf.device), min=0.0, max=0.999)
        for c, mval in self.per_class_margin.items():
            m = (pred == int(c))
            if m.any():
                class_mar[m] = torch.tensor(mval, device=conf.device)

        # 一致性检查：低 margin 降权
        if self.consistency_check and self.top_k > 1:
            conf = torch.where(margin > class_mar, conf, conf * 0.8)

        # 统计
        self.history.append({
            'epoch': int(epoch) if epoch is not None else None,
            'base_threshold': float(base_thr),
            'avg_conf': float(conf.mean().item()),
            'num_pred': int(conf.numel())
        })

        new_labels = labels.copy()
        u_idx = np.where(unlabeled_mask)[0]
        conf_np = conf.detach().cpu().numpy()
        mar_np = margin.detach().cpu().numpy() if self.top_k > 1 else np.ones_like(conf_np)
        # 逐样本采纳（需同时满足阈值和 margin）
        for j, (cval, mval, pr) in enumerate(zip(conf_np, mar_np, pred.detach().cpu().numpy())):
            if cval >= class_thr[j].item() and (self.top_k == 1 or mval >= class_mar[j].item()):
                new_labels[u_idx[j]] = int(pr)
        all_conf[u_idx] = conf_np
        return new_labels, all_conf

    def generate_pseudo_labels(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
        
    def _get_dynamic_threshold(self, epoch: Optional[int]) -> float:
        """获取动态阈值"""
        if not self.progressive_threshold or epoch is None:
            return self.confidence_threshold
            
        # 从高阈值逐渐降低（每10个epoch降低0.05）
        decay = (epoch // 10) * 0.05
        current_threshold = max(0.5, self.confidence_threshold - decay)
        
        return current_threshold
        
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
        confidences = torch.where(
            high_margin_mask,
            confidences,
            confidences * 0.8
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
    iters: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单 LP：Y <- alpha * S * Y + (1-alpha) * Y0
    - projected_embeddings: z_all (N,D)，已是 projector 输出（未必归一化，此处内部归一化）
    - labels: 观测标签（含 -1 表示未标注）
    返回：(pred_labels, confidences)
    """
    z = torch.from_numpy(projected_embeddings).float()
    z = F.normalize(z, dim=1)       # 归一化
    N, D = z.shape

    # 余弦相似构图，取 kNN
    sim = torch.mm(z, z.t())        # [N,N]
    sim.fill_diagonal_(0.0)

    # kNN 稀疏邻接
    topk = sim.topk(k, dim=1).indices     # [N,k]
    adj = torch.zeros_like(sim)
    adj.scatter_(1, topk, sim.gather(1, topk))
    adj = torch.max(adj, adj.t())

    # 行归一化 S
    row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
    S = adj / row_sum

    # Y0 (one-hot for labeled)
    C = int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
    if C == 0:
        # 无标签则返回全 -1
        return np.full(N, -1, dtype=np.int64), np.zeros(N, dtype=np.float32)

    Y0 = torch.zeros(N, C)
    labeled_idx = np.where(labels >= 0)[0]
    if len(labeled_idx) > 0:
        Y0[labeled_idx, labels[labeled_idx]] = 1.0

    # 迭代传播
    Y = Y0.clone()
    for _ in range(iters):
        Y = alpha * (S @ Y) + (1 - alpha) * Y0

    conf, pred = Y.max(dim=1)
    return pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)


def fuse_pseudo_labels(
    base_labels: np.ndarray,
    base_conf: np.ndarray,
    glp_labels: np.ndarray,
    glp_conf: np.ndarray,
    observed_labels: np.ndarray,
    thr: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    融合高级伪标签生成器与图传播标签：
    - 若两者一致且 max(conf) >= thr，则采纳；conf 取最大或平均都可（此处取最大）。
    - 若不一致，选择置信度更高且 >= thr 的那个；否则保持 -1。
    - 对已有真实标签的样本，保持真实标签，置信度=1.0。
    """
    N = len(base_labels)
    fused_labels = -1 * np.ones(N, dtype=np.int64)
    fused_conf = np.zeros(N, dtype=np.float32)

    # 先保留真实标签
    labeled_mask = observed_labels >= 0
    fused_labels[labeled_mask] = observed_labels[labeled_mask]
    fused_conf[labeled_mask] = 1.0

    # 仅在未标注位置融合
    U = np.where(~labeled_mask)[0]
    for i in U:
        bl, bc = base_labels[i], float(base_conf[i])
        gl, gc = glp_labels[i], float(glp_conf[i])

        # 都预测了同一类
        if bl >= 0 and gl >= 0 and bl == gl and max(bc, gc) >= thr:
            fused_labels[i] = bl
            fused_conf[i] = max(bc, gc)
        else:
            # 选择更高置信度的那个（需 >= thr）
            if bl >= 0 and bc >= thr and (bc >= gc):
                fused_labels[i] = bl
                fused_conf[i] = bc
            elif gl >= 0 and gc >= thr:
                fused_labels[i] = gl
                fused_conf[i] = gc
            else:
                fused_labels[i] = -1
                fused_conf[i] = 0.0

    return fused_labels, fused_conf
