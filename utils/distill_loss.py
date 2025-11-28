import torch
from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, alpha=0.7, temperature=4):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # 知识蒸馏损失（软标签）
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
        ) * (self.temperature**2)

        # 交叉熵损失（硬标签）
        hard_loss = self.ce_loss(student_logits, targets)

        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class ProgressiveDistillation(nn.Module):
    """渐进式蒸馏 - 逐步增加蒸馏强度"""

    def __init__(self, initial_alpha=0.1, final_alpha=0.7, initial_temperature=4, final_temperature=8):
        super(ProgressiveDistillation, self).__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets, epoch, total_epochs):
        # 渐进调整alpha
        progress = epoch / total_epochs
        alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        temperature = self.initial_temperature + (self.final_temperature - self.initial_temperature) * progress

        # 蒸馏损失
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
        ) * (temperature**2)

        # 交叉熵损失
        hard_loss = self.ce_loss(student_logits, targets)

        return alpha * soft_loss + (1 - alpha) * hard_loss


class FeatureAdaptiveDistillation(nn.Module):
    """特征适配蒸馏 - 将教师特征适配到学生空间"""

    def __init__(self, teacher_dim, student_dim, temperature=4):
        super(FeatureAdaptiveDistillation, self).__init__()
        self.temperature = temperature

        # 适配器：将学生特征映射到教师特征空间
        self.adapter = nn.Sequential(
            nn.Linear(student_dim, teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(teacher_dim // 2, teacher_dim),
        )

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits,
        teacher_logits,
        student_feat,
        teacher_feat,
        targets,
        alpha=0.5,
    ):
        # 特征适配蒸馏
        adapted_feat = self.adapter(student_feat)
        feature_loss = F.mse_loss(adapted_feat, teacher_feat)

        # 输出蒸馏
        output_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
        ) * (self.temperature**2)

        # 交叉熵损失
        ce_loss = self.ce_loss(student_logits, targets)

        return alpha * output_loss + (1 - alpha) * ce_loss + 0.1 * feature_loss


class SelectiveDistillation(nn.Module):
    """选择性蒸馏 - 只蒸馏学生能理解的知识"""

    def __init__(self, temperature=4, confidence_threshold=0.7):
        super(SelectiveDistillation, self).__init__()
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        batch_size = student_logits.size(0)

        # 计算教师置信度
        teacher_probs = F.softmax(teacher_logits, dim=1)
        teacher_conf, _ = torch.max(teacher_probs, dim=1)

        # 选择高置信度样本进行蒸馏
        mask = teacher_conf > self.confidence_threshold
        valid_count = mask.sum().item()

        if valid_count == 0:
            # 没有高置信度样本，只使用交叉熵
            return self.ce_loss(student_logits, targets)

        # 蒸馏损失（只对高置信度样本）
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits[mask] / self.temperature, dim=1),
            F.softmax(teacher_logits[mask] / self.temperature, dim=1),
        ).mean() * (self.temperature**2)

        # 交叉熵损失（所有样本）
        hard_loss = self.ce_loss(student_logits, targets)

        # 动态调整权重
        alpha = valid_count / batch_size

        return alpha * soft_loss + (1 - alpha) * hard_loss
