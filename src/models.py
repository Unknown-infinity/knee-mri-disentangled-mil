"""
Model Definitions
==================
Contains:
  - GatedAttentionMIL         : V1 single-head MIL (baseline)
  - MultiPlaneCrossAttentionNet: V1 baseline model
  - GeMPooling                : Generalized Mean Pooling (V2 improvement)
  - TaskSpecificGatedMIL      : V2 task-disentangled MIL
  - MultiPlaneDisentangledNet : V2 proposed model

References:
  - Ilse et al. (2018) "Attention-based Deep Multiple Instance Learning"
  - Radenović et al. (2019) "Fine-tuning CNN Image Retrieval with No Human Annotation"
  - Ridnik et al. (2021) "Asymmetric Loss for Multi-Label Classification"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# V1 BASELINE COMPONENTS
# =============================================================================

class GatedAttentionMIL(nn.Module):
    """
    Single-head gated attention MIL (Ilse et al. 2018).
    Produces one attention distribution for all tasks — a known limitation
    addressed in the V2 TaskSpecificGatedMIL.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.V = nn.Linear(in_features, hidden_dim)
        self.U = nn.Linear(in_features, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (S, D) slice embeddings
        Returns:
            bag_embedding: (1, D)
            weights: (S,)
        """
        a_v    = torch.tanh(self.V(h))
        a_u    = torch.sigmoid(self.U(h))
        scores = self.w(a_v * a_u)                        # (S, 1)
        weights = torch.softmax(scores, dim=0)             # (S, 1)
        bag_embedding = torch.sum(weights * h, dim=0, keepdim=True)  # (1, D)
        return bag_embedding, weights.squeeze(-1)


class MultiPlaneCrossAttentionNet(nn.Module):
    """
    V1 Baseline: ConvNeXt-Tiny + single-head MIL + cross-plane transformer.
    Uses AdaptiveAvgPool2d and a shared classification head.
    """

    def __init__(self, num_classes: int = 3, embed_dim: int = 768):
        super().__init__()
        try:
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            print("[INFO] Loaded ImageNet-pretrained ConvNeXt-Tiny weights.")
        except Exception as e:
            print(f"[WARNING] Pretrained weight download failed ({e}). Using random init.")
            backbone = models.convnext_tiny(weights=None)

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mil_sag = GatedAttentionMIL(embed_dim)
        self.mil_cor = GatedAttentionMIL(embed_dim)
        self.mil_axi = GatedAttentionMIL(embed_dim)

        self.plane_pos = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, dropout=0.2, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim * 3, num_classes)

    def _encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(0)                                           # (S, 3, 256, 256)
        return self.pool(self.feature_extractor(x)).flatten(1)     # (S, 768)

    def forward(self, sag, cor, axi):
        h_sag = self._encode_slices(sag)
        h_cor = self._encode_slices(cor)
        h_axi = self._encode_slices(axi)

        e_sag, w_sag = self.mil_sag(h_sag)
        e_cor, w_cor = self.mil_cor(h_cor)
        e_axi, w_axi = self.mil_axi(h_axi)

        tokens = torch.stack([e_sag, e_cor, e_axi], dim=1) + self.plane_pos  # (1, 3, 768)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        logits = self.classifier(tokens.flatten(1))                # (1, 3)
        return logits, (w_sag, w_cor, w_axi)


# =============================================================================
# V2 PROPOSED COMPONENTS
# =============================================================================

class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling (Radenović et al. 2019).

    Preserves localized focal lesion signals (e.g., meniscal hyper-intensities)
    better than average pooling by emphasising high-activation spatial regions.

    With p=1  → Average pooling
    With p→∞  → Max pooling (approximately)
    With p=3  → Intermediate emphasis on activated regions (default)

    p is a learnable parameter, so the model can adapt it during training.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class TaskSpecificGatedMIL(nn.Module):
    """
    Task-Disentangled Gated Attention MIL.

    Key improvement over single-head MIL: computes num_tasks independent
    attention distributions over slices. This resolves anatomical slice
    competition between mid-sagittal ACL tears and peripheral meniscal tears.

    For a scan with both ACL and meniscus pathology:
      - ACL head attends to mid-sagittal slices (intercondylar notch)
      - Meniscus head attends to peripheral slices (medial/lateral horns)
    simultaneously and without mutual suppression.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, num_tasks: int = 3):
        super().__init__()
        self.V       = nn.Linear(in_features, hidden_dim)
        self.U       = nn.Linear(in_features, hidden_dim)
        self.w       = nn.Linear(hidden_dim, num_tasks)
        self.dropout = nn.Dropout(0.25)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (S, D) slice embeddings
        Returns:
            bag_embeddings: (num_tasks, D) — one task-specific summary per task
            weights:        (S, num_tasks) — attention distribution per task
        """
        a_v    = torch.tanh(self.V(h))
        a_u    = torch.sigmoid(self.U(h))
        scores = self.w(self.dropout(a_v * a_u))   # (S, num_tasks)
        weights = torch.softmax(scores, dim=0)      # (S, num_tasks)
        bag_embeddings = torch.matmul(weights.T, h) # (num_tasks, D)
        return bag_embeddings, weights


class MultiPlaneDisentangledNet(nn.Module):
    """
    V2 Proposed Model: End-to-End Multi-Plane Disentangled Attention MIL.

    Architecture:
      1. Shared ConvNeXt-Tiny backbone + learnable GeM pooling
         → per-slice embeddings (S_p, 768) for each plane p
      2. Plane-specific LayerNorm for multi-contrast harmonisation
      3. Task-Disentangled Gated MIL per plane
         → (3_tasks, 768) plane-specific task embeddings
      4. Stack to (3_tasks, 3_planes, 768) + positional encoding
      5. Per-task Cross-Plane Transformer (8-head MHSA + FFN)
      6. 3 separate task heads: Dropout(0.3) → Linear(768*3, 1)

    Key design decisions:
      - Single unified model vs. 4-network Kasuya pipeline
      - No generative image synthesis / no hardcoded thresholds
      - Task-disentangled attention resolves ACL vs. meniscus slice competition
      - GeM pooling preserves focal lesion signals vs. average pooling dilution
    """

    def __init__(self, num_tasks: int = 3, embed_dim: int = 768):
        super().__init__()
        # Backbone — use weights=None when loading from local .pth checkpoint
        # (avoids requiring internet access at inference time)
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.gem_pool = GeMPooling(p=3.0)

        # Plane-specific adapter norms — handle FST2, T1, PD contrast differences
        self.norm_sag = nn.LayerNorm(embed_dim)
        self.norm_cor = nn.LayerNorm(embed_dim)
        self.norm_axi = nn.LayerNorm(embed_dim)

        # Task-disentangled MIL — one per anatomical plane
        self.mil_sag = TaskSpecificGatedMIL(embed_dim, num_tasks=num_tasks)
        self.mil_cor = TaskSpecificGatedMIL(embed_dim, num_tasks=num_tasks)
        self.mil_axi = TaskSpecificGatedMIL(embed_dim, num_tasks=num_tasks)

        # Learnable plane positional encoding
        self.plane_pos = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        # Cross-Plane Transformer block (applied per task)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, dropout=0.25, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Dedicated per-task classification heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(embed_dim * 3, 1)
            ) for _ in range(num_tasks)
        ])

    def _encode_slices(self, x: torch.Tensor, plane_norm: nn.LayerNorm) -> torch.Tensor:
        x = x.squeeze(0)                                           # (S, 3, 224, 224)
        feats = self.gem_pool(self.feature_extractor(x)).flatten(1)  # (S, 768)
        return plane_norm(feats)

    def forward(self, sag, cor, axi):
        h_sag = self._encode_slices(sag, self.norm_sag)
        h_cor = self._encode_slices(cor, self.norm_cor)
        h_axi = self._encode_slices(axi, self.norm_axi)

        # Each: (3_tasks, 768)
        e_sag, w_sag = self.mil_sag(h_sag)
        e_cor, w_cor = self.mil_cor(h_cor)
        e_axi, w_axi = self.mil_axi(h_axi)

        # Stack → (3_tasks, 3_planes, 768) + plane positional encoding
        tokens = torch.stack([e_sag, e_cor, e_axi], dim=1) + self.plane_pos

        # Cross-plane attention (batch dimension = num_tasks here)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))  # (3_tasks, 3_planes, 768)

        # Per-task logit from its fused 3-plane representation
        logits = torch.cat([
            self.task_heads[t](tokens[t].flatten().unsqueeze(0))
            for t in range(len(self.task_heads))
        ], dim=1)  # (1, 3)

        return logits, (w_sag, w_cor, w_axi)
