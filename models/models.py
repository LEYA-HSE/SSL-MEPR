import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .layers.layer1 import GraphAttentionLayer_v1
from .layers.layer2 import GraphAttentionLayer_v2
from .layers.layer3 import GraphAttentionLayer_v3
from .layers.layer4 import GraphAttentionLayer_v4

from .attention.crossmpt.Model_CrossMPT import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder,
    EncoderLayer,
)


# ───────────────────────────── Core building blocks ─────────────────────────────
class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.layernorm(x + self.adapter(x))


class GuideBank(nn.Module):
    def __init__(self, out_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(out_dim, hidden_dim))

    def forward(self):
        return self.embeddings


class GraphAttentionLayer(nn.Module):
    """Standard graph attention layer for multimodal fusion."""
    def __init__(self, in_dim, out_dim=None, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        out_dim = out_dim or in_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, h, adj):
        """
        h:   [B, N, D]
        adj: [B, N, N] binary mask
        """
        B, N, D = h.size()
        Wh = self.W(h)                                # [B, N, D']
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D']
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D']
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)     # [B, N, N, 2D']
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # mask non-neighbors
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)         # [B, N, D']
        return h_prime


class FeatureSlice(nn.Module):
    """
    Slice concatenated vector [emo‖pkl]:
      - mode='both' → return as is
      - mode='emo'  → take the left half (Emo)
      - mode='pkl'  → take the right half (PKL)
    """
    def __init__(self, mode: str = "both"):
        super().__init__()
        if mode not in ("both", "emo", "pkl"):
            raise ValueError("feature_slice must be 'both' | 'emo' | 'pkl'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "both":
            return x
        half = x.size(-1) // 2
        return x[..., :half] if self.mode == "emo" else x[..., half:]


# ───────────────────────────── Multimodal fusion (with ablations) ─────────────────────────────
class MultiModalFusionModelWithAblation(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        emo_out_dim=7,
        pkl_out_dim=5,
        device='cpu',
        ablation_config=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        # Ablation configuration
        self.ablation_config = ablation_config or {}
        self.disabled_modalities = set(self.ablation_config.get("disabled_modalities", []))
        self.disable_graph_attn = self.ablation_config.get("disable_graph_attn", False)
        self.disable_cross_attn = self.ablation_config.get("disable_cross_attn", False)
        self.disable_emo_logit_proj = self.ablation_config.get("disable_emo_logit_proj", False)
        self.disable_pkl_logit_proj = self.ablation_config.get("disable_pkl_logit_proj", False)
        self.disable_guide_bank = self.ablation_config.get("disable_guide_bank", False)

        self.modalities = {
            'body': 1024 * 2,
            'face': 512 * 2,
            'scene': 768 * 2,
            'audio': 256 * 2,
            'text': 256 * 2,
        }
        # Alternative asymmetric dims (kept for reference)
        # self.modalities = {
        #     'body': 256+1024,
        #     'face': 1024+256,
        #     'scene': 512+512,
        #     'audio': 256+256,
        #     'text': 256+512,
        # }

        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                ModalityProjector(in_dim, hidden_dim, dropout),
                AdapterFusion(hidden_dim, dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        if not self.disable_graph_attn:
            self.graph_attn = GraphAttentionLayer(hidden_dim, dropout=dropout)
            # self.graph_attn = GraphAttentionLayer_v4(hidden_dim, dropout=dropout)

        self.emo_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pkl_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        if not self.disable_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # Cross-MPT alternative (kept commented for reference)
            # c = copy.deepcopy
            # attn = MultiHeadedAttention(num_heads, hidden_dim)
            # ff = PositionwiseFeedForward(hidden_dim, hidden_dim * 4, dropout)
            # self.cross_emo = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), 1)
            # self.cross_pkl = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), 1)

        self.emo_head = nn.Linear(hidden_dim, emo_out_dim)
        self.pkl_head = nn.Linear(hidden_dim, pkl_out_dim)

        # Optional learned fusers (kept for reference)
        # self.emo_fusion = nn.Linear(2, 1)
        # self.pkl_fusion = nn.Linear(2, 1)

        if not self.disable_guide_bank:
            self.guide_bank_emo = GuideBank(emo_out_dim, hidden_dim)
            self.guide_bank_pkl = GuideBank(pkl_out_dim, hidden_dim)

        if not self.disable_emo_logit_proj:
            self.emo_logit_proj = nn.Linear(emo_out_dim, hidden_dim)
        if not self.disable_pkl_logit_proj:
            self.per_logit_proj = nn.Linear(pkl_out_dim, hidden_dim)

    def forward(self, batch):
        x_mods = []
        valid_modalities = []

        for mod, feat in batch['features'].items():
            if feat is not None and mod in self.projectors and mod not in self.disabled_modalities:
                x_proj = self.projectors[mod](feat.to(self.device))  # [B, D]
                x_mods.append(x_proj)
                valid_modalities.append(mod)

        if not x_mods:
            raise ValueError("No valid modality features found")

        x_mods = torch.stack(x_mods, dim=1)  # [B, N, D]
        B, N, D = x_mods.size()

        if self.disable_graph_attn:
            context = x_mods
        else:
            adj = torch.ones(B, N, N, device=self.device)
            context = self.graph_attn(x_mods, adj)  # [B, N, D]

        emo_q = self.emo_query.expand(B, 1, -1)  # [B, 1, D]
        pkl_q = self.pkl_query.expand(B, 1, -1)  # [B, 1, D]

        if self.disable_cross_attn:
            emo_repr = context.mean(dim=1)
            pkl_repr = context.mean(dim=1)
        else:
            emo_repr, _ = self.cross_attn(emo_q, context, context)
            pkl_repr, _ = self.cross_attn(pkl_q, context, context)
            emo_repr = emo_repr.squeeze(1)
            pkl_repr = pkl_repr.squeeze(1)

            # Cross-MPT alternative (kept commented for reference)
            # emb1_e, emb2_e = self.cross_emo(emo_q, context, None, None)   # [B,1,D], [B,N,D]
            # emo_repr = torch.cat([emb1_e, emb2_e], dim=1).mean(dim=1)     # [B,D]
            #
            # emb1_p, emb2_p = self.cross_pkl(pkl_q, context, None, None)   # [B,1,D], [B,N,D]
            # pkl_repr = torch.cat([emb1_p, emb2_p], dim=1).mean(dim=1)     # [B,D]

        emo_logit_feats = []
        per_logit_feats = []
        for mod in valid_modalities:
            emo_logit_feats.append(batch['emotion_logits'][mod].to(self.device))
            per_logit_feats.append(batch['personality_scores'][mod].to(self.device))

        if emo_logit_feats and not self.disable_emo_logit_proj:
            emo_repr += self.emo_logit_proj(torch.stack(emo_logit_feats).mean(dim=0))
        if per_logit_feats and not self.disable_pkl_logit_proj:
            pkl_repr += self.per_logit_proj(torch.stack(per_logit_feats).mean(dim=0))

        emo_pred = self.emo_head(emo_repr)
        pkl_pred = torch.sigmoid(self.pkl_head(pkl_repr))

        if not self.disable_guide_bank:
            if not self.ablation_config.get("disable_guide_emo", False):
                guides_emo = self.guide_bank_emo()  # [emo_out_dim, D]
                emo_sim = F.cosine_similarity(emo_repr.unsqueeze(1), guides_emo.unsqueeze(0), dim=-1)
                # emo_stack = torch.stack([emo_pred, emo_sim], dim=-1)  # [B, C, 2]
                # emo_final = self.emo_fusion(emo_stack).squeeze(-1)    # [B, C]
                emo_final = (emo_pred + emo_sim) / 2
            else:
                emo_final = emo_pred

            if not self.ablation_config.get("disable_guide_pkl", False):
                guides_pkl = self.guide_bank_pkl()  # [pkl_out_dim, D]
                pkl_sim = F.cosine_similarity(pkl_repr.unsqueeze(1), guides_pkl.unsqueeze(0), dim=-1)
                # pkl_stack = torch.stack([pkl_pred, torch.sigmoid(pkl_sim)], dim=-1)
                # pkl_final = self.pkl_fusion(pkl_stack).squeeze(-1)
                pkl_final = (pkl_pred + torch.sigmoid(pkl_sim)) / 2
            else:
                pkl_final = pkl_pred
        else:
            emo_final = emo_pred
            pkl_final = pkl_pred

        return {'emotion_logits': emo_final, "personality_scores": pkl_final}


# ───────────────────────────── Single-task (emotion OR PKL) ─────────────────────────────
class SingleTaskSlimModel(nn.Module):
    """
    Single-domain model, four modes:
      ID 0: both → Emo
      ID 1: emo  → Emo
      ID 2: both → PKL
      ID 3: pkl  → PKL
    """
    # Base dimension of a single half of the concatenated vector
    BASE_DIMS = {
        "body": 1024,
        "face": 512,
        "scene": 768,
        "audio": 256,
        "text": 256,
    }

    def __init__(
        self,
        target: str,                 # 'emo' | 'pkl'
        feature_slice: str = "both", # 'both' | 'emo' | 'pkl'
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        device: str = "cpu",
    ):
        super().__init__()
        if target not in ("emo", "pkl"):
            raise ValueError("target must be 'emo' or 'pkl'")

        self.device = device
        self.target = target

        # Input dims depend on slicing mode
        if feature_slice == "both":
            self.modalities = {m: d * 2 for m, d in self.BASE_DIMS.items()}
        else:
            self.modalities = self.BASE_DIMS.copy()

        self.projectors = nn.ModuleDict({
            m: nn.Sequential(
                ModalityProjector(in_dim, hidden_dim, dropout),
                AdapterFusion(hidden_dim, dropout),
            )
            for m, in_dim in self.modalities.items()
        })

        self.slice = FeatureSlice(feature_slice)

        self.gat = GraphAttentionLayer(hidden_dim, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        if target == "emo":
            self.head = nn.Linear(hidden_dim, emo_out_dim)
            self.guide = GuideBank(emo_out_dim, hidden_dim)
            self.logit_proj = nn.Linear(emo_out_dim, hidden_dim)
        else:
            self.head = nn.Linear(hidden_dim, pkl_out_dim)
            self.guide = None
            self.logit_proj = nn.Linear(pkl_out_dim, hidden_dim)

    def forward(self, batch: dict):
        """batch['features'] is a dict: modality → Tensor[B, dim]."""
        feats = []
        for mod, x in batch["features"].items():
            if x is None or mod not in self.projectors:
                continue
            x = self.slice(x)
            feats.append(self.projectors[mod](x.to(self.device)))

        if not feats:
            raise RuntimeError("No valid modality features")

        x = torch.stack(feats, dim=1)  # [B, N, D]
        B, N, _ = x.shape
        ctx = self.gat(x, torch.ones(B, N, N, device=self.device))

        rep = self.cross(self.query.expand(B, -1, -1), ctx, ctx)[0].squeeze(1)

        # Auxiliary logits from the opposite domain
        aux_key = "emotion_logits" if self.target == "emo" else "personality_scores"
        aux = [t.to(self.device) for t in batch.get(aux_key, {}).values() if t is not None]
        if aux:
            rep = rep + self.logit_proj(torch.stack(aux).mean(0))

        out = self.head(rep)
        if self.target == "emo":
            sim = F.cosine_similarity(rep.unsqueeze(1), self.guide().unsqueeze(0), -1)
            return {"emotion_logits": (out + sim) / 2, "personality_scores": None}

        return {"emotion_logits": None, "personality_scores": torch.sigmoid(out)}


class MultiModalFusionModel(nn.Module):
    """Simplified multimodal fusion model without ablation switches."""
    def __init__(self, hidden_dim=512, num_heads=8, emo_out_dim=7, pkl_out_dim=5, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.modalities = {
            'body': 1024 * 2,
            'face': 512 * 2,
            'scene': 768 * 2,
            'audio': 256 * 2,
            'text': 256 * 2,
        }

        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                ModalityProjector(in_dim, hidden_dim),
                AdapterFusion(hidden_dim),
            )
            for mod, in_dim in self.modalities.items()
        })

        self.graph_attn = GraphAttentionLayer(hidden_dim)

        self.emo_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pkl_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.emo_head = nn.Linear(hidden_dim, emo_out_dim)
        self.pkl_head = nn.Linear(hidden_dim, pkl_out_dim)

        self.guide_bank_emo = GuideBank(emo_out_dim, hidden_dim)
        self.guide_bank_pkl = GuideBank(pkl_out_dim, hidden_dim)

        self.emo_logit_proj = nn.Linear(emo_out_dim, hidden_dim)
        self.per_logit_proj = nn.Linear(pkl_out_dim, hidden_dim)

    def forward(self, batch):
        x_mods = []
        valid_modalities = []
        for mod, feat in batch['features'].items():
            if feat is not None and mod in self.projectors:
                x_proj = self.projectors[mod](feat.to(self.device))  # [B, D]
                x_mods.append(x_proj)
                valid_modalities.append(mod)

        if not x_mods:
            raise ValueError("No valid modality features found")

        x_mods = torch.stack(x_mods, dim=1)  # [B, N, D]
        B, N, D = x_mods.size()

        # Fully-connected adjacency across modalities
        adj = torch.ones(B, N, N, device=self.device)

        context = self.graph_attn(x_mods, adj)  # [B, N, D]

        emo_q = self.emo_query.expand(B, -1, -1)  # [B, 1, D]
        pkl_q = self.pkl_query.expand(B, -1, -1)  # [B, 1, D]

        emo_repr, _ = self.cross_attn(emo_q, context, context)
        pkl_repr, _ = self.cross_attn(pkl_q, context, context)

        emo_repr = emo_repr.squeeze(1)
        pkl_repr = pkl_repr.squeeze(1)

        emo_logit_feats = []
        per_logit_feats = []
        for mod in valid_modalities:
            emo_logit_feats.append(batch['emotion_logits'][mod].to(self.device))
            per_logit_feats.append(batch['personality_scores'][mod].to(self.device))

        if emo_logit_feats:
            emo_repr += self.emo_logit_proj(torch.stack(emo_logit_feats).mean(dim=0))
        if per_logit_feats:
            pkl_repr += self.per_logit_proj(torch.stack(per_logit_feats).mean(dim=0))

        emo_pred = self.emo_head(emo_repr)
        pkl_pred = torch.sigmoid(self.pkl_head(pkl_repr))

        guides_emo = self.guide_bank_emo()  # [emo_out_dim, D]
        guides_pkl = self.guide_bank_pkl()  # [pkl_out_dim, D]

        emo_sim = F.cosine_similarity(emo_repr.unsqueeze(1), guides_emo.unsqueeze(0), dim=-1)
        pkl_sim = F.cosine_similarity(pkl_repr.unsqueeze(1), guides_pkl.unsqueeze(0), dim=-1)

        emo_final = (emo_pred + emo_sim) / 2
        pkl_final = (pkl_pred + torch.sigmoid(pkl_sim)) / 2

        return {'emotion_logits': emo_final, "personality_scores": pkl_final}


# ───────────────────────────── Asymmetric slicer ─────────────────────────────
class AsymFeatureSlice(nn.Module):
    """
    Slice concatenated [Emo‖PKL] vector using a per-modality split:
      - mode='both' → return as is
      - mode='emo'  → keep left (Emo)
      - mode='pkl'  → keep right (PKL)
    """
    def __init__(self, mode: str, split_idx: dict[str, int]):
        super().__init__()
        if mode not in ("both", "emo", "pkl"):
            raise ValueError("mode must be 'both' | 'emo' | 'pkl'")
        self.mode = mode
        self.split_idx = split_idx

    def forward(self, x: torch.Tensor, mod: str) -> torch.Tensor:
        if self.mode == "both":
            return x
        k = self.split_idx.get(mod, x.size(-1) // 2)
        return x[..., :k] if self.mode == "emo" else x[..., k:]


# ───────────────────────────── Single-task (asymmetric) ─────────────────────────────
class SingleTaskAsymModel(nn.Module):
    """Single-task model for asymmetric concatenated inputs."""

    EMO_DIMS = dict(body=256, face=1024, scene=512, audio=256, text=256)
    PKL_DIMS = dict(body=1024, face=256, scene=512, audio=256, text=512)
    TOTAL_DIMS = {}
    for m in EMO_DIMS:
        TOTAL_DIMS[m] = EMO_DIMS[m] + PKL_DIMS[m]

    def __init__(
        self,
        target: str,                 # 'emo' | 'pkl'
        feature_slice: str = "both", # 'both' | 'emo' | 'pkl'
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        device: str = "cpu",
    ):
        super().__init__()
        if target not in ("emo", "pkl"):
            raise ValueError("target must be 'emo' or 'pkl'")

        self.device = device
        self.target = target

        # Select projector input dims based on slicing mode
        if feature_slice == "both":
            mods = self.TOTAL_DIMS
        elif feature_slice == "emo":
            mods = self.EMO_DIMS
        else:  # 'pkl'
            mods = self.PKL_DIMS

        self.projectors = nn.ModuleDict({
            m: nn.Sequential(
                ModalityProjector(d, hidden_dim, dropout),
                AdapterFusion(hidden_dim, dropout),
            )
            for m, d in mods.items()
        })

        # Per-modality split indices
        self.slice = AsymFeatureSlice(feature_slice, split_idx=self.EMO_DIMS)

        self.gat = GraphAttentionLayer(hidden_dim, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        if target == "emo":
            self.head = nn.Linear(hidden_dim, emo_out_dim)
            self.guide = GuideBank(emo_out_dim, hidden_dim)
            self.logit_proj = nn.Linear(emo_out_dim, hidden_dim)
        else:
            self.head = nn.Linear(hidden_dim, pkl_out_dim)
            self.guide = None
            self.logit_proj = nn.Linear(pkl_out_dim, hidden_dim)

    def forward(self, batch: dict):
        feats = []
        for mod, x in batch["features"].items():
            if x is None or mod not in self.projectors:
                continue
            x = self.slice(x, mod)  # slice by modality
            feats.append(self.projectors[mod](x.to(self.device)))

        if not feats:
            raise RuntimeError("No valid modality features")

        x = torch.stack(feats, 1)  # [B, N, D]
        B, N, _ = x.shape
        ctx = self.gat(x, torch.ones(B, N, N, device=self.device))
        rep = self.cross(self.query.expand(B, -1, -1), ctx, ctx)[0].squeeze(1)

        aux_key = "emotion_logits" if self.target == "emo" else "personality_scores"
        aux = [t.to(self.device) for t in batch.get(aux_key, {}).values() if t is not None]
        if aux:
            rep = rep + self.logit_proj(torch.stack(aux).mean(0))

        out = self.head(rep)
        if self.target == "emo":
            sim = F.cosine_similarity(rep.unsqueeze(1), self.guide().unsqueeze(0), -1)
            return {"emotion_logits": (out + sim) / 2, "personality_scores": None}

        return {"emotion_logits": None, "personality_scores": torch.sigmoid(out)}
