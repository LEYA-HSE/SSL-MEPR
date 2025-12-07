import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
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
            nn.Linear(hidden_dim // 2, hidden_dim)
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
        # h: [B, N, D]
        # adj: [B, N, N] with 0/1 mask
        B, N, D = h.size()
        Wh = self.W(h)  # [B, N, D']
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D']
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D']
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # [B, N, N, 2D']
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # mask non-neighbors
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # [B, N, D']
        return h_prime

class FeatureSlice(nn.Module):
    """
    Обрезает конкат-вектор [emo‖pkl]:
        mode='both' — вернуть как есть
        mode='emo'  — взять ЛЕВУЮ половину (Emo)
        mode='pkl'  — взять ПРАВУЮ половину (PKL)
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
class MultiModalFusionModelWithAblation(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1,
                 emo_out_dim=7, pkl_out_dim=5, device='cpu', ablation_config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

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

        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                ModalityProjector(in_dim, hidden_dim, dropout),
                AdapterFusion(hidden_dim, dropout)
            )
            for mod, in_dim in self.modalities.items()
        })

        if not self.disable_graph_attn:
            self.graph_attn = GraphAttentionLayer(hidden_dim, dropout=dropout)

        self.emo_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pkl_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        if not self.disable_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.emo_head = nn.Linear(hidden_dim, emo_out_dim)
        self.pkl_head = nn.Linear(hidden_dim, pkl_out_dim)

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
                x_proj = self.projectors[mod](feat.to(self.device))  # [D]
                x_mods.append(x_proj)
                valid_modalities.append(mod)

        if not x_mods:
            raise ValueError("No valid modality features found")

        x_mods = torch.stack(x_mods, dim=1)  # [B=1, N, D]
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

                emo_final = (emo_pred + emo_sim) / 2
            else:
                emo_final = emo_pred

            if not self.ablation_config.get("disable_guide_pkl", False):
                guides_pkl = self.guide_bank_pkl()  # [pkl_out_dim, D]
                pkl_sim = F.cosine_similarity(pkl_repr.unsqueeze(1), guides_pkl.unsqueeze(0), dim=-1)

                pkl_final = (pkl_pred + torch.sigmoid(pkl_sim)) / 2
            else:
                pkl_final = pkl_pred
        else:
            emo_final = emo_pred
            pkl_final = pkl_pred

        return {'emotion_logits': emo_final, "personality_scores": pkl_final}
