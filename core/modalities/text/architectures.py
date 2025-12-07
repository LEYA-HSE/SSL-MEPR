# core/modalities/text/architectures.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomMambaBlock(nn.Module):
    def __init__(self, d_input, d_model, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_model)
        self.s_B = nn.Linear(d_model, d_model)
        self.s_C = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.in_proj(x)
        B = self.s_B(x)
        C = self.s_C(x)
        x = x + B + C
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = self.norm(x + x_in)
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_slice = self.pe[: x.size(1)].to(x.device)
        x = x + pe_slice.detach()
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, positional_encoding=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def _ensure_on_device(self, device: torch.device):
        self.self_attention.to(device)
        self.feed_forward.to(device)
        self.add_norm_after_attention.to(device)
        self.add_norm_after_ff.to(device)
        if self.positional_encoding is not None:
            self.positional_encoding.to(device)

    def forward(self, query, key, value):
        dev = query.device
        self._ensure_on_device(dev)

        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(query, key, value, need_weights=False)
        x = self.add_norm_after_attention(attn_output, query)
        ff_output = self.feed_forward(x)
        x = self.add_norm_after_ff(ff_output, x)
        return x


class EmotionMamba(nn.Module):
    def __init__(self, input_dim_emotion=1024, input_dim_personality=1024, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, positional_encoding=True, num_transformer_heads=4, transformer_dropout=0.1, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo = self.emo_proj(emotion_input)  # (B, T, hidden_dim)
        for layer in self.emotion_encoder:
            emo = layer(emo)
        out_emo = self.emotion_fc_out(emo.mean(dim=1))  # (B, num_emotions)
        if return_features:
            return {
                'emotion_logits': out_emo,
                'last_encoder_features': emo,
            }
        else:
            return {'emotion_logits': out_emo}


class PersonalityTransformer(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_personality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.personality_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        per = self.per_proj(personality_input)

        for layer in self.personality_encoder:
            per += layer(per, per, per)

        out_per = self.personality_fc_out(per.mean(dim=1))

        if return_features:
            return {
                'personality_scores': self.activation(out_per),
                'last_encoder_features': per,
            }
        else:
            return {'personality_scores': self.activation(out_per)}


class FusionTransformer(nn.Module):
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

        self.hidden_dim = hidden_dim

        self.emo_model = emo_model
        self.per_model = per_model

        for param in self.emo_model.parameters():
            param.requires_grad = False
        for param in self.per_model.parameters():
            param.requires_grad = False

        self.emo_model.to(self.device)
        self.per_model.to(self.device)

        self.emo_proj = nn.Sequential(
            nn.Linear(self.emo_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.per_proj = nn.Sequential(
            nn.Linear(self.per_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_to_personality_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.personality_to_emotion_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.emotion_personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

        self.to(self.device)

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo_features = self.emo_model(emotion_input=emotion_input, return_features=True)
        per_features = self.per_model(personality_input=personality_input, return_features=True)

        emo_emd = self.emo_proj(emo_features['last_encoder_features'])
        per_emd = self.per_proj(per_features['last_encoder_features'])

        max_len = max(emo_emd.shape[1], per_emd.shape[1])
        emo_emd_np = emo_emd.detach().cpu().numpy()
        per_emd_np = per_emd.detach().cpu().numpy()
        emo_emd_np = np.pad(emo_emd_np[:, :max_len, :], ((0, 0), (0, max(0, max_len - emo_emd_np.shape[1])), (0, 0)), "constant")
        per_emd_np = np.pad(per_emd_np[:, :max_len, :], ((0, 0), (0, max(0, max_len - per_emd_np.shape[1])), (0, 0)), "constant")
        emo_emd = torch.tensor(emo_emd_np, device=self.device)
        per_emd = torch.tensor(per_emd_np, device=self.device)

        for layer in self.emotion_to_personality_attn:
            emo_emd += layer(emo_emd, per_emd, per_emd)

        for layer in self.personality_to_emotion_attn:
            per_emd += layer(per_emd, emo_emd, emo_emd)

        fused = torch.cat([emo_emd, per_emd], dim=-1)

        dev = fused.device
        self.emotion_personality_fc_out.to(dev)
        self.personality_emotion_fc_out.to(dev)

        emotion_logits = self.emotion_personality_fc_out(fused.mean(dim=1))
        personality_scores = self.personality_emotion_fc_out(fused.mean(dim=1))

        base_emo_logits = emo_features['emotion_logits'].to(self.device)
        base_per_scores = per_features['personality_scores'].to(self.device)

        if return_features:
            return {
                'emotion_logits': (emotion_logits + base_emo_logits) / 2,
                'personality_scores': (self.activation(personality_scores) + base_per_scores) / 2,
                'last_emo_encoder_features': emo_emd,
                'last_per_encoder_features': per_emd,
            }
        else:
            return {
                'emotion_logits': (emotion_logits + base_emo_logits) / 2,
                'personality_scores': (self.activation(personality_scores) + base_per_scores) / 2,
            }
