# modalities/audio/modules.py

import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0] # [B, T, 1024]
        return hidden_states


class CustomMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.s_B = nn.Linear(d_model, d_model)
        self.s_C = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.in_proj(x)
        x = x + self.s_B(x) + self.s_C(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return self.norm(x + x_in)


class CustomMambaClassifier(nn.Module):
    def __init__(self, input_size=1024, d_model=256, num_layers=2, num_classes=7, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([CustomMambaBlock(d_model, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        pooled = [x[i, :L].mean(dim=0) if L > 0 else torch.zeros(x.size(-1), device=x.device)
                  for i, L in enumerate(lengths)]
        return self.fc(torch.stack(pooled, dim=0))


class CustomMambaRegressor(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_targets, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([CustomMambaBlock(d_model, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, num_targets)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        pooled = [x[i, :L].mean(dim=0) if L > 0 else torch.zeros(x.size(-1), device=x.device)
                  for i, L in enumerate(lengths)]
        return self.head(torch.stack(pooled, dim=0))


class FusionTransformer(nn.Module):
    def __init__(self, emo_enc, per_enc, config):
        super().__init__()
        for p in emo_enc.parameters(): p.requires_grad = False
        for p in per_enc.parameters(): p.requires_grad = False

        self.emo_enc = emo_enc
        self.per_enc = per_enc

        h = config['hidden_dim']
        d = config['dropout']
        heads = config['tr_heads']

        self.emo_proj = nn.Sequential(
            nn.Linear(emo_enc.output_dim, h),
            nn.LayerNorm(h),
            nn.Dropout(d)
        )
        self.per_proj = nn.Sequential(
            nn.Linear(per_enc.output_dim, h),
            nn.LayerNorm(h),
            nn.Dropout(d)
        )

        self.mha_e2p = nn.MultiheadAttention(embed_dim=h, num_heads=heads, dropout=d, batch_first=True)
        self.mha_p2e = nn.MultiheadAttention(embed_dim=h, num_heads=heads, dropout=d, batch_first=True)

        self.emo_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(d),
            nn.Linear(h, config['num_emotions'])
        )
        self.per_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(d),
            nn.Linear(h, config['num_traits'])
        )

    def forward(self, emo_input=None, per_input=None, return_features=False):
        base_emo_logits = base_per_scores = fe = fp = None

        if emo_input is not None:
            x_e, len_e = emo_input
            feat_e = self.emo_enc.extract_features(x_e, len_e)
            base_emo_logits = self.emo_enc(x_e, len_e)
            emo_emd = self.emo_proj(feat_e)
            fe = emo_emd.mean(dim=1)

        if per_input is not None:
            x_p, len_p = per_input
            feat_p = self.per_enc.extract_features(x_p, len_p)
            base_per_scores = self.per_enc(x_p, len_p)
            per_emd = self.per_proj(feat_p)
            fp = per_emd.mean(dim=1)

        if emo_input is not None and per_input is not None:
            attn_e2p, _ = self.mha_e2p(query=emo_emd, key=per_emd, value=per_emd)
            emo_emd = emo_emd + attn_e2p

            attn_p2e, _ = self.mha_p2e(query=per_emd, key=emo_emd, value=emo_emd)
            per_emd = per_emd + attn_p2e

            fe = emo_emd.mean(dim=1)
            fp = per_emd.mean(dim=1)
            cat = torch.cat([fe, fp], dim=-1)

            emo_new = self.emo_head(cat)
            per_new = self.per_head(cat)

            final_emo = (emo_new + base_emo_logits) / 2
            final_per = (per_new + base_per_scores) / 2

            return {
                'emotion_logits': final_emo,
                'personality_scores': final_per,
                'last_emo_encoder_features': emo_emd,
                'last_per_encoder_features': per_emd,
            } if return_features else {
                'emotion_logits': final_emo,
                'personality_scores': final_per,
            }

        elif emo_input is not None:
            return {
                'emotion_logits': base_emo_logits,
                'last_emo_encoder_features': fe,
            } if return_features else {
                'emotion_logits': base_emo_logits,
            }

        else:
            return {
                'personality_scores': base_per_scores,
                'last_per_encoder_features': fp,
            } if return_features else {
                'personality_scores': base_per_scores,
            }
