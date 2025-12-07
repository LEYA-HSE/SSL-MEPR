# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .help_layers import TransformerEncoderLayer
from data_loading.pretrained_extractors import CustomMambaBlock
        
class EmotionMamba(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, len_seq = 30, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        # self.emo_proj = nn.Sequential(
        #     nn.Linear(input_dim_emotion, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     # nn.BatchNorm1d(len_seq),
        #     nn.Dropout(dropout)
        # )
        self.emo_proj = nn.Sequential(
            # nn.BatchNorm1d(len_seq),
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
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo = self.emo_proj(emotion_input)

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
    
class PersonalityMamba(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, len_seq = 30, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        # self.per_proj = nn.Sequential(
        #     nn.Linear(input_dim_personality, hidden_dim),
        #     # nn.BatchNorm1d(len_seq),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout)
        # )

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_personality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.personality_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
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

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        per = self.per_proj(personality_input)

        for layer in self.personality_encoder:
            per = layer(per)

        out_per = self.personality_fc_out(per.mean(dim=1))
    
        if return_features:
            return {
                'personality_scores': self.activation(out_per) if activation else out_per,
                'last_encoder_features': per,
            }
        else:
            return {'personality_scores': self.activation(out_per) if activation else out_per}
    
class EmotionTransformer(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, len_seq = 30, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        # self.emo_proj = nn.Sequential(
        #     nn.Linear(input_dim_emotion, hidden_dim),
        #     # nn.BatchNorm1d(len_seq),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout)
        # )

        self.emo_proj = nn.Sequential(
            # nn.BatchNorm1d(len_seq),
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo = self.emo_proj(emotion_input)

        for layer in self.emotion_encoder:
            emo += layer(emo, emo, emo)

        out_emo = self.emotion_fc_out(emo.mean(dim=1))  # (B, num_emotions)
        
        if return_features:
            return {
                'emotion_logits': out_emo,
                'last_encoder_features': emo,
            }
        else:
            return {'emotion_logits': out_emo}
    
class PersonalityTransformer(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, len_seq = 30, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        # self.per_proj = nn.Sequential(
        #     nn.Linear(input_dim_personality, hidden_dim),
        #     # nn.BatchNorm1d(len_seq),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout)
        # )
        self.per_proj = nn.Sequential(
            # nn.BatchNorm1d(len_seq),
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

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        per = self.per_proj(personality_input)

        for layer in self.personality_encoder:
            per += layer(per, per, per)

        out_per = self.personality_fc_out(per.mean(dim=1))
        
        if return_features:
            return {
                'personality_scores': self.activation(out_per) if activation else out_per,
                'last_encoder_features': per,
            }
        else:
            return {'personality_scores': self.activation(out_per) if activation else out_per}
    
class FusionTransformer(nn.Module):
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, 
                 hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, 
                 per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, 
                 tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.emo_model = emo_model
        self.per_model = per_model

        for param in self.emo_model.parameters():
            param.requires_grad = False

        for param in self.per_model.parameters():
            param.requires_grad = False

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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )        

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        emo_features = self.emo_model(emotion_input=emotion_input, return_features=True)
        per_features = self.per_model(personality_input=personality_input, return_features=True)

        emo_emd = self.emo_proj(emo_features['last_encoder_features'])
        per_emd = self.per_proj(per_features['last_encoder_features'])

        for layer in self.emotion_to_personality_attn:
            emo_emd += layer(emo_emd, per_emd, per_emd) # or per_emd, emo_emd, emo_emd

        for layer in self.personality_to_emotion_attn:
            per_emd += layer(per_emd, emo_emd, emo_emd) # or emo_emd, per_emd, per_emd

        fused = torch.cat([emo_emd, per_emd], dim=-1)
        emotion_logits = self.emotion_personality_fc_out(fused.mean(dim=1))
        personality_scores = self.personality_emotion_fc_out(fused.mean(dim=1))

        personality_scores = self.activation(personality_scores) if activation else personality_scores

        if return_features:
            return {
                'emotion_logits': (emotion_logits+emo_features['emotion_logits'])/2,
                'personality_scores': (personality_scores+per_features['personality_scores'])/2,
                'last_emo_encoder_features': emo_emd,
                'last_per_encoder_features': per_emd,
            }
        else:
            return {'emotion_logits': (emotion_logits+emo_features['emotion_logits'])/2,
                    'personality_scores': (personality_scores+per_features['personality_scores'])/2,}
        
class EnhancedFusionTransformer(nn.Module):
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, 
                 hidden_dim=256, out_features=512, per_activation="sigmoid", mamba_layer_number=2, mamba_d_model=256,
                 positional_encoding=True, num_transformer_heads=8, tr_layer_number=2, 
                 dropout=0.2, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        # Замороженные предобученные модели
        self.emo_model = emo_model
        self.per_model = per_model
        for param in self.emo_model.parameters():
            param.requires_grad = False
        for param in self.per_model.parameters():
            param.requires_grad = False

        # Улучшенные проекционные слои с расширенной емкостью
        self.emo_proj = nn.Sequential(
            nn.Linear(self.emo_model.hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.per_proj = nn.Sequential(
            nn.Linear(self.per_model.hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Улучшенные трансформерные слои с остаточными соединениями
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

        # Доменно-специфичные адаптивные веса
        self.domain_weights = nn.Parameter(torch.ones(2))
        self.softmax = nn.Softmax(dim=0)

        # Улучшенные выходные слои с промежуточными представлениями
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_head = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(out_features, num_traits)
        )

        # Активации
        self.per_activation = nn.Sigmoid() if per_activation == "sigmoid" else nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        # Получаем фичи из предобученных моделей
        emo_features = self.emo_model(emotion_input=emotion_input, return_features=True)
        per_features = self.per_model(personality_input=personality_input, return_features=True)

        # Проекция в общее пространство
        emo_emb = self.emo_proj(emo_features['last_encoder_features'])
        per_emb = self.per_proj(per_features['last_encoder_features'])

        # Кросс-доменное внимание с остаточными соединениями
        for layer in self.emotion_to_personality_attn:
            emo_emb = emo_emb + self.gelu(layer(emo_emb, per_emb, per_emb))

        for layer in self.personality_to_emotion_attn:
            per_emb = per_emb + self.gelu(layer(per_emb, emo_emb, emo_emb))

        # Конкатенация и адаптивное взвешивание
        fused = torch.cat([emo_emb, per_emb], dim=-1)
        
        # Средний пулинг с сохранением размерности [batch, seq, features] -> [batch, features]
        pooled = fused.mean(dim=1)
        
        # Вычисляем доменные веса
        domain_weights = self.softmax(self.domain_weights)
        
        # Прогнозирование с адаптивным взвешиванием
        emotion_logits = self.emotion_head(pooled) * domain_weights[0]
        personality_scores = self.personality_head(pooled) * domain_weights[1]
        
        # Residual connection с оригинальными предсказаниями
        emotion_logits = (emotion_logits + emo_features['emotion_logits']) / 2
        personality_scores = (self.per_activation(personality_scores) + per_features['personality_scores']) / 2

        if return_features:
            return {
                'emotion_logits': emotion_logits,
                'personality_scores': personality_scores,
                'domain_weights': domain_weights.detach(),
                'last_emo_encoder_features': emo_emb,
                'last_per_encoder_features': per_emb,
            }
        else:
            return {
                'emotion_logits': emotion_logits,
                'personality_scores': personality_scores
            }
        
class ProbabilityFusion(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(ProbabilityFusion, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))

    def forward(self, pred):
        # print(pred)
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_matrix = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix, normalized_weights
    
class FusionTransformerWithProbWeightedFusion(nn.Module):
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, mamba_layer_number=2, mamba_d_model=256,
                 hidden_dim=256, out_features=512, per_activation="sigmoid", 
                 positional_encoding=True, num_transformer_heads=8, tr_layer_number=2, 
                 dropout=0.2, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        # Замороженные предобученные модели
        self.emo_model = emo_model
        self.per_model = per_model
        for param in self.emo_model.parameters():
            param.requires_grad = False
        for param in self.per_model.parameters():
            param.requires_grad = False

        # Улучшенные проекционные слои с расширенной емкостью
        self.emo_proj = nn.Sequential(
            nn.Linear(self.emo_model.hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.per_proj = nn.Sequential(
            nn.Linear(self.per_model.hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Улучшенные трансформерные слои с остаточными соединениями
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

        self.weighted_emo = ProbabilityFusion(num_matrices=2, num_classes=num_emotions)
        self.weighted_per = ProbabilityFusion(num_matrices=2, num_classes=num_traits)

        # Улучшенные выходные слои с промежуточными представлениями
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_head = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(out_features, num_traits)
        )

        # Активации
        self.per_activation = nn.Sigmoid() if per_activation == "sigmoid" else nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        # Получаем фичи из предобученных моделей
        emo_features = self.emo_model(emotion_input=emotion_input, return_features=True)
        per_features = self.per_model(personality_input=personality_input, return_features=True)

        # Проекция в общее пространство
        emo_emb = self.emo_proj(emo_features['last_encoder_features'])
        per_emb = self.per_proj(per_features['last_encoder_features'])

        # Кросс-доменное внимание с остаточными соединениями
        for layer in self.emotion_to_personality_attn:
            emo_emb = emo_emb + layer(per_emb, emo_emb, emo_emb)

        for layer in self.personality_to_emotion_attn:
            per_emb = per_emb + layer(emo_emb, per_emb, per_emb)

        # Конкатенация и адаптивное взвешивание
        fused = torch.cat([emo_emb, per_emb], dim=-1)
        
        # Средний пулинг с сохранением размерности [batch, seq, features] -> [batch, features]
        pooled = fused.mean(dim=1)
        
        emotion_logits = self.emotion_head(pooled)
        personality_scores = self.personality_head(pooled)

        emotion_logits, weighted_emo = self.weighted_emo([emotion_logits, emo_features['emotion_logits']])
        personality_scores, weighted_per = self.weighted_per([self.per_activation(personality_scores), per_features['personality_scores']])

        if return_features:
            return {
                'emotion_logits': emotion_logits,
                'personality_scores': personality_scores,
                'weighted_emo': weighted_emo,
                'weighted_per': weighted_per,
                'last_emo_encoder_features': emo_emb,
                'last_per_encoder_features': per_emb,
            }
        else:
            return {
                'emotion_logits': emotion_logits,
                'personality_scores': personality_scores
            }
        
class FusionTransformer2(nn.Module):
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.emo_model = emo_model
        self.per_model = per_model

        for param in self.emo_model.parameters():
            param.requires_grad = False

        for param in self.per_model.parameters():
            param.requires_grad = False

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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )        

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        emo_features = self.emo_model(emotion_input=emotion_input, return_features=True)
        per_features = self.per_model(personality_input=personality_input, return_features=True)

        emo_emd = self.emo_proj(emo_features['last_encoder_features'])
        per_emd = self.per_proj(per_features['last_encoder_features'])

        for layer in self.emotion_to_personality_attn:
            emo_emd += layer(emo_emd, per_emd, per_emd) # or per_emd, emo_emd, emo_emd

        for layer in self.personality_to_emotion_attn:
            per_emd += layer(per_emd, emo_emd, emo_emd) # or emo_emd, per_emd, per_emd

        fused = torch.cat([emo_emd, per_emd], dim=-1)
        emotion_logits = self.emotion_personality_fc_out(fused.mean(dim=1))
        personality_scores = self.personality_emotion_fc_out(fused.mean(dim=1))

        personality_scores = self.activation(personality_scores) if activation else personality_scores

        if return_features:
            return {
                'emotion_logits': emotion_logits,
                'personality_scores': personality_scores,
                'last_emo_encoder_features': emo_emd,
                'last_per_encoder_features': per_emd,
            }
        else:
            return {'emotion_logits': emotion_logits,
                    'personality_scores': personality_scores}