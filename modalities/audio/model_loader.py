# modalities/audio/model_loader.py

import torch
from .architectures import (
    CustomMambaClassifier,
    CustomMambaRegressor,
    FusionTransformer,
)

def load_pretrained_emotion_encoder(path, device):
    enc = CustomMambaClassifier(input_size=1024, d_model=256, num_layers=3, num_classes=7, dropout=0.2).to(device)
    ck = torch.load(path, map_location=device)
    enc.load_state_dict(ck['model_state_dict'])
    enc.output_dim = 256

    def extract_features(x, lengths):
        h = enc.input_proj(x)
        for blk in enc.blocks:
            h = blk(h)
        return h

    enc.extract_features = extract_features
    enc.eval()
    return enc

def load_pretrained_personality_encoder(path, device):
    enc = CustomMambaRegressor(input_size=1024, d_model=256, num_layers=3, num_targets=5, dropout=0.2).to(device)
    ck = torch.load(path, map_location=device)
    enc.load_state_dict(ck)
    enc.output_dim = 256

    def extract_features(x, lengths):
        h = enc.input_proj(x)
        for blk in enc.blocks:
            h = blk(h)
        return h

    enc.extract_features = extract_features
    enc.eval()
    return enc

def load_fusion_model(
    fusion_ckpt_path: str,
    emo_encoder_ckpt: str,
    per_encoder_ckpt: str,
    device: str = 'cpu'
):
    device = torch.device(device)
    emo_enc = load_pretrained_emotion_encoder(emo_encoder_ckpt, device)
    per_enc = load_pretrained_personality_encoder(per_encoder_ckpt, device)
    ckpt       = torch.load(fusion_ckpt_path, map_location=device)
    best_cfg   = ckpt['config']
    state_dict = ckpt['state_dict']
    model = FusionTransformer(emo_enc, per_enc, best_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device
