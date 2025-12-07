import torch
from .architectures import (
    EmotionMamba,
    PersonalityTransformer,
    FusionTransformer,
)

def load_pretrained_emotion_encoder(checkpoint_path, device):
    emotion_model = EmotionMamba(
        input_dim_emotion=384,
        input_dim_personality=384,
        hidden_dim=256,
        out_features=256,
        mamba_layer_number=4,
        mamba_d_model=256,
        dropout=0.2
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    emotion_model.load_state_dict(state_dict)

    def extract_features(inputs, lengths):
        features = emotion_model.emo_proj(inputs)
        for block in emotion_model.emotion_encoder:
            features = block(features)
        return features

    emotion_model.extract_features = extract_features
    emotion_model.eval()
    return emotion_model

def load_pretrained_personality_encoder(checkpoint_path, device):
    personality_model = PersonalityTransformer(
        input_dim_emotion=384,
        input_dim_personality=384,
        hidden_dim=512,
        out_features=128,
        num_transformer_heads=8,
        tr_layer_number=1,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    personality_model.load_state_dict(checkpoint)

    def extract_features(inputs, lengths):
        features = personality_model.per_proj(inputs)
        for block in personality_model.personality_encoder:
            features = block(features, features, features)
        return features

    personality_model.extract_features = extract_features
    personality_model.eval()
    return personality_model

def load_fusion_model(
    fusion_checkpoint_path: str,
    emotion_encoder_checkpoint: str,
    personality_encoder_checkpoint: str,
    device: str = "cpu",
):
    device = torch.device(device)

    emotion_encoder = load_pretrained_emotion_encoder(emotion_encoder_checkpoint, device)
    personality_encoder = load_pretrained_personality_encoder(personality_encoder_checkpoint, device)

    checkpoint = torch.load(fusion_checkpoint_path, map_location=device)

    fusion_model = FusionTransformer(
        emo_model=emotion_encoder,
        per_model=personality_encoder,
        input_dim_emotion=384,
        input_dim_personality=384,
        hidden_dim=256,
        out_features=256,
        tr_layer_number=2,
        num_transformer_heads=4
    ).to(device)
    fusion_model.load_state_dict(checkpoint)
    fusion_model.eval()
    return fusion_model, device
