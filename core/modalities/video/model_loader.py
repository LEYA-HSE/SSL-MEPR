# coding: utf-8
import torch
import toml
from pathlib import Path
from .architectures import EmotionMamba, PersonalityMamba, FusionTransformer, EmotionTransformer

MODALITY_META = {
    "body": {
        "toml": Path("core/modalities/video/config/inference_config_body.toml"),
        "ckpt": "core/modalities/video/checkpoints/body/clip_body_mamba_transformer_fusion_model.pt",
    },
    "face": {
        "toml": Path("core/modalities/video/config/inference_config_face.toml"),
        "ckpt": "core/modalities/video/checkpoints/face/clip_face_mamba_transformer_fusion_model.pt",
    },
    "scene": {
        "toml": Path("core/modalities/video/config/inference_config_scene.toml"),
        "ckpt": "core/modalities/video/checkpoints/scene/clip_fusion_transformer_transformer_mamba_best_model_dev.pt",
    },
}

def parse_config(toml_path: str) -> dict:
    toml_data = toml.load(toml_path)

    embeddings = toml_data["embeddings"]
    model = toml_data["train"]["model"]

    config = {
        # Embeddings
        "image_embedding_dim": embeddings["image_embedding_dim"],
        "counter_need_frames": embeddings["counter_need_frames"],

        # Emotion branch
        "hidden_dim_emo": model["hidden_dim_emo"],
        "out_features_emo": model["out_features_emo"],
        "tr_layer_number_emo": model["tr_layer_number_emo"],
        "num_transformer_heads_emo": model["num_transformer_heads_emo"],
        "positional_encoding_emo": model["positional_encoding_emo"],
        "mamba_d_state_emo": model["mamba_d_state_emo"],
        "mamba_layer_number_emo": model["mamba_layer_number_emo"],


        # Personality branch
        "hidden_dim_per": model["hidden_dim_per"],
        "out_features_per": model["out_features_per"],
        "tr_layer_number_per": model["tr_layer_number_per"],
        "num_transformer_heads_per": model["num_transformer_heads_per"],
        "positional_encoding_per": model["positional_encoding_per"],
        "mamba_d_state_per": model["mamba_d_state_per"],
        "mamba_layer_number_per": model["mamba_layer_number_per"],
        "best_per_activation": model.get("best_per_activation", "sigmoid"),

        # Fusion transformer
        "hidden_dim": model["hidden_dim"],
        "out_features": model["out_features"],
        "per_activation": model.get("per_activation", "sigmoid"),
        "tr_layer_number": model["tr_layer_number"],
        "num_transformer_heads": model["num_transformer_heads"],
        "positional_encoding": model["positional_encoding"],
        "mamba_d_state": model["mamba_d_state"],
        "mamba_layer_number": model["mamba_layer_number"],

        # General
        "dropout": model["dropout"],
    }

    return config

def get_fusion_model(modality: str, device: str = "cuda") -> FusionTransformer:
    if modality not in MODALITY_META:
        raise ValueError("modality must be one of 'body', 'face', 'scene'")

    meta = MODALITY_META[modality]
    config = parse_config(meta["toml"])
    if modality == 'scene':
        EmoClass = EmotionTransformer
    else:
        EmoClass = EmotionMamba

    emotion_model = EmoClass(
        input_dim_emotion     = config["image_embedding_dim"],
        input_dim_personality = config["image_embedding_dim"],
        len_seq               = config["counter_need_frames"],
        hidden_dim            = config["hidden_dim_emo"],
        out_features          = config["out_features_emo"],
        tr_layer_number       = config["tr_layer_number_emo"],
        num_transformer_heads = config["num_transformer_heads_emo"],
        positional_encoding   = config["positional_encoding_emo"],
        mamba_d_model         = config["mamba_d_state_emo"],
        mamba_layer_number    = config["mamba_layer_number_emo"],
        dropout               = config["dropout"],
        num_emotions          = 7,
        num_traits            = 5,
        device                = device,
    ).to(device).eval()

    personality_model = PersonalityMamba(
        input_dim_emotion     = config["image_embedding_dim"],
        input_dim_personality = config["image_embedding_dim"],
        len_seq               = config["counter_need_frames"],
        hidden_dim            = config["hidden_dim_per"],
        out_features          = config["out_features_per"],
        per_activation        = config["best_per_activation"],
        tr_layer_number       = config["tr_layer_number_per"],
        num_transformer_heads = config["num_transformer_heads_per"],
        positional_encoding   = config["positional_encoding_per"],
        mamba_d_model         = config["mamba_d_state_per"],
        mamba_layer_number    = config["mamba_layer_number_per"],
        dropout               = config["dropout"],
        num_emotions          = 7,
        num_traits            = 5,
        device                = device,
    ).to(device).eval()

    fusion_model = FusionTransformer(
        emo_model             = emotion_model,
        per_model             = personality_model,
        input_dim_emotion     = config["image_embedding_dim"],
        input_dim_personality = config["image_embedding_dim"],
        hidden_dim            = config["hidden_dim"],
        out_features          = config["out_features"],
        per_activation        = config["per_activation"],
        tr_layer_number       = config["tr_layer_number"],
        num_transformer_heads = config["num_transformer_heads"],
        positional_encoding   = config["positional_encoding"],
        mamba_d_model         = config["mamba_d_state"],
        mamba_layer_number    = config["mamba_layer_number"],
        dropout               = config["dropout"],
        num_emotions          = 7,
        num_traits            = 5,
        device                = device,
    ).to(device).eval()

    checkpoint_state = torch.load(meta["ckpt"], map_location=device)
    fusion_model.load_state_dict(checkpoint_state)

    return fusion_model
