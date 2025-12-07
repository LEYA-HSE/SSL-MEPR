# inference.py
# coding: utf-8
from pathlib import Path
import cv2
import torch
from transformers import CLIPProcessor
import numpy as np
from PIL import Image

from models.models import *
from utils.config_loader import ConfigLoader
from data_loading.feature_extractor import PretrainedImageEmbeddingExtractor

class FullPipelineFeatureExtractor:
    def __init__(self, seed: int, counter_need_frames: int, image_feature_extractor: PretrainedImageEmbeddingExtractor):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.seed = seed
        self.counter_need_frames = counter_need_frames
        self.image_feature_extractor = image_feature_extractor

    def pth_processing(self, img):
        img = self.processor(images=img, return_tensors="pt").to("cuda")
        img = img['pixel_values']
        return img
    
    def select_uniform_frames(self, frames, N) -> list:
        if len(frames) <= N:
            return list(frames)  # Если кадров меньше N, вернуть все
        else:
            np.random.seed(self.seed)
            indices = np.linspace(0, len(frames) - 1, num=N, dtype=int)
            return [frames[i] for i in indices]
    
    def __call__(self, video_path: str | Path) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        need_curr_frames = self.select_uniform_frames(np.arange(num_frames), self.counter_need_frames)

        counter = 1
        all_frames = []
        while True:
            ret, im0 = cap.read()
            if not ret:
                break
            
            if counter in need_curr_frames:
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                curr_fr = im0
                curr_fr = self.pth_processing(Image.fromarray(curr_fr))
                all_frames.append(curr_fr)
            counter += 1

        cap.release()

        all_frames = torch.cat(all_frames, dim=0)
        video_features = self.image_feature_extractor.extract(all_frames).to('cpu')

        torch.cuda.empty_cache()

        return video_features


def main():
    CONFIG_PATH = 'clip_config_transformer_mamba.toml'
    VIDEO_PATH = '/home/serg_fedchn/LEYA/EAAI_2025/Datasets/CMU-MOSEI/test/-6rXp3zJ3kc_14.4680_22.8820.mp4'

    config = ConfigLoader(CONFIG_PATH)

    image_feature_extractor = PretrainedImageEmbeddingExtractor(config)
    full_pipeline_feature_extractor = FullPipelineFeatureExtractor(seed=config.random_seed,
                                                                   counter_need_frames=config.counter_need_frames,
                                                                   image_feature_extractor=image_feature_extractor)
    
    dict_models = {
        "EmotionMamba": EmotionMamba,
        "PersonalityMamba": PersonalityMamba,
        "EmotionTransformer": EmotionTransformer,
        "PersonalityTransformer": PersonalityTransformer,
        "FusionTransformer": FusionTransformer
    }

    # параметры задаем для лучшей эмоциональной модели
    model_cls = dict_models[config.name_best_emo_model]
    emo_model = model_cls(
    input_dim_emotion     = config.image_embedding_dim,
    input_dim_personality = config.image_embedding_dim,
    hidden_dim            = config.hidden_dim_emo,
    out_features          = config.out_features_emo,
    tr_layer_number       = config.tr_layer_number_emo,
    num_transformer_heads = config.num_transformer_heads_emo,
    positional_encoding   = config.positional_encoding_emo,
    mamba_d_model         = config.mamba_d_state_emo,
    mamba_layer_number    = config.mamba_layer_number_emo,
    dropout               = config.dropout,
    num_emotions          = 7,
    num_traits            = 5,
    device                = config.emb_device
    ).to(config.emb_device)
    # параметры задаем для лучшей персональной модели
    model_cls = dict_models[config.name_best_per_model]
    per_model = model_cls(
    input_dim_emotion     = config.image_embedding_dim,
    input_dim_personality = config.image_embedding_dim,
    hidden_dim            = config.hidden_dim_per,
    out_features          = config.out_features_per,
    per_activation        = config.best_per_activation,
    tr_layer_number       = config.tr_layer_number_per,
    num_transformer_heads = config.num_transformer_heads_per,
    positional_encoding   = config.positional_encoding_per,
    mamba_d_model         = config.mamba_d_state_per,
    mamba_layer_number    = config.mamba_layer_number_per,
    dropout               = config.dropout,
    num_emotions          = 7,
    num_traits            = 5,
    device                = config.emb_device
    ).to(config.emb_device)

    emo_state = torch.load(config.path_to_saved_emotion_model, map_location=config.emb_device)
    emo_model.load_state_dict(emo_state)
    emo_model.eval()

    per_state = torch.load(config.path_to_saved_personality_model, map_location=config.emb_device)
    per_model.load_state_dict(per_state)
    per_model.eval()

    model_cls = dict_models[config.model_name]
    model = model_cls(
        emo_model             = emo_model,
        per_model             = per_model,
        input_dim_emotion     = config.image_embedding_dim,
        input_dim_personality = config.image_embedding_dim,
        hidden_dim            = config.hidden_dim,
        out_features          = config.out_features,
        per_activation        = config.per_activation,
        tr_layer_number       = config.tr_layer_number,
        num_transformer_heads = config.num_transformer_heads,
        positional_encoding   = config.positional_encoding,
        mamba_d_model         = config.mamba_d_state,
        mamba_layer_number    = config.mamba_layer_number,
        dropout               = config.dropout,
        num_emotions          = 7,
        num_traits            = 5,
        device                = config.emb_device
        ).to(config.emb_device)
    
    fus_state = torch.load(config.path_to_saved_fusion_model, map_location=config.emb_device)
    model.load_state_dict(fus_state)
    model.eval()

    with torch.no_grad():
        video_features = full_pipeline_feature_extractor(VIDEO_PATH).to(config.emb_device)[None, :, :]
        outputs = model(emotion_input=video_features, personality_input=video_features)
        print(outputs)


if __name__ == "__main__":
    main()
