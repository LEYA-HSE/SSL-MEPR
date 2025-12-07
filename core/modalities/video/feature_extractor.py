import torch
from transformers import CLIPProcessor,CLIPModel
from .model_loader import get_fusion_model

class PretrainedImageEmbeddingExtractor:


    def __init__(self, device="cuda",
                 clip_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.body_model = get_fusion_model("body", device)
        self.face_model = get_fusion_model("face", device)
        self.scene_model = get_fusion_model("scene", device)

    @torch.no_grad()
    def extract(self, *, body_tensor=None, face_tensor=None, scene_tensor=None):

        results = {}

        modality_map = {
            "body": (body_tensor, self.body_model),
            "face": (face_tensor, self.face_model),
            "scene": (scene_tensor, self.scene_model),
        }

        for name, (tensor, model) in modality_map.items():
            if tensor is None:
                continue

            features = self.clip_model.get_image_features(tensor.to(self.device))  # [N, D]

            # features_input = features.unsqueeze(0)  # [1, N, D]
            features_input = torch.unsqueeze(features, 0)

            out = model(
                emotion_input=features_input,
                personality_input=features_input,
                return_features=True,
            )

            result = {
                "emotion_logits": out["emotion_logits"].cpu(),
                "personality_scores": out["personality_scores"].cpu(),
                "last_emo_encoder_features": out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": out["last_per_encoder_features"].cpu(),
            }

            results[name] = result

        return results
