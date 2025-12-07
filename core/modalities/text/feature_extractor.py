# modalities/text/feature_extractor.py
import torch
from transformers import AutoTokenizer, AutoModel
from .model_loader import load_fusion_model


class PretrainedTextEmbeddingExtractor:


    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "BAAI/bge-small-en-v1.5",
        fusion_ckpt: str = "core/modalities/text/checkpoints/Mamba_Transformer_bge-small_fusion.pt",
        emo_ckpt: str   = "core/modalities/text/checkpoints/Mamba_bge-small_emotion.pt",
        per_ckpt: str   = "core/modalities/text/checkpoints/Transformer_bge-small_personality.pt",
    ):
        self.device = torch.device(device)

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device).eval()

        self.fusion, _ = load_fusion_model(
            fusion_ckpt, emo_ckpt, per_ckpt, device=self.device
        )

    @torch.no_grad()
    def extract(self, texts: list[str] | str) -> dict:
        if isinstance(texts, str):
            texts = [texts]

        batch = self.tok(texts, padding=True, truncation=True,
                        return_tensors="pt").to(self.device)

        hidden = self.enc(**batch).last_hidden_state  # (B, T, 384)

        out = self.fusion(
            emotion_input=hidden,
            personality_input=hidden,
            return_features=True,
        )

        return {
            "emotion_logits": out["emotion_logits"].cpu(),
            "personality_scores": out["personality_scores"].cpu(),
            "last_emo_encoder_features": out["last_emo_encoder_features"].cpu(),
            "last_per_encoder_features": out["last_per_encoder_features"].cpu(),
        }
