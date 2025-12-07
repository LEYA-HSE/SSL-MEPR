# coding: utf-8
import os, pickle, logging
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

# Extracts frame ROIs and converts them to tensors
from modalities.video.video_preprocessor import get_metadata


class MultimodalDataset(Dataset):
    """
    Multimodal dataset for body, face (later — audio, text, scene).
    Reads a CSV, extracts features, and caches them into a pickle file.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        audio_dir: str,
        config,
        split: str,
        modality_processors: dict,
        modality_feature_extractors: dict,
        dataset_name: str,
        device: str = "cuda",
    ):
        super().__init__()

        # ───────── basic fields ─────────
        self.csv_path          = csv_path
        self.video_dir         = video_dir
        self.audio_dir         = audio_dir
        self.config            = config
        self.split             = split
        self.dataset_name      = dataset_name
        self.device            = device
        self.segment_length    = config.counter_need_frames
        self.subset_size       = config.subset_size
        self.average_features  = config.average_features

        # ───────── modality dicts ─────────
        self.modality_processors         = modality_processors
        self.extractors: dict[str, object] = modality_feature_extractors

        # ───────── cache setup ─────────
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.feature_filename   = (
            f"{self.dataset_name}_{self.split}"
            f"_seed_{config.random_seed}_subset_size_{self.subset_size}"
            f"_average_features_{self.average_features}_feature_norm_{config.emb_normalize}.pickle"
        )

        # ───── label setup ─────
        if self.dataset_name == 'cmu_mosei':
            self.emotion_columns = [
                "Neutral", "Anger", "Disgust", "Fear",
                "Happiness", "Sadness", "Surprise"
            ]
            self.personality_columns  = []
        elif self.dataset_name == 'fiv2':
            self.personality_columns = [
                "openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"
            ]
            self.emotion_columns = []
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.num_emotion = 7
        self.num_personality = 5

        # ───────── read CSV ─────────
        self.df = pd.read_csv(self.csv_path).dropna()

        if self.subset_size > 0:
            self.df = self.df.head(self.subset_size)
            logging.info(f"[DatasetMultiModal] Using only the first {len(self.df)} records (subset_size={self.subset_size}).")

        self.video_names = sorted(self.df["video_name"].unique())
        self.meta: list[dict] = []

        # ───────── load from pickle or prepare from scratch ─────────
        if self.save_prepared_data:
            os.makedirs(self.save_feature_path, exist_ok=True)
            self.pickle_path = os.path.join(self.save_feature_path, self.feature_filename)
            self._load_pickle(self.pickle_path)

            if not self.meta:
                self._prepare_data()
                self._save_pickle(self.pickle_path)
        else:
            self._prepare_data()

    # ────────────────────────── utils ──────────────────────────── #
    def _find_file(self, base_dir: str, base_filename: str):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_filename:
                    return os.path.join(root, file)
        return None

    def _save_pickle(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.meta = []

    def _make_label_dict(
        self,
        emotion:      torch.Tensor | None,
        personality:  torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | None]:
        """
        Returns a dict with both keys. If labels are missing, values are None.
        """
        return {
            "emotion":     emotion,
            "personality": personality
        }

    # ──────────────────────────────────────────────────────────────────
    # feature extraction

    def _prepare_data(self):

        for name in tqdm(self.video_names, desc="Extracting multimodal features"):

            video_path = self._find_file(self.video_dir, name)
            audio_path = self._find_file(self.audio_dir, name)

            if video_path is None:
                print(f"❌ Video not found: {name}")
                continue
            if audio_path is None:
                print(f"❌ Audio not found: {name}")
                continue

            entry = {
                "sample_name": name,
                "video_path": video_path,
                "audio_path": audio_path,
                "features": {},
            }

            # ---------- visual modalities -------------------- #
            try:
                # --- detection and frame preprocessing -----------------------
                __, body, face, scene = get_metadata(
                    video_path      = video_path,
                    segment_length  = self.segment_length,
                    image_processor = self.modality_processors.get("body"),
                    device          = self.device,
                )

                # --- feature extraction via pretrained models -----------------
                extracted = self.extractors["body"].extract(
                    body_tensor = body,
                    face_tensor = face,
                    scene_tensor = scene,
                )

                for m in ("body", "face", "scene"):
                    entry["features"][m] = (
                        self._aggregate(extracted.get(m), self.average_features)
                    )
            except Exception as e:
                logging.warning(f"Video extract error {name}: {e}")

            # ---------- audio / text ------------------------------ #
            try:
                audio_feats = self.extractors["audio"].extract(audio_path=audio_path)
                entry["features"]["audio"] = self._aggregate(audio_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Audio extract error {name}: {e}")
                entry["features"]["audio"] = None

            try:
                txt_raw = self.df[self.df["video_name"] == name]["text"].values[0]
                text_feats = self.extractors["text"].extract(txt_raw)
                entry["features"]["text"] = self._aggregate(text_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Text extract error {name}: {e}")
                entry["features"]["text"] = None

            # ---------- labels ------------------------------------- #
            try:
                emotion_tensor     = None
                personality_tensor = None

                #   ─ emotion ─
                if self.emotion_columns:
                    emotion_tensor = torch.tensor(
                        self.df.loc[
                            self.df["video_name"] == name, self.emotion_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    emotion_tensor = torch.full(
                        (self.num_emotion,), torch.nan, dtype=torch.float32
                    )

                #   ─ personality ─
                if self.personality_columns:
                    personality_tensor = torch.tensor(
                        self.df.loc[
                            self.df["video_name"] == name, self.personality_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    personality_tensor = torch.full(
                        (self.num_personality,), torch.nan, dtype=torch.float32
                    )

                entry["labels"] = self._make_label_dict(
                    emotion_tensor,
                    personality_tensor
                )

            except Exception as e:
                logging.warning(f"Label extract error {name}: {e}")
                entry["labels"] = self._make_label_dict(
                    torch.full((self.num_emotion,), torch.nan, dtype=torch.float32),
                    torch.full((self.num_personality,), torch.nan, dtype=torch.float32),
                )

            self.meta.append(entry)
            torch.cuda.empty_cache()

    def _aggregate(self, feats, average: bool = None):
        """
        Unified feature aggregation.

        Args:
            feats (Union[Tensor, dict, None]): Input features.
            average (bool): If True — average over time (dim=1) when applicable.

        Returns:
            Aggregated features or None.

        - If feats is a Tensor with shape [B, T, D] and average=True → average over T.
        - If average=False → return as is.
        - If feats is a dict → recurse over values.
        - If feats is None → return None.
        """

        if average is None:
            average = self.average_features

        if feats is None:
            return None

        if isinstance(feats, torch.Tensor):
            if average and feats.ndim == 3:
                feats = feats.mean(dim=1)  # → [B, D]
            return feats.squeeze()

        if isinstance(feats, dict):
            return {
                key: self._aggregate(val, average)
                for key, val in feats.items()
            }

        raise TypeError(f"Unsupported feature type: {type(feats)}")

    # ───────────────────── dataset API ─────────────────────────── #
    def __len__(self):  return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]
