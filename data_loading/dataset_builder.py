# coding: utf-8
"""
dataset_builder.py
------------------
*  Формирует MultimodalDataset для указанного split'а
*  Создаёт DataLoader с кастомным collate_fn
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data_loading.dataset_multimodal import MultimodalDataset

CORE_KEYS = {
    "body":  ["last_emo_encoder_features", "last_per_encoder_features"],
    "face":  ["last_emo_encoder_features", "last_per_encoder_features"],
    "scene": ["last_emo_encoder_features", "last_per_encoder_features"],
    "audio": ["last_emo_encoder_features", "last_per_encoder_features"],
    "text":  ["last_emo_encoder_features", "last_per_encoder_features"],
}

# ────────────────────────────────────────────────────────────────────────
#                             COLLATE FN
# ────────────────────────────────────────────────────────────────────────
def _stack_core_feats(feat_dict: dict, modal: str) -> torch.Tensor:
    """Конкатенируем только нужные ключи в единый вектор."""
    parts = [feat_dict[k] for k in CORE_KEYS[modal] if k in feat_dict]
    return torch.cat(parts)             # [D_total]

def custom_collate_fn(batch):
    # Удаляем None и те, где хоть одна модальность отсутствует
    filtered_batch = []
    for sample in batch:
        if sample is None or "features" not in sample:
            continue
        modalities = sample["features"].keys()
        has_all_modalities = all(sample["features"].get(m) is not None for m in modalities)
        if has_all_modalities:
            filtered_batch.append(sample)

    if not filtered_batch:
        return None

    # --------- собираем features ---------
    features = {}          # modality → Tensor([B, D])
    metas    = {}          # modality → dict списков «побочных» полей (логиты)

    modalities = filtered_batch[0]["features"].keys()

    emo_pred = {}
    per_pred = {}

    for m in modalities:
        core_vecs = []
        emo_logits = []
        per_logits = []
        for sample in filtered_batch:
            core_vecs.append(_stack_core_feats(sample["features"][m], m))
            emo_logits.append(sample["features"][m]["emotion_logits"])
            per_logits.append(sample["features"][m]["personality_scores"])

        features[m] = torch.stack(core_vecs)
        emo_pred[m] = torch.stack(emo_logits)
        per_pred[m] = torch.stack(per_logits)

    # --------- labels ---------
    emo = [b["labels"]["emotion"] for b in filtered_batch]
    person = [b["labels"]["personality"] for b in filtered_batch]
    emo = torch.stack(emo)
    person = torch.stack(person)

    return {
        "features": features,
        "labels": {
            "emotion": emo,
            "personality": person,
        },
        "emotion_logits": emo_pred,
        "personality_scores": per_pred,
    }


# ────────────────────────────────────────────────────────────────────────
#               Функция создания датасета + DataLoader
# ────────────────────────────────────────────────────────────────────────
def make_dataset_and_loader(
    config,
    split: str,
    modality_processors: Dict[str, Any],
    modality_extractors: Dict[str, Any],
    *,
    only_dataset: str | None = None,
):
    """
    Собирает (возможное объединение) MultimodalDataset'ов и возвращает DataLoader.
    * config.datasets — словарь с описанием каждого датасета (см. config.toml)
    * split — 'train' / 'dev' / 'test'
    * only_dataset — если указан, обрабатывается только он
    """
    if not getattr(config, "datasets", None):
        raise ValueError("⛔ В конфиге не указана секция [datasets].")

    datasets: List[MultimodalDataset] = []

    for dataset_name, ds_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        csv_path = ds_cfg["csv_path"].format(
            base_dir=ds_cfg["base_dir"], split=split
        )
        video_dir = ds_cfg["video_dir"].format(
            base_dir=ds_cfg["base_dir"], split=split
        )
        audio_dir = ds_cfg["audio_dir"].format(
            base_dir=ds_cfg["base_dir"], split=split
        )

        dataset = MultimodalDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            config=config,
            split=split,
            modality_processors=modality_processors,
            modality_feature_extractors=modality_extractors,
            dataset_name=dataset_name,
            device=config.device,
        )
        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ Для split='{split}' не найдено ни одного датасета.")

    full_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
    )

    return full_dataset, loader
