# coding: utf-8
# train_utils.py

import torch
torch.autograd.set_detect_anomaly(True)
import logging
import random
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from typing import Type
import os
import datetime

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from utils.losses import MultiTaskLoss
from utils.measures import uar, mf1, acc_func, ccc
from models.models import (
EmotionMamba,
PersonalityMamba,
EmotionTransformer,
PersonalityTransformer,
FusionTransformer,
FusionTransformer2,
EnhancedFusionTransformer,
FusionTransformerWithProbWeightedFusion
)
from utils.schedulers import SmartScheduler
from data_loading.dataset_multimodal import DatasetVideo
from sklearn.utils.class_weight import compute_class_weight

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

def pad_to(x, target_size):
    n_repeat = target_size - x.size(0)
    if n_repeat <= 0:
        return x
    pad = x[-1:].repeat(n_repeat, *[1 for _ in x.shape[1:]])
    return torch.cat([x, pad], dim=0)

def transform_matrix(matrix):
    threshold1 = 1 - 1/7 
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo, true_emo):
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    true_emo = true_emo.cpu().detach().numpy()
    true_emo = np.where(true_emo > 0, 1, 0)[:, 1:].tolist()
    return pred_emo, true_emo

def get_smoothed_labels(audio_paths, original_labels, smooth_labels_df, smooth_mask, emotion_columns,  device):
    """
    audio_paths: список путей к аудиофайлам
    smooth_mask: тензор boolean с индексами для сглаживания
    Возвращает тензор сглаженных меток только для отмеченных примеров
    """

    # Получаем индексы для сглаживания
    smooth_indices = torch.where(smooth_mask)[0]

    # Создаем тензор для результатов (такого же размера как оригинальные метки)
    smoothed_labels = torch.zeros_like(original_labels)

    # print(smooth_labels_df, audio_paths)

    for idx in smooth_indices:
        audio_path = audio_paths[idx]
        # Получаем сглаженную метку из вашего DataFrame или другого источника
        smoothed_label = smooth_labels_df.loc[
            smooth_labels_df['video_name'] == audio_path[:-4],
            emotion_columns
        ].values[0]

        smoothed_labels[idx] = torch.tensor(smoothed_label, device=device)

    return smoothed_labels

def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    video_path = [b["video_path"] for b in batch]

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    videos = [b["video"] for b in batch] # new
    video_tensor = pad_sequence(videos, batch_first=True) # new

    return {
        "video_path": video_path,
        "video": video_tensor, # new
        "label": label_tensor,
    }

def get_weights(dataloader):
    emo_train_labels = []

    for i, data in enumerate(tqdm(dataloader)):
        true_emo = data['label']
        emo_train_labels.extend(true_emo.numpy())
        num_positives = np.sum(emo_train_labels, axis=0)
        num_negatives = len(emo_train_labels) - num_positives
        class_weights_emo = num_negatives / num_positives
    print('Веса классов emo: ', class_weights_emo)
    return class_weights_emo

def make_dataset_and_loader(config, split: str, audio_feature_extractor: Type = None, text_feature_extractor: Type = None, image_feature_extractor: Type = None, whisper_model: Type = None, only_dataset: str = None):
    """
    Универсальная функция: объединяет датасеты или возвращает один при only_dataset.
    При объединении train-датасетов — использует WeightedRandomSampler для балансировки.
    """
    datasets = []

    if not hasattr(config, "datasets") or not config.datasets:
        raise ValueError("⛔ В конфиге не указана секция [datasets].")

    for dataset_name, dataset_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        csv_path = dataset_cfg["csv_path"].format(base_dir=dataset_cfg["base_dir"], task=dataset_cfg["task"], split=split)
        # wav_dir  = dataset_cfg["wav_dir"].format(base_dir=dataset_cfg["base_dir"], split=split)
        video_dir  = dataset_cfg["video_dir"].format(base_dir=dataset_cfg["base_dir"], task=dataset_cfg["task"], split=split)
        task = dataset_cfg["task"]

        logging.info(f"[{dataset_name.upper()}], Task={task}, Split={split}: CSV={csv_path}, Video_DIR={video_dir}")

        dataset = DatasetVideo(
            csv_path=csv_path, 
            video_dir=video_dir, 
            config=config, 
            split=split, 
            image_feature_extractor=image_feature_extractor,
            dataset_name=dataset_name,
            task=task)

        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ Для split='{split}' не найдено ни одного подходящего датасета.")

    if len(datasets) == 1:

        full_dataset = datasets[0]
        loader = DataLoader(
            full_dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
    else:

        if split == "train":
            # sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                # sampler=sampler,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )
        else:
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )

    return full_dataset, loader

def run_emo_eval(model, loader, criterion, device="cuda", mode = "emotion"):
    """
    Оценка модели по задаче эмоций. Возвращает (loss, uar, mf1).
    """
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_targets = []
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            labels = batch["label"].to(device)      # shape: (B, 7)
            video  = batch["video"].to(device)      # shape: (B, D, F)

            if mode == "emotion":
                logits = model(emotion_input=video)
            elif mode == "fusion":
                logits = model(emotion_input=video, personality_input=video)
            loss = criterion({"emotion_logits": logits["emotion_logits"]}, {'emotion': labels})

            bs = video.shape[0]
            total_loss += loss.item() * bs
            total += bs

            # preds = logits['emotion_logits'].argmax(dim=1)
            # target = labels.argmax(dim=1)
            preds, target = process_predictions(logits['emotion_logits'], labels)
            total_preds.extend(preds)
            total_targets.extend(target)

    avg_loss = total_loss / total

    uar_m = uar(total_targets, total_preds)
    mf1_m = mf1(total_targets, total_preds)

    return avg_loss, uar_m, mf1_m

def run_per_eval(model, loader, criterion, device="cuda", mode="personality"):
    """
    Оценка модели по задаче персональные качества личности. Возвращает (loss, m_acc, m_ccc).
    """
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_targets = []
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            labels = batch["label"].to(device)      # shape: (B, 7)
            video  = batch["video"].to(device)      # shape: (B, D, F)
            if mode == "personality":
                logits = model(personality_input=video)
            elif mode == "fusion":
                logits = model(emotion_input=video, personality_input=video)
            loss = criterion({"personality_scores": logits["personality_scores"]}, {'personality': labels})

            bs = video.shape[0]
            total_loss += loss.item() * bs
            total += bs

            preds = logits['personality_scores']
            total_preds.extend(preds.detach().cpu().numpy())
            total_targets.extend(labels.detach().cpu().numpy())

    total_preds = np.array(total_preds)
    total_targets = np.array(total_targets)

    avg_loss = total_loss / total

    m_acc = acc_func(total_targets, total_preds)
    m_ccc = ccc(total_targets, total_preds)

    return avg_loss, m_acc, m_ccc

def train_once(config, train_loaders, dev_loaders, test_loaders, metrics_csv_path=None, model_stage='emotion'):
    """
    Логика обучения (train/dev/test).
    Возвращает лучшую метрику на dev и словарь метрик.
    """

    logging.info("== Запуск тренировки (train/dev/test) ==")

    checkpoint_dir = None
    if config.save_best_model:
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = f"{metrics_csv_path[:-4]}_timestamp"
        os.makedirs(checkpoint_dir, exist_ok=True)

    csv_writer = None
    csv_file = None

    if config.path_to_df_ls:
        df_ls = pd.read_csv(config.path_to_df_ls)

    if metrics_csv_path:
        csv_file = open(metrics_csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["split", "epoch", "dataset", "loss", "uar", "mf1", "mean"])


    # Seed
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(config.random_seed)
        generator = torch.Generator()
        generator.manual_seed(config.random_seed)
        logging.info(f"== Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("== Random seed не фиксирован (0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    weight_emotion        = config.weight_emotion
    weight_pers           = config.weight_pers
    pers_loss_type        = config.pers_loss_type
    flag_emo_weight       = config.flag_emo_weight
    weight_decay          = config.weight_decay
    momentum              = config.momentum
    lr                    = config.lr
    num_epochs            = config.num_epochs
    max_patience          = config.max_patience
    scheduler_type        = config.scheduler_type

    dict_models = {
        "EmotionMamba": EmotionMamba,
        "PersonalityMamba": PersonalityMamba,
        "EmotionTransformer": EmotionTransformer,
        "PersonalityTransformer": PersonalityTransformer,
        "FusionTransformer": FusionTransformer,
        "FusionTransformer2": FusionTransformer2,
        "EnhancedFusionTransformer": EnhancedFusionTransformer,
        "FusionTransformerWithProbWeightedFusion":FusionTransformerWithProbWeightedFusion,
    }

    model_cls = dict_models[config.model_name]

    if model_stage != 'fusion':
        model = model_cls(
            input_dim_emotion     = config.image_embedding_dim,
            input_dim_personality = config.image_embedding_dim,
            len_seq               = config.counter_need_frames, 
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
            device                = device
            ).to(device)
    if model_stage == 'fusion':
        # параметры задаем для лучшей эмоциональной модели
        model_cls = dict_models[config.name_best_emo_model]
        emo_model = model_cls(
        input_dim_emotion     = config.image_embedding_dim,
        input_dim_personality = config.image_embedding_dim,
        len_seq               = config.counter_need_frames, 
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
        device                = device
        ).to(device)
        # параметры задаем для лучшей персональной модели
        model_cls = dict_models[config.name_best_per_model]
        per_model = model_cls(
        input_dim_emotion     = config.image_embedding_dim,
        input_dim_personality = config.image_embedding_dim,
        len_seq               = config.counter_need_frames, 
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
        device                = device
        ).to(device)

        emo_state = torch.load(config.path_to_saved_emotion_model, map_location=device)
        emo_model.load_state_dict(emo_state)

        emo_state = torch.load(config.path_to_saved_personality_model, map_location=device)
        per_model.load_state_dict(emo_state)

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
            device                = device
            ).to(device)

    # Оптимизатор и лосс
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,momentum = momentum
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {config.optimizer}")

    logging.info(f"Используем оптимизатор: {config.optimizer}, learning rate: {lr}")

    if flag_emo_weight:
        emo_weights = get_weights(train_loaders['cmu_mosei'])
        criterion = MultiTaskLoss(weight_emotion=weight_emotion, 
                                weight_personality=weight_pers, 
                                emo_weights=torch.FloatTensor(emo_weights).to(device),
                                personality_loss_type=pers_loss_type)
    else:
        criterion = MultiTaskLoss(weight_emotion=weight_emotion, 
                                weight_personality=weight_pers,
                                personality_loss_type=pers_loss_type)
    # LR Scheduler
    steps_per_epoch = sum(1 for batch in train_loaders['cmu_mosei'] if batch is not None)
    scheduler = SmartScheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    # Early stopping по dev
    best_dev_mean = float("-inf")
    best_dev_metrics = {}
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"\n=== Эпоха {epoch} ===")
        model.train()

        if model_stage == 'emotion':
            dataloader = train_loaders['cmu_mosei']

            total_loss = 0.0
            total_samples = 0
            total_preds_emo = []
            total_targets_emo = []

            for batch in tqdm(dataloader):
                if batch is None:
                    continue

                inputs  = batch['video'].to(device)
                labels = batch['label'].to(device).type(torch.float32)
                # print(inputs.shape)
                outputs = model(emotion_input=inputs)
                # print(outputs, labels)
                loss = criterion(outputs, {"emotion": labels})
                # print(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(batch_level=True)

                bs = inputs.shape[0]
                total_loss += loss.item() * bs
                total_samples += bs

                # preds_emo = outputs['emotion_logits'].argmax(dim=1)
                # terget_emo = labels.argmax(dim=1)
                preds_emo, terget_emo =  process_predictions(outputs['emotion_logits'], labels)
                # terget_emo = emo_labels
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(terget_emo)

            train_loss = total_loss / total_samples
            uar_m = uar(total_targets_emo, total_preds_emo)
            mf1_m = mf1(total_targets_emo, total_preds_emo)

            mean_train = np.mean([uar_m, mf1_m])

            logging.info(
                f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, MF1={mf1_m:.4f}, "
                f"MEAN={mean_train:.4f}")
            
        elif model_stage == 'personality':
            # print(222)
            dataloader = train_loaders['fiv2']

            total_loss = 0.0
            total_samples = 0
            total_preds_pers = []
            total_targets_pers = []

            for batch in tqdm(dataloader):
                if batch is None:
                    continue

                inputs  = batch['video'].to(device)
                labels = batch['label'].to(device)

                outputs = model(personality_input=inputs)
                loss = criterion(outputs, {"personality": labels})

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(batch_level=True)

                bs = inputs.shape[0]
                total_loss += loss.item() * bs
                total_samples += bs

                preds_pers = outputs['personality_scores']
                total_preds_pers.extend(preds_pers.cpu().detach().numpy())
                total_targets_pers.extend(labels.cpu().detach().numpy())

            total_preds_pers = np.array(total_preds_pers)
            total_targets_pers = np.array(total_targets_pers)

            train_loss = total_loss / total_samples
            m_acc = acc_func(total_targets_pers, total_preds_pers)
            m_ccc = ccc(total_targets_pers, total_preds_pers)

            mean_train = np.mean([m_acc, m_ccc])

            logging.info(
                f"[TRAIN] Loss={train_loss:.4f}, mACC={m_acc:.4f}, mCCC={m_ccc:.4f}, "
                f"MEAN={mean_train:.4f}")
            
        elif model_stage == 'fusion':
            # print(333)

            total_loss = 0.0
            total_samples = 0
            total_preds_emo = []
            total_targets_emo = []
            total_preds_per = []
            total_targets_per = []

            emo_iter = infinite_loader(train_loaders['cmu_mosei'])
            pers_iter = infinite_loader(train_loaders['fiv2'])

            for step in tqdm(range(steps_per_epoch)):
                emo_batch = next(emo_iter)
                pers_batch = next(pers_iter)

                if emo_batch is None or pers_batch is None:
                    continue

                emo_input  = emo_batch['video'].to(device)
                emo_labels = emo_batch['label'].to(device).type(torch.float32)

                pers_input  = pers_batch['video'].to(device)
                pers_labels = pers_batch['label'].to(device)

                emo_bs = emo_input.size(0)
                pers_bs = pers_input.size(0)

                if emo_bs != pers_bs:
                    max_bs = max(emo_bs, pers_bs)

                    if emo_bs < max_bs:
                        emo_input = pad_to(emo_input, max_bs)
                        emo_labels = pad_to(emo_labels, max_bs)
                    if pers_bs < max_bs:
                        pers_input = pad_to(pers_input, max_bs)
                        pers_labels = pad_to(pers_labels, max_bs)
                
                outputs = model(emotion_input=emo_input, personality_input=pers_input)
                loss = criterion(outputs, {'emotion': emo_labels, 'personality': pers_labels})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(batch_level=True)

                bs = emo_input.shape[0]
                total_loss += loss.item() * bs

                # preds_emo = outputs['emotion_logits'].argmax(dim=1)
                # terget_emo = emo_labels.argmax(dim=1)
                preds_emo, terget_emo =  process_predictions(outputs['emotion_logits'], emo_labels)
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(terget_emo)
                
                preds_per = outputs['personality_scores']
                terget_per = pers_labels
                total_preds_per.extend(preds_per.cpu().detach().numpy().tolist())
                total_targets_per.extend(terget_per.cpu().detach().numpy().tolist())

                total_samples += bs

            total_preds_per = np.array(total_preds_per)
            total_targets_per = np.array(total_targets_per)

            train_loss = total_loss / total_samples
            uar_m = uar(total_targets_emo, total_preds_emo)
            mf1_m = mf1(total_targets_emo, total_preds_emo)

            m_acc = acc_func(total_targets_per, total_preds_per)
            m_ccc = ccc(total_targets_per, total_preds_per)

            mean_train = np.mean([uar_m, mf1_m, m_acc, m_ccc])

            logging.info(
                f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, MF1={mf1_m:.4f}, "
                f"mACC={m_acc:.4f}, mCCC={m_ccc:.4f}, MEAN={mean_train:.4f},"
            )

        # --- DEV ---
        dev_means = []
        dev_metrics_by_dataset = []

        if model_stage == 'emotion':
            name = 'cmu_mosei'
            loader = dev_loaders[name]
            d_loss, d_uar, d_mf1 = run_emo_eval(
                    model, loader, criterion, device, model_stage
                )
            dev_means.extend([d_uar, d_mf1])
            dev_metrics = {
                    "loss": d_loss, "uar": d_uar,
                    "mf1": d_mf1, "mean": np.mean([d_uar, d_mf1])
                }
            
            if csv_writer:
                row = ["dev", epoch, name, *dev_metrics.values()]
                while len(row) < 9:  # выравнивание колонок
                    row.append("")
                csv_writer.writerow(row)

            logging.info(f"[DEV:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in dev_metrics.items()))

            dev_metrics_by_dataset.append({
                "name": name,
                **dev_metrics
            })
            
            
        elif model_stage == 'personality':
            name = 'fiv2'
            loader = dev_loaders[name]
            d_loss, d_acc, d_ccc = run_per_eval(
                    model, loader, criterion, device, model_stage
                )
            dev_means.extend([d_acc, d_ccc])
            dev_metrics = {
                    "loss": d_loss, "acc": d_acc,
                    "ccc": d_ccc, "mean": np.mean([d_acc, d_ccc])
                }

            if csv_writer:
                row = ["dev", epoch, name, *dev_metrics.values()]
                while len(row) < 9:  # выравнивание колонок
                    row.append("")
                csv_writer.writerow(row)

            logging.info(f"[DEV:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in dev_metrics.items()))

            dev_metrics_by_dataset.append({
                "name": name,
                **dev_metrics
            })

        elif model_stage == 'fusion':
            name = 'cmu_mosei'
            loader = dev_loaders[name]
            d_loss, d_uar, d_mf1 = run_emo_eval(
                    model, loader, criterion, device, model_stage
                )
            dev_means.extend([d_uar, d_mf1])
            dev_metrics = {
                    "loss": d_loss, "uar": d_uar,
                    "mf1": d_mf1, "mean": np.mean([d_uar, d_mf1])
                }
            
            dev_metrics_by_dataset.append({
                "name": name,
                **dev_metrics
            })
            
            logging.info(f"[DEV:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in dev_metrics.items()))
            
            name = 'fiv2'
            loader = dev_loaders[name]
            d_loss, d_acc, d_ccc = run_per_eval(
                    model, loader, criterion, device, model_stage
                )
            dev_means.extend([d_acc, d_ccc])
            dev_metrics = {
                    "loss": d_loss, "acc": d_acc,
                    "ccc": d_ccc, "mean": np.mean([d_acc, d_ccc])
                }
            
            logging.info(f"[DEV:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in dev_metrics.items()))

            dev_metrics_by_dataset.append({
                "name": name,
                **dev_metrics
            })

        mean_dev = np.mean(dev_means)

        # --- TEST ---
        test_means = []
        test_metrics_by_dataset = []

        if model_stage == 'emotion':
            name = 'cmu_mosei'
            loader = test_loaders[name]
            t_loss, t_uar, t_mf1 = run_emo_eval(
                    model, loader, criterion, device, model_stage
                )
            test_means.extend([t_uar, t_mf1])
            test_metrics = {
                    "loss": t_loss, "uar": t_uar,
                    "mf1": t_mf1, "mean": np.mean([t_uar, t_mf1])
                }
            
            if csv_writer:
                row = ["test", epoch, name, *test_metrics.values()]
                while len(row) < 9:  # выравнивание колонок
                    row.append("")
                csv_writer.writerow(row)

            logging.info(f"[TEST:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in test_metrics.items()))

            test_metrics_by_dataset.append({
                "name": name,
                **test_metrics
            })
            
            
        elif model_stage == 'personality':
            name = 'fiv2'
            loader = test_loaders[name]
            t_loss, t_acc, t_ccc = run_per_eval(
                    model, loader, criterion, device, model_stage
                )
            test_means.extend([t_acc, t_ccc])
            test_metrics = {
                    "loss": t_loss, "acc": t_acc,
                    "ccc": t_ccc, "mean": np.mean([t_acc, t_ccc])
                }

            if csv_writer:
                row = ["test", epoch, name, *test_metrics.values()]
                while len(row) < 9:  # выравнивание колонок
                    row.append("")
                csv_writer.writerow(row)

            logging.info(f"[TEST:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in test_metrics.items()))

            test_metrics_by_dataset.append({
                "name": name,
                **test_metrics
            })
        elif model_stage == 'fusion':
            name = 'cmu_mosei'
            loader = test_loaders[name]
            t_loss, t_uar, t_mf1 = run_emo_eval(
                    model, loader, criterion, device, model_stage
                )
            test_means.extend([t_uar, t_mf1])
            test_metrics = {
                    "loss": t_loss, "uar": t_uar,
                    "mf1": t_mf1, "mean": np.mean([t_uar, t_mf1])
                }
            
            test_metrics_by_dataset.append({
                "name": name,
                **test_metrics
            })
            
            logging.info(f"[TEST:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in test_metrics.items()))
            
            name = 'fiv2'
            loader = test_loaders[name]
            t_loss, t_acc, t_ccc = run_per_eval(
                    model, loader, criterion, device, model_stage
                )
            test_means.extend([t_acc, t_ccc])
            test_metrics = {
                    "loss": t_loss, "acc": t_acc,
                    "ccc": t_ccc, "mean": np.mean([t_acc, t_ccc])
                }
            
            logging.info(f"[TEST:{name}] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in test_metrics.items()))

            test_metrics_by_dataset.append({
                "name": name,
                **test_metrics
            })
        
        mean_test = np.mean(test_means)

        if config.opt_set == "test":
            scheduler.step(mean_test)
            mean_target = mean_test
        else:
            scheduler.step(mean_dev)
            mean_target = mean_dev

        if mean_target > best_dev_mean:
            best_dev_mean = mean_target
            patience_counter = 0
            best_dev_metrics = {
                "mean": mean_dev,
                "by_dataset": dev_metrics_by_dataset
            }
            best_test_metrics = {
                "mean": np.mean([ds["mean"] for ds in test_metrics_by_dataset]),
                "by_dataset": test_metrics_by_dataset
            }

            if config.save_best_model:
                dev_str = f"{mean_target:.4f}".replace(".", "_")
                model_path = os.path.join(checkpoint_dir, f"best_model_dev.pt")
                # model_path = os.path.join(checkpoint_dir, f"best_model_dev_{dev_str}_epoch_{epoch}.pt")
                torch.save(model.state_dict(), model_path)
                logging.info(f"💾 Модель сохранена по лучшему dev (эпоха {epoch}): {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"Early stopping: {max_patience} эпох без улучшения.")
                break

    logging.info("Тренировка завершена. Все split'ы обработаны!")

    if csv_file:
        csv_file.close()

    return best_dev_mean, best_dev_metrics, best_test_metrics
