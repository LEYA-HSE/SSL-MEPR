# coding: utf-8
from __future__ import annotations

import os, logging
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lion_pytorch import Lion

from utils.schedulers import SmartScheduler
from utils.logger_setup import color_metric, color_split
from utils.measures import mf1, uar, acc_func, ccc
from utils.losses import MultiTaskLossWithNaN
from models.models import MultiModalFusionModelWithAblation, SingleTaskSlimModel, SingleTaskAsymModel


# ─────────────────────────────── utils ────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_matrix(matrix):
    threshold1 = 1 - 1 / 7
    threshold2 = 1 / 7
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


def drop_domains_in_batch(batch: dict, cfg):
    """Zero-out cross-domain logits according to cfg flags."""
    if cfg.single_task:
        if getattr(cfg, "drop_personality_domain", False) and "personality_scores" in batch:
            for mod in batch["personality_scores"]:
                batch["personality_scores"][mod] = None
        if getattr(cfg, "drop_emotion_domain", False) and "emotion_logits" in batch:
            for mod in batch["emotion_logits"]:
                batch["emotion_logits"][mod] = None
    return batch


# ─────────────────────────── evaluation ────────────────────────────
@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   cfg) -> Dict[str, float]:
    """Compute metrics over the entire loader."""
    model.eval()
    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = drop_domains_in_batch(batch, cfg)
        out = model(batch)

        # Emotion
        logits_e = out["emotion_logits"]
        if logits_e is not None:
            y_e = batch["labels"]["emotion"]
            valid_e = ~torch.isnan(y_e).all(dim=1)
            if valid_e.any():
                p, t = process_predictions(logits_e[valid_e], y_e[valid_e])
                emo_preds.extend(p)
                emo_tgts.extend(t)

        # Personality
        preds_p = out["personality_scores"]
        if preds_p is not None:
            preds_p = preds_p.cpu()
            y_p = batch["labels"]["personality"]
            valid_p = ~torch.isnan(y_p).all(dim=1)
            if valid_p.any():
                pkl_preds.append(preds_p[valid_p].numpy())
                pkl_tgts.append(y_p[valid_p].numpy())

    metrics: dict[str, float] = {}
    if emo_tgts:
        tgt, prd = np.asarray(emo_tgts), np.asarray(emo_preds)
        metrics["mF1"] = mf1(tgt, prd)
        metrics["mUAR"] = uar(tgt, prd)
    if pkl_tgts:
        tgt, prd = np.vstack(pkl_tgts), np.vstack(pkl_preds)
        metrics["ACC"] = acc_func(tgt, prd)
        metrics["CCC"] = ccc(tgt, prd)
    return metrics


def log_and_aggregate_split(name: str,
                            loaders: dict[str, DataLoader],
                            model: torch.nn.Module,
                            device: torch.device,
                            cfg) -> dict[str, float]:
    """
    Log metrics for each dataset in the split and compute aggregated means.
    """
    logging.info(f"—— {name} metrics ——")
    all_metrics: dict[str, float] = {}

    for ds_name, loader in loaders.items():
        m = evaluate_epoch(model, loader, device, cfg)
        all_metrics.update({f"{k}_{ds_name}": v for k, v in m.items()})
        msg = " · ".join(color_metric(k, v) for k, v in m.items())
        logging.info(f"[{color_split(name)}:{ds_name}] {msg}")

    mf1s = [v for k, v in all_metrics.items() if k.startswith("mF1_")]
    uars = [v for k, v in all_metrics.items() if k.startswith("mUAR_")]
    accs = [v for k, v in all_metrics.items() if k.startswith("ACC_")]
    cccs = [v for k, v in all_metrics.items() if k.startswith("CCC_")]

    if mf1s and uars:
        all_metrics["mean_emo"] = float(np.mean(mf1s + uars))
    if accs and cccs:
        all_metrics["mean_pkl"] = float(np.mean(accs + cccs))

    if "mean_emo" in all_metrics or "mean_pkl" in all_metrics:
        summary_parts = []
        if "mean_emo" in all_metrics:
            summary_parts.append(color_metric("mean_emo", all_metrics["mean_emo"]))
        if "mean_pkl" in all_metrics:
            summary_parts.append(color_metric("mean_pkl", all_metrics["mean_pkl"]))
        logging.info(f"{name} Summary | " + " ".join(summary_parts))

    return all_metrics


# ────────────────────────── main train() ──────────────────────────
def train(cfg,
          mm_loader: DataLoader,
          dev_loaders: dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):
    """
    cfg          – config object
    mm_loader    – multimodal train DataLoader
    dev_loaders  – optional validation loaders dict
    test_loaders – optional test loaders dict
    """
    seed_everything(cfg.random_seed)
    device = cfg.device

    # ─── Single-task routing (4 options) ─────────────────────────────
    if cfg.single_task:
        #   0: Emo+PKL → Emo     1: Emo → Emo
        #   2: Emo+PKL → PKL     3: PKL → PKL
        slice_map = [("both", "emo"), ("emo", "emo"),
                     ("both", "pkl"), ("pkl", "pkl")]
        try:
            feature_slice, task_target = slice_map[cfg.single_task_id]
        except IndexError:
            raise ValueError("single_task_id must be 0-3")

        # disable logits of the other domain when needed
        cfg.drop_personality_domain = feature_slice == "emo"
        cfg.drop_emotion_domain     = feature_slice == "pkl"
    else:
        feature_slice = task_target = None
        cfg.drop_personality_domain = cfg.drop_emotion_domain = False

    # ─── Ablation config for multi-modal model ────────────────────
    ablation_config = {}
    if not cfg.single_task:
        modality_combinations = [
            [],  # 0: use all modalities

            # Single modalities
            ["audio"],      # 1
            ["text"],       # 2
            ["scene"],      # 3
            ["face"],       # 4
            ["body"],       # 5

            # Two modalities
            ["audio", "text"],      # 6
            ["audio", "scene"],     # 7
            ["audio", "face"],      # 8
            ["audio", "body"],      # 9
            ["text", "scene"],      # 10
            ["text", "face"],       # 11
            ["text", "body"],       # 12
            ["scene", "face"],      # 13
            ["scene", "body"],      # 14
            ["face", "body"],       # 15

            # Three modalities
            ["audio", "text", "scene"],    # 16
            ["audio", "text", "face"],     # 17
            ["audio", "text", "body"],     # 18
            ["audio", "scene", "face"],    # 19
            ["audio", "scene", "body"],    # 20
            ["audio", "face", "body"],     # 21
            ["text", "scene", "face"],     # 22
            ["text", "scene", "body"],     # 23
            ["text", "face", "body"],      # 24
            ["scene", "face", "body"],     # 25
        ]

        components = [
            -1,
            "disable_graph_attn",
            "disable_cross_attn",
            "disable_emo_logit_proj",
            "disable_pkl_logit_proj",
            "disable_guide_emo",
            "disable_guide_pkl",
        ]

        ablation_config = (
            {
                "disabled_modalities": modality_combinations[cfg.id_ablation_type_by_modality],
                components[cfg.id_ablation_type_by_component]: True,
            }
            if components[cfg.id_ablation_type_by_component] != -1
            else {
                "disabled_modalities": modality_combinations[cfg.id_ablation_type_by_modality]
            }
        )

    # ─── Model selection ──────────────────────────────────────────
    if cfg.single_task:
        # model = SingleTaskSlimModel(
        model = SingleTaskAsymModel(
            feature_slice=feature_slice,   # "both" | "emo" | "pkl"
            target=task_target,            # "emo" | "pkl"
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_transformer_heads,
            dropout=cfg.dropout,
            emo_out_dim=7,
            pkl_out_dim=5,
            device=device,
        ).to(device)

        if task_target == "emo":
            cfg.weight_emotion, cfg.weight_pers, cfg.selection_metric = 1.0, 0.0, "mean_emo"
        else:
            cfg.weight_emotion, cfg.weight_pers, cfg.selection_metric = 0.0, 1.0, "mean_pkl"

    else:
        model = MultiModalFusionModelWithAblation(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_transformer_heads,
            dropout=cfg.dropout,
            emo_out_dim=7,
            pkl_out_dim=5,
            device=device,
            ablation_config=ablation_config,
        ).to(device)

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    logging.info(f"⚙️ Optimizer: {cfg.optimizer}, learning rate: {cfg.lr}")

    # --- Scheduler ---
    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )

    # --- Loss ---
    criterion = MultiTaskLossWithNaN(
        weight_emotion=cfg.weight_emotion,
        weight_personality=cfg.weight_pers,
        emo_weights=(
            torch.FloatTensor(
                [5.890161, 7.534918, 11.228363, 27.722221, 1.3049748, 5.6189237, 26.639517]
            ).to(device)
            if cfg.flag_emo_weight else None
        ),
        personality_loss_type=cfg.pers_loss_type,
        emotion_loss_type=cfg.emotion_loss_type
    )

    best_dev, best_test = {}, {}
    best_score = -float("inf")
    patience_counter = 0

    # ── 1. Epochs ──────────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCH {epoch + 1}/{cfg.num_epochs} ═══")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds_emo, total_targets_emo = [], []
        total_preds_per, total_targets_per = [], []

        for batch in tqdm(mm_loader):
            if batch is None:
                continue

            batch = drop_domains_in_batch(batch, cfg)

            emo_labels = batch["labels"]["emotion"].to(device)
            per_labels = batch["labels"]["personality"].to(device)

            valid_emo = ~torch.isnan(emo_labels).all(dim=1)
            valid_per = ~torch.isnan(per_labels).all(dim=1)

            outputs = model(batch)
            loss = criterion(outputs, {
                "emotion": emo_labels,
                "personality": per_labels,
                "valid_emo": valid_emo,
                "valid_per": valid_per
            })
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(batch_level=True)

            bs = emo_labels.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            if outputs['emotion_logits'] is not None and valid_emo.any():
                preds_emo, targets_emo = process_predictions(
                    outputs['emotion_logits'][valid_emo],
                    emo_labels[valid_emo]
                )
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(targets_emo)

            if outputs['personality_scores'] is not None and valid_per.any():
                preds_per = outputs['personality_scores'][valid_per]
                targets_per = per_labels[valid_per]
                total_preds_per.extend(preds_per.cpu().detach().numpy().tolist())
                total_targets_per.extend(targets_per.cpu().detach().numpy().tolist())

        # --- train metrics ---
        train_loss = total_loss / max(1, total_samples)

        # ===== EMOTIONS =====
        if total_targets_emo:
            tgt_emo = np.asarray(total_targets_emo)
            prd_emo = np.asarray(total_preds_emo)
            mF1_train = mf1(tgt_emo, prd_emo)
            mUAR_train = uar(tgt_emo, prd_emo)
            mean_emo_train = float(np.mean([mF1_train, mUAR_train]))
        else:
            mF1_train = mUAR_train = mean_emo_train = float('nan')

        # ===== PERSONALITY =====
        if total_targets_per:
            t_per = np.asarray(total_targets_per)
            p_per = np.asarray(total_preds_per)

            acc_train = acc_func(t_per, p_per)

            ccc_vals = []
            for i in range(t_per.shape[1]):
                mask = ~np.isnan(t_per[:, i])
                if mask.sum() == 0:
                    continue
                ccc_vals.append(ccc(t_per[mask, i], p_per[mask, i]))
            ccc_train = float(np.mean(ccc_vals)) if ccc_vals else float('nan')

            mean_pkl_train = np.nanmean([acc_train, ccc_train])
        else:
            acc_train = ccc_train = mean_pkl_train = float('nan')

        logging.info(
            f"[{color_split('TRAIN')}] "
            f"Loss={train_loss:.4f} | "
            f"EMO: UAR={mUAR_train:.4f}, MF1={mF1_train:.4f}, MEAN_EMO={mean_emo_train:.4f} | "
            f"PKL: ACC={acc_train:.4f}, CCC={ccc_train:.4f}, MEAN_PKL={mean_pkl_train:.4f}"
        )

        # ── Evaluation ──
        cur_dev = log_and_aggregate_split("Dev", dev_loaders, model, device, cfg)
        cur_test = log_and_aggregate_split("Test", test_loaders, model, device, cfg) if test_loaders else {}

        cur_eval = cur_dev if cfg.early_stop_on == "dev" else cur_test

        mean_emo = cur_eval.get("mean_emo")
        mean_pkl = cur_eval.get("mean_pkl", 0.0)

        # ── selection metric depending on mode ──
        if cfg.single_task:
            metric_val = cur_eval["mean_emo"] if task_target == "emo" else cur_eval["mean_pkl"]
        else:
            if mean_emo is not None and mean_pkl is not None:
                metric_val = 0.5 * (mean_emo + mean_pkl)
            else:
                metric_val = mean_emo if mean_emo is not None else mean_pkl

        scheduler.step(metric_val)

        improved = metric_val > best_score

        if improved:
            best_score = metric_val
            best_dev = cur_dev
            best_test = cur_test
            patience_counter = 0

            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            emo_str = f"{mean_emo:.4f}" if mean_emo is not None else "NA"
            pkl_str = f"{mean_pkl:.4f}" if mean_pkl is not None else "NA"

            ckpt_path = Path(cfg.checkpoint_dir) / f"best_ep{epoch + 1}_emo{emo_str}_pkl{pkl_str}.pt"

            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"✔ Best model saved: {ckpt_path.name}")
        else:
            patience_counter += 1
            logging.warning(f"No improvement — patience {patience_counter}/{cfg.max_patience}")
            if patience_counter >= cfg.max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    return best_dev, best_test
