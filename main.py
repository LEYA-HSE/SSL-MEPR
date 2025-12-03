# coding: utf-8
import logging
import os
import shutil
import datetime
import requests
import toml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from data_loading.dataset_builder import make_dataset_and_loader

from modalities.video.feature_extractor import PretrainedImageEmbeddingExtractor
from modalities.audio.feature_extractor import PretrainedAudioEmbeddingExtractor
from modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor

from training.train import train


# ───────────────────── optional .env load ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ───────────────────── Telegram helper ─────────────────────
def _notify_telegram(text: str, enabled: bool = True) -> bool:
    """Send a Telegram message if enabled and TELEGRAM_BOT_TOKEN/CHAT_ID are set.
       Returns True/False and logs the reason when skipped."""
    if not enabled:
        logging.info("TG notify: disabled by config")
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("TG notify: skipped (no TELEGRAM_BOT_TOKEN/CHAT_ID)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        # Log Telegram response
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}
        if r.ok and isinstance(payload, dict) and payload.get("ok"):
            logging.info("TG notify: sent")
            return True
        logging.warning(f"TG notify: API error {r.status_code} -> {payload}")
        return False
    except Exception as e:
        logging.warning(f"TG notify failed: {e}")
        return False


def main():
    # ──────────────────── 1) Config & directories ────────────────────
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    base_config.checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # ──────────────────── 2) Logging ────────────────────────────
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    use_tg = base_config.use_telegram
    logging.info(
        f"use_telegram = {use_tg}  "
        f"(env token={bool(os.getenv('TELEGRAM_BOT_TOKEN'))}, "
        f"chat={bool(os.getenv('TELEGRAM_CHAT_ID'))})"
    )

    # Startup ping to confirm connectivity
    _notify_telegram(f"🚀 Start: <b>{model_name}</b>\n📁 {results_dir}", enabled=use_tg)

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    # ──────────────────── 3) Feature extractors ────────────────
    logging.info("🔧 Initializing modalities...")

    image_feature_extractor = PretrainedImageEmbeddingExtractor(device=base_config.device)
    logging.info("🖼️ Image extractor initialized")

    audio_feature_extractor = PretrainedAudioEmbeddingExtractor(device=base_config.device)
    logging.info("🔊 Audio extractor initialized")

    text_feature_extractor = PretrainedTextEmbeddingExtractor(device=base_config.device)
    logging.info("📄 Text extractor initialized")

    modality_processors = {
        "body": image_feature_extractor.processor,
        "face": image_feature_extractor.processor,
        "audio": audio_feature_extractor.processor,
        "text": None,
        "scene": image_feature_extractor.processor,
    }

    modality_extractors = {
        "body": image_feature_extractor,
        "face": image_feature_extractor,
        "audio": audio_feature_extractor,
        "text": text_feature_extractor,
        "scene": image_feature_extractor,
    }

    # ──────────────────── 4) Dataloaders ────────────────────────────
    train_loaders, dev_loaders, test_loaders = {}, {}, {}

    for dataset_name in tqdm(base_config.datasets, desc="Dataloaders", leave=False):
        logging.info(f"📦 Loading dataset: {dataset_name}")

        # train
        _, train_loader = make_dataset_and_loader(
            base_config,
            "train",
            modality_processors,
            modality_extractors,
            only_dataset=dataset_name,
        )

        # dev / val
        dev_split = (
            "dev"
            if os.path.exists(
                base_config.datasets[dataset_name]["csv_path"].format(
                    base_dir=base_config.datasets[dataset_name]["base_dir"], split="dev"
                )
            )
            else "val"
        )

        _, dev_loader = make_dataset_and_loader(
            base_config,
            dev_split,
            modality_processors,
            modality_extractors,
            only_dataset=dataset_name,
        )

        # test
        test_split_path = base_config.datasets[dataset_name]["csv_path"].format(
            base_dir=base_config.datasets[dataset_name]["base_dir"], split="test"
        )
        if os.path.exists(test_split_path):
            _, test_loader = make_dataset_and_loader(
                base_config,
                "test",
                modality_processors,
                modality_extractors,
                only_dataset=dataset_name,
            )
        else:
            test_loader = dev_loader  # fallback

        train_loaders[dataset_name] = train_loader
        dev_loaders[dataset_name] = dev_loader
        test_loaders[dataset_name] = test_loader

    # ──────────────────── 5) prepare_only ──────────────────────────
    if base_config.prepare_only:
        logging.info("== prepare_only mode: data preparation only, no training ==")
        return

    train_datasets = []
    for ds_name in base_config.datasets:
        ds, loader = make_dataset_and_loader(
            base_config,
            "train",
            modality_processors,
            modality_extractors,
            only_dataset=ds_name,
        )
        train_datasets.append(ds)

    # ───────────────────── merge train datasets ──────────────────────
    union_train_ds = ConcatDataset(train_datasets)
    # Reuse collate_fn from any of the original loaders (identical across datasets)
    sample_loader = next(iter(train_loaders.values()))
    union_train_loader = DataLoader(
        union_train_ds,
        batch_size=base_config.batch_size,
        shuffle=True,
        num_workers=base_config.num_workers,
        collate_fn=sample_loader.collate_fn,
    )

    # ──────────────────── 6) Hparam search / single run ──
    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config=base_config,
            train_loader=union_train_loader,
            dev_loader=dev_loaders,
            test_loader=test_loaders,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
            default_values=default_values,
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config=base_config,
            train_loader=union_train_loader,
            dev_loader=dev_loaders,
            test_loader=test_loaders,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: exhaustive search finished\n📁 {results_dir}",
            enabled=use_tg,
        )

    elif base_config.search_type == "none":
        logging.info("== Single training run (no hyperparameter search) ==")
        train(
            cfg=base_config,
            mm_loader=union_train_loader,
            dev_loaders=dev_loaders,
            test_loaders=test_loaders,
        )

    else:
        raise ValueError(
            f"⛔️ Invalid search_type in config: '{base_config.search_type}'. "
            f"Use 'greedy', 'exhaustive', or 'none'."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Crash notification always goes out
        _notify_telegram(
            f"❌ Crash: <code>{type(e).__name__}</code>\n{e}",
            enabled=True,
        )
        raise
