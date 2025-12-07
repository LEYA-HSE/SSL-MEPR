# train.py
# coding: utf-8
import logging
import os
import shutil
import datetime
# import whisper
import toml
# os.environ["HF_HOME"] = "models"

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from training.train_utils_video import (
    make_dataset_and_loader,
    train_once
)
from data_loading.feature_extractor import PretrainedImageEmbeddingExtractor

def main():
    # CONFIG_PATH = 'config.toml'
    CONFIG_PATH = 'clip_config_transformer_mamba.toml'
    #  Грузим конфиг
    base_config = ConfigLoader(CONFIG_PATH)

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f"results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # Настраиваем logging
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)

    #  Грузим конфиг
    base_config.show_config()

    shutil.copy(CONFIG_PATH, os.path.join(results_dir, "config_copy.toml"))
    #  Файл, куда будет писать наш жадный поиск
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    audio_feature_extractor= None
    text_feature_extractor = None
    image_feature_extractor = PretrainedImageEmbeddingExtractor(base_config)

    # Инициализируем Whisper-модель один раз
    logging.info(f"Инициализация Whisper: модель={base_config.whisper_model}, устройство={base_config.whisper_device}")
    whisper_model = None

    # Делаем датасеты/лоадеры
    # Общий train_loader
    # _, train_loader = make_dataset_and_loader(base_config, "train", audio_feature_extractor, text_feature_extractor, image_feature_extractor, image_feature_extractor, whisper_model)

    # Раздельные dev/test
    dev_loaders = {}
    test_loaders = {}
    train_loaders = {}

    for dataset_name in base_config.datasets:
        _, train_loader = make_dataset_and_loader(base_config, "train",  audio_feature_extractor, text_feature_extractor, image_feature_extractor, whisper_model, only_dataset=dataset_name)
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], task=base_config.datasets[dataset_name]["task"], split="dev")):
            _, dev_loader = make_dataset_and_loader(base_config, "dev",  audio_feature_extractor, text_feature_extractor, image_feature_extractor, whisper_model, only_dataset=dataset_name)
        else:
            _, dev_loader = make_dataset_and_loader(base_config, "val",  audio_feature_extractor, text_feature_extractor, image_feature_extractor, whisper_model, only_dataset=dataset_name)        
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], task=base_config.datasets[dataset_name]["task"], split="test")):
            _, test_loader = make_dataset_and_loader(base_config, "test",  audio_feature_extractor, text_feature_extractor, image_feature_extractor, whisper_model, only_dataset=dataset_name)
        else:
            test_loader = dev_loader

        # train_loaders.append((dataset_name, train_loader))
        # dev_loaders.append((dataset_name, dev_loader))
        # test_loaders.append((dataset_name, test_loader))
        train_loaders[dataset_name] = train_loader
        dev_loaders[dataset_name] = dev_loader
        test_loaders[dataset_name] = test_loader

    if base_config.prepare_only:
        logging.info("== Режим prepare_only: только подготовка данных, без обучения ==")
        return

    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config       = base_config,
            train_loader      = train_loaders,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            default_values    = default_values,
            csv_prefix        = csv_prefix,
            model_stage       = base_config.model_stage
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config       = base_config,
            train_loader      = train_loaders,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            csv_prefix        = csv_prefix,
            model_stage       = base_config.model_stage

        )

    elif base_config.search_type == "none":
        logging.info("== Режим одиночной тренировки (без поиска параметров) ==")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

        train_once(
            config           = base_config,
            train_loaders    = train_loaders,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
            metrics_csv_path = csv_file_path,
            model_stage      = base_config.model_stage
        )

    else:
        raise ValueError(f"⛔️ Неверное значение search_type в конфиге: '{base_config.search_type}'. Используй 'greedy', 'exhaustive' или 'none'.")


if __name__ == "__main__":
    main()
