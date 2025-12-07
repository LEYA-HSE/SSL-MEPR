# data_loading/feature_extractor.py

import torch
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    AutoModelForAudioClassification,
    Wav2Vec2Processor
)
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from data_loading.pretrained_extractors import EmotionModel, get_model_mamba, Mamba

from transformers import CLIPModel
from collections import deque

class PretrainedAudioEmbeddingExtractor:
    """
    Извлекает эмбеддинги из аудио, используя модель (например 'amiriparian/ExHuBERT'),
    с учётом pooling, нормализации и т.д.
    """

    def __init__(self, config):
        """
        Ожидается, что в config есть поля:
         - audio_model_name (str)         : название модели (ExHuBERT и т.п.)
         - emb_device (str)              : "cpu" или "cuda"
         - audio_pooling (str | None)    : "mean", "cls", "max", "min", "last" или None (пропустить пуллинг)
         - emb_normalize (bool)          : делать ли L2-нормализацию выхода
         - max_audio_frames (int)        : ограничение длины по временной оси (если 0 - не ограничивать)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.audio_model_name
        self.pooling = config.audio_pooling       # может быть None
        self.normalize_output = config.emb_normalize
        self.max_audio_frames = getattr(config, "max_audio_frames", 0)
        self.audio_classifier_checkpoint = config.audio_classifier_checkpoint

        # Инициализируем processor и audio_embedder
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.audio_embedder = EmotionModel.from_pretrained(self.model_name).to(self.device)

        # Загружаем модель
        self.classifier_model = self.load_classifier_model_from_checkpoint(self.audio_classifier_checkpoint)


    def extract(self, waveform: torch.Tensor, sample_rate=16000):
        """
        Извлекает эмбеддинги из аудиоданных.

        :param waveform: Тензор формы (T).
        :param sample_rate: Частота дискретизации (int).
        :return: Тензоры:
            вернётся (B, classes), (B, sequence_length, hidden_dim).
        """

        embeddings = self.process_audio(waveform, sample_rate)
        tensor_emb = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        lengths = [tensor_emb.shape[1]]

        with torch.no_grad():
            logits, hidden = self.classifier_model(tensor_emb, lengths, with_features=True)

            # Если pooling=None => вернём (B, seq_len, hidden_dim)
            if hidden.dim() == 3:
                if self.pooling is None:
                    emb = hidden
                else:
                    if self.pooling == "mean":
                        emb = hidden.mean(dim=1)
                    elif self.pooling == "cls":
                        emb = hidden[:, 0, :]
                    elif self.pooling == "max":
                        emb, _ = hidden.max(dim=1)
                    elif self.pooling == "min":
                        emb, _ = hidden.min(dim=1)
                    elif self.pooling == "last":
                        emb = hidden[:, -1, :]
                    elif self.pooling == "sum":
                        emb = hidden.sum(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
            else:
                # На всякий случай, если получилось (B, hidden_dim)
                emb = hidden

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return logits, emb

    def process_audio(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        inputs = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.audio_embedder(input_values)
            embeddings = outputs

        return embeddings.detach().cpu().numpy()

    def load_classifier_model_from_checkpoint(self, checkpoint_path):
        if checkpoint_path == "best_audio_model.pt":
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            exp_params = checkpoint['exp_params']
            classifier_model = get_model_mamba(exp_params).to(self.device)
            classifier_model.load_state_dict(checkpoint['model_state_dict'])
        elif checkpoint_path == "best_audio_model_2.pt":
            model_params = {
                "input_size": 1024,
                "d_model": 256,
                "num_layers": 2,
                "num_classes": 7,
                "dropout": 0.2
            }
            classifier_model = get_model_mamba(model_params).to(self.device)
            classifier_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        classifier_model.eval()
        return classifier_model

class AudioEmbeddingExtractor:
    """
    Извлекает эмбеддинги из аудио, используя модель (например 'amiriparian/ExHuBERT'),
    с учётом pooling, нормализации и т.д.
    """

    def __init__(self, config):
        """
        Ожидается, что в config есть поля:
         - audio_model_name (str)         : название модели (ExHuBERT и т.п.)
         - emb_device (str)              : "cpu" или "cuda"
         - audio_pooling (str | None)    : "mean", "cls", "max", "min", "last" или None (пропустить пуллинг)
         - emb_normalize (bool)          : делать ли L2-нормализацию выхода
         - max_audio_frames (int)        : ограничение длины по временной оси (если 0 - не ограничивать)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.audio_model_name
        self.pooling = config.audio_pooling       # может быть None
        self.normalize_output = config.emb_normalize
        # self.max_audio_frames = getattr(config, "max_audio_frames", 0)
        self.max_audio_frames = config.sample_rate * config.wav_length


        # Попробуем загрузить feature_extractor (не у всех моделей доступен)
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            logging.info(f"[Audio] Using AutoFeatureExtractor for '{self.model_name}'")
        except Exception as e:
            self.feature_extractor = None
            logging.warning(f"[Audio] No built-in FeatureExtractor found. Model={self.model_name}. Error: {e}")

        # Загружаем модель
        # Если у модели нет head-классификации, бывает достаточно AutoModel
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                output_hidden_states=True   # чтобы точно был last_hidden_state
            ).to(self.device)
            logging.info(f"[Audio] Loaded AutoModel with output_hidden_states=True: {self.model_name}")
        except Exception as e:
            logging.warning(f"[Audio] Fallback to AudioClassification model. Reason: {e}")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                output_hidden_states=True
            ).to(self.device)

    def extract(self, waveform_batch: torch.Tensor, sample_rate=16000):
        """
        Извлекает эмбеддинги из аудиоданных.

        :param waveform_batch: Тензор формы (B, T) или (B, 1, T).
        :param sample_rate: Частота дискретизации (int).
        :return: Тензор:
          - если pooling != None, будет (B, hidden_dim)
          - если pooling == None и last_hidden_state имел форму (B, seq_len, hidden_dim),
            вернётся (B, seq_len, hidden_dim).
        """

        # Если пришло (B, 1, T), уберём ось "1"
        if waveform_batch.dim() == 3 and waveform_batch.shape[1] == 1:
            waveform_batch = waveform_batch.squeeze(1)  # -> (B, T)

        # Усечение по времени, если нужно
        if self.max_audio_frames > 0 and waveform_batch.shape[1] > self.max_audio_frames:
            waveform_batch = waveform_batch[:, :self.max_audio_frames]

        # Если есть feature_extractor - используем
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                waveform_batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_audio_frames if self.max_audio_frames > 0 else None
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(input_values=inputs["input_values"])
        else:
            # Иначе подадим напрямую "input_values" на модель
            inputs = {"input_values": waveform_batch.to(self.device)}
            outputs = self.model(**inputs)

        # Теперь outputs может быть BaseModelOutput (с last_hidden_state, hidden_states, etc.)
        # Или SequenceClassifierOutput (с logits), если это модель-классификатор
        if hasattr(outputs, "last_hidden_state"):
            # (B, seq_len, hidden_dim)
            hidden = outputs.last_hidden_state
            # logging.debug(f"[Audio] last_hidden_state shape: {hidden.shape}")
        elif hasattr(outputs, "logits"):
            # logits: (B, num_labels)
            # Для пуллинга по "seq_len" притворимся, что seq_len=1
            hidden = outputs.logits.unsqueeze(1)  # (B,1,num_labels)
            logging.debug(f"[Audio] Found logits shape: {outputs.logits.shape} => hidden={hidden.shape}")
        else:
            # Модель может сразу возвращать тензор
            hidden = outputs

        # Если у нас 2D-тензор (B, hidden_dim), значит всё уже спулено
        if hidden.dim() == 2:
            emb = hidden
        elif hidden.dim() == 3:
            # (B, seq_len, hidden_dim)
            if self.pooling is None:
                # Возвращаем как есть
                emb = hidden
            else:
                # Выполним пуллинг
                if self.pooling == "mean":
                    emb = hidden.mean(dim=1)
                elif self.pooling == "cls":
                    emb = hidden[:, 0, :]  # [B, hidden_dim]
                elif self.pooling == "max":
                    emb, _ = hidden.max(dim=1)
                elif self.pooling == "min":
                    emb, _ = hidden.min(dim=1)
                elif self.pooling == "last":
                    emb = hidden[:, -1, :]
                else:
                    emb = hidden.mean(dim=1)  # на всякий случай fallback
        else:
            # На всякий: если ещё какая-то форма
            raise ValueError(f"[Audio] Unexpected hidden shape={hidden.shape}, pooling={self.pooling}")

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return emb


class TextEmbeddingExtractor:
    """
    Извлекает эмбеддинги из текста (например 'jinaai/jina-embeddings-v3'),
    с учётом pooling (None, mean, cls, и т.д.), нормализации и усечения.
    """

    def __init__(self, config):
        """
        Параметры в config:
         - text_model_name (str)
         - emb_device (str)
         - text_pooling (str | None)
         - emb_normalize (bool)
         - max_tokens (int)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.text_model_name
        self.pooling = config.text_pooling        # может быть None
        self.normalize_output = config.emb_normalize
        self.max_tokens = config.max_tokens

        # trust_remote_code=True нужно для моделей вроде jina
        logging.info(f"[Text] Loading tokenizer for {self.model_name} with trust_remote_code=True")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        logging.info(f"[Text] Loading model for {self.model_name} with trust_remote_code=True")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            output_hidden_states=True,    # хотим иметь last_hidden_state
            force_download=False
        ).to(self.device)

    def extract(self, text_list):
        """
        :param text_list: список строк (или одна строка)
        :return: тензор (B, hidden_dim) или (B, seq_len, hidden_dim), если pooling=None
        """

        if isinstance(text_list, str):
            text_list = [text_list]

        inputs = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Обычно у AutoModel last_hidden_state.shape = (B, seq_len, hidden_dim)
            hidden = outputs.last_hidden_state
            # logging.debug(f"[Text] last_hidden_state shape: {hidden.shape}")

            # Если pooling=None => вернём (B, seq_len, hidden_dim)
            if hidden.dim() == 3:
                if self.pooling is None:
                    emb = hidden
                else:
                    if self.pooling == "mean":
                        emb = hidden.mean(dim=1)
                    elif self.pooling == "cls":
                        emb = hidden[:, 0, :]
                    elif self.pooling == "max":
                        emb, _ = hidden.max(dim=1)
                    elif self.pooling == "min":
                        emb, _ = hidden.min(dim=1)
                    elif self.pooling == "last":
                        emb = hidden[:, -1, :]
                    elif self.pooling == "sum":
                        emb = hidden.sum(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
            else:
                # На всякий случай, если получилось (B, hidden_dim)
                emb = hidden

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return emb

class PretrainedTextEmbeddingExtractor:
    """
    Извлекает эмбеддинги из текста (например 'jinaai/jina-embeddings-v3'),
    с учётом pooling (None, mean, cls, и т.д.), нормализации и усечения.
    """

    def __init__(self, config):
        """
        Параметры в config:
         - text_model_name (str)
         - emb_device (str)
         - text_pooling (str | None)
         - emb_normalize (bool)
         - max_tokens (int)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.text_model_name
        self.pooling = config.text_pooling        # может быть None
        self.normalize_output = config.emb_normalize
        self.max_tokens = config.max_tokens
        self.text_classifier_checkpoint = config.text_classifier_checkpoint

        self.model = Mamba(num_layers = 2, d_input = 1024, d_model = 512, num_classes=7, model_name=self.model_name, max_tokens=self.max_tokens, pooling=None,device=self.device).to(self.device)
        checkpoint = torch.load(self.text_classifier_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def extract(self, text_list):
        """
        :param text_list: список строк (или одна строка)
        :return: тензор (B, hidden_dim) или (B, seq_len, hidden_dim), если pooling=None
        """

        if isinstance(text_list, str):
            text_list = [text_list]

        with torch.no_grad():
            logits, hidden = self.model(text_list, with_features=True)

            if hidden.dim() == 3:
                if self.pooling is None:
                    emb = hidden
                else:
                    if self.pooling == "mean":
                        emb = hidden.mean(dim=1)
                    elif self.pooling == "cls":
                        emb = hidden[:, 0, :]
                    elif self.pooling == "max":
                        emb, _ = hidden.max(dim=1)
                    elif self.pooling == "min":
                        emb, _ = hidden.min(dim=1)
                    elif self.pooling == "last":
                        emb = hidden[:, -1, :]
                    elif self.pooling == "sum":
                        emb = hidden.sum(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
            else:
                # На всякий случай, если получилось (B, hidden_dim)
                emb = hidden

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return logits, emb


class PretrainedImageEmbeddingExtractor:
    """
    Извлекает эмбеддинги из изображений
    """

    def __init__(self, config):
        """
        Параметры в config:
         - image_classifier_checkpoint (str)
         - emb_device (str)
         - cut_target_layer (str)
        """
        self.config = config
        self.device = config.emb_device
        self.image_model_type = config.image_model_type

        if self.image_model_type == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.model = resnet18(weights=weights).to(self.device)
            self.model.eval()
            self.features = nn.Sequential(*list(self.model.children())[:-config.cut_target_layer])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.image_model_type == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.model = resnet50(weights=weights).to(self.device)
            self.model.eval()
            self.features = nn.Sequential(*list(self.model.children())[:-config.cut_target_layer])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.image_model_type == "емоresnet50":
            self.model = torch.jit.load(config.image_classifier_checkpoint).to(
                self.device
            )
            self.model.eval()
            self.features = nn.Sequential(
                *list(self.model.children())[: -config.cut_target_layer]
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.image_model_type == "emo":
            self.model = torch.jit.load(config.image_classifier_checkpoint).to(
                self.device
            )
            self.model.eval()
            self.features = nn.Sequential(
                *list(self.model.children())[: -config.cut_target_layer]
            )
            self.feature_final = self.model.fc_feats  # Убираем avgpool и fc

        elif self.image_model_type == 'clip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        elif self.image_model_type == 'body_movement':
            print()

        else:
            raise ValueError(
                f"❌ Неизвестный image_model_type: {self.image_model_type}"
            )

    def extract(self, x):
        """
        :return: тензор (B*seq_len, hidden_dim)
        """

        with torch.no_grad():
            if self.image_model_type == "resnet18" or self.image_model_type == "емоresnet50" or self.image_model_type == "resnet50":
                x = self.features(x)
                x = self.avgpool(x)
                x = x.view(x.shape[0], -1)
            elif self.image_model_type == "emo":
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.feature_final(x)
            elif self.image_model_type == "clip":
                x = self.model.get_image_features(x)
        return x

class PoseFeatureExtractor:
    def __init__(self, list_hand_crafted_features, angle_features, window_size=5):
        # Все исходные признаки
        self.list_hand_crafted_features = list_hand_crafted_features
        
        # Угловые признаки для специальной обработки
        self.angle_features = angle_features
        
        # Параметры для скользящего окна
        self.window_size = window_size
        self.feature_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        
        # Для хранения предыдущих значений скорости
        self.prev_velocity = None
        
        # Генерация имен всех признаков
        self._generate_feature_names()
    
    def _generate_feature_names(self):
        """Генерирует имена всех признаков, включая производные"""
        self.all_feature_names = []
        
        # Оригинальные признаки
        self.all_feature_names.extend(self.list_hand_crafted_features)
        
        # Тригонометрические преобразования углов
        for angle in self.angle_features:
            self.all_feature_names.extend([f"sin_{angle}", f"cos_{angle}"])
        
        # Производные признаки (динамика)
        for feat in self.list_hand_crafted_features:
            self.all_feature_names.extend([f"delta_{feat}", f"velocity_{feat}", f"accel_{feat}"])
        
        # Взаимодействия важных пар признаков
        self.interaction_pairs = [
            ("head_tilt_left", "shoulder_tilt_left"),
            ("head_tilt_right", "shoulder_tilt_right"),
            ("left_hand_above_shoulder", "right_hand_above_shoulder"),
            ("hands_crossed", "shoulder_asymmetry")
        ]
        
        for pair in self.interaction_pairs:
            self.all_feature_names.append(f"interaction_{pair[0]}_{pair[1]}")
    
    def _trigonometric_transform(self, angle, angle_value):
        """Применяет тригонометрические преобразования к углу"""
        rad = np.radians(angle_value)
        return {
            f"sin_{angle}": np.sin(rad),
            f"cos_{angle}": np.cos(rad)
        }
    
    def _calculate_interactions(self, features):
        """Вычисляет взаимодействия между важными парами признаков"""
        interactions = {}
        for feat1, feat2 in self.interaction_pairs:
            interactions[f"interaction_{feat1}_{feat2}"] = features[feat1] * features[feat2]
        return interactions
    
    def _calculate_moving_averages(self):
        """Вычисляет скользящие средние для всех признаков"""
        if len(self.feature_history) < self.window_size:
            return {f"ma_{feat}": 0 for feat in self.list_hand_crafted_features}
        
        ma_features = {}
        for feat in self.list_hand_crafted_features:
            values = [frame[feat] for frame in self.feature_history]
            ma_features[f"ma_{feat}"] = np.mean(values)
        
        return ma_features
    
    def _calculate_dynamics(self, current_features, current_time):
        """Вычисляет динамические признаки (разности, скорости, ускорения)"""
        dynamic_features = {}
        
        if len(self.feature_history) == 0:
            # Нет истории - заполняем нулями
            for feat in self.list_hand_crafted_features:
                dynamic_features.update({
                    f"delta_{feat}": 0,
                    f"velocity_{feat}": 0,
                    f"accel_{feat}": 0
                })
            return dynamic_features
        
        # Берем последний кадр из истории
        prev_features = self.feature_history[-1]
        prev_time = self.time_history[-1]
        time_diff = current_time - prev_time
        
        # Для ускорения нужно проверить, есть ли предыдущая скорость
        has_prev_velocity = self.prev_velocity is not None
        
        for feat in self.list_hand_crafted_features:
            # Разность значений
            delta = current_features[feat] - prev_features[feat]
            dynamic_features[f"delta_{feat}"] = delta
            
            # Скорость изменения
            velocity = delta / time_diff if time_diff > 0 else 0
            dynamic_features[f"velocity_{feat}"] = velocity
            
            # Ускорение
            if has_prev_velocity:
                accel = (velocity - self.prev_velocity[feat]) / time_diff if time_diff > 0 else 0
                dynamic_features[f"accel_{feat}"] = accel
            else:
                dynamic_features[f"accel_{feat}"] = 0
        
        return dynamic_features
    
    def process_frame(self, current_features, current_time):
        """Обрабатывает кадр с признаками и возвращает расширенный набор признаков"""
        # 1. Копируем оригинальные признаки
        features = current_features.copy()
        
        # 2. Добавляем тригонометрические преобразования углов
        trig_features = {}
        for angle in self.angle_features:
            if angle in current_features:
                trig_features.update(self._trigonometric_transform(angle, current_features[angle]))
        features.update(trig_features)
        
        # 3. Добавляем взаимодействия признаков
        interaction_features = self._calculate_interactions(features)
        features.update(interaction_features)
        
        # 4. Добавляем скользящие средние
        ma_features = self._calculate_moving_averages()
        features.update(ma_features)
        
        # 5. Добавляем динамические признаки
        dynamic_features = self._calculate_dynamics(features, current_time)
        features.update(dynamic_features)
        
        # 6. Обновляем историю и предыдущие значения
        self.feature_history.append(features.copy())
        self.time_history.append(current_time)
        
        # Сохраняем текущие скорости для следующего кадра
        self.prev_velocity = {feat: dynamic_features[f"velocity_{feat}"] 
                            for feat in self.list_hand_crafted_features}
        
        # 7. Создаем выходной вектор в правильном порядке
        output_vector = [features.get(name, 0) for name in self.all_feature_names]
        
        return np.array(output_vector, dtype=np.float32)
    
    def get_feature_names(self):
        """Возвращает имена всех признаков в правильном порядке"""
        return self.all_feature_names
    
import numpy as np
from collections import deque

class EfficientPoseFeatureExtractor:
    def __init__(self, list_hand_crafted_features, angle_features, window_size=5):
        # Исходные признаки (34 шт.)
        self.base_features = list_hand_crafted_features
        
        # Только ключевые углы для тригонометрии
        self.key_angles = ["head_pitch_angle", "head_roll_angle", "head_yaw_angle"]
        
        # Только осмысленные динамические признаки
        self.dynamic_features = [
            "head_pitch_angle", "head_roll_angle", "head_yaw_angle",
            "shoulder_asymmetry", "hands_crossed"
        ]
        
        # Практически полезные взаимодействия
        self.interaction_pairs = [
            ("head_tilt_left", "shoulder_tilt_left"),
            ("head_tilt_right", "shoulder_tilt_right"),
            ("hands_crossed", "shoulder_asymmetry")
        ]
        
        # История для скользящих средних
        self.window_size = window_size
        self.feature_history = deque(maxlen=window_size)
        
        # Генерация имен фичей
        self._generate_feature_names()
    
    def _generate_feature_names(self):
        """Генерация только полезных признаков"""
        self.feature_names = []
        
        # 1. Базовые признаки (34)
        self.feature_names.extend(self.base_features)
        
        # 2. Тригонометрия только для ключевых углов (3 угла * 2 = 6)
        for angle in self.key_angles:
            self.feature_names.extend([f"sin_{angle}", f"cos_{angle}"])
        
        # 3. Динамика только для ключевых признаков (5 признаков * 3 = 15)
        for feat in self.dynamic_features:
            self.feature_names.extend([f"delta_{feat}", f"velocity_{feat}", f"accel_{feat}"])
        
        # 4. Взаимодействия (3 пары)
        for a, b in self.interaction_pairs:
            self.feature_names.append(f"{a}_x_{b}")
        
        # 5. Скользящие средние только для углов (7 признаков)
        for angle in self.key_angles + ["shoulder_asymmetry"]:
            self.feature_names.append(f"ma_{angle}")
    
    def _add_trigonometrics(self, features):
        """Добавляет sin/cos для ключевых углов"""
        result = {}
        for angle in self.key_angles:
            rad = np.radians(features[angle])
            result[f"sin_{angle}"] = np.sin(rad)
            result[f"cos_{angle}"] = np.cos(rad)
        return result
    
    def _add_dynamics(self, current_features, current_time):
        """Вычисляет динамику только для ключевых признаков"""
        if not self.feature_history:
            return {f"{prefix}_{feat}": 0 
                for feat in self.dynamic_features 
                for prefix in ["delta", "velocity", "accel"]}
        
        prev_features = self.feature_history[-1]
        time_diff = current_time - prev_features["timestamp"]
        
        dynamics = {}
        for feat in self.dynamic_features:
            delta = current_features[feat] - prev_features[feat]
            velocity = delta / time_diff if time_diff > 0 else 0
            
            dynamics[f"delta_{feat}"] = delta
            dynamics[f"velocity_{feat}"] = velocity
            
            # Ускорение (если есть предыдущая скорость)
            if "prev_velocity" in prev_features:
                prev_vel = prev_features["prev_velocity"].get(feat, 0)
                accel = (velocity - prev_vel) / time_diff if time_diff > 0 else 0
                dynamics[f"accel_{feat}"] = accel
            else:
                dynamics[f"accel_{feat}"] = 0
        
        return dynamics
    
    def _add_interactions(self, features):
        """Вычисляет полезные взаимодействия"""
        return {
            f"{a}_x_{b}": features[a] * features[b]
            for a, b in self.interaction_pairs
        }
    
    def _add_moving_averages(self):
        """Скользящие средние только для углов"""
        if len(self.feature_history) < self.window_size:
            return {f"ma_{angle}": 0 for angle in self.key_angles + ["shoulder_asymmetry"]}
        
        ma = {}
        for angle in self.key_angles + ["shoulder_asymmetry"]:
            values = [f[angle] for f in self.feature_history]
            ma[f"ma_{angle}"] = np.mean(values)
        return ma
    
    def process_frame(self, frame_features, timestamp):
        """Основной метод обработки кадра"""
        # 1. Базовые признаки
        features = frame_features.copy()
        
        # 2. Тригонометрия
        features.update(self._add_trigonometrics(frame_features))
        
        # 3. Динамика
        dynamics = self._add_dynamics(frame_features, timestamp)
        features.update(dynamics)
        
        # 4. Взаимодействия
        features.update(self._add_interactions(frame_features))
        
        # 5. Скользящие средние
        features.update(self._add_moving_averages())
        
        # Сохраняем историю
        history_record = frame_features.copy()
        history_record["timestamp"] = timestamp
        history_record["prev_velocity"] = {k: v for k, v in dynamics.items() 
                                        if k.startswith("velocity_")}
        self.feature_history.append(history_record)
        
        # Возвращаем вектор в правильном порядке
        return np.array([features.get(name, 0) for name in self.feature_names], dtype=np.float32)
    
    def get_feature_names(self):
        return self.feature_names