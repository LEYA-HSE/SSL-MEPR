# coding: utf-8

import os
import logging
import torch
# import whisper
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor
from data_loading.feature_extractor import PoseFeatureExtractor

class DatasetVideo(Dataset):
    """
    Датасет для детектирования формирования обучающих данных по видео.
    """

    def __init__(
        self,
        csv_path,
        video_dir,
        config,
        split,
        image_feature_extractor,
        dataset_name,
        task
    ):
        """
        :param csv_path: Путь к CSV-файлу.
        :param video_dir: Путь к видео
        :param label_columns: "diagnosis".
        :param split: "train", "dev" или "test".
        :param image_feature_extractor: Экстрактор видео признаков
        :param audio_feature_extractor: Экстрактор аудио признаков
        :param subset_size: Если > 0, используется только первые N дивео из CSV (для отладки).
        :param dataset_name: Название корпуса
        """
        super().__init__()
        self.split = split
        self.video_dir = video_dir
        self.image_feature_extractor = image_feature_extractor
        self.subset_size    = config.subset_size
        self.seed = config.random_seed
        self.dataset_name = dataset_name
        # self.emotion_columns = config.emotion_columns
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path = config.save_feature_path
        self.roi_video = config.roi_video # body or body_movement, face or scene
        self.counter_need_frames = config.counter_need_frames
        self.image_size = config.image_size
        self.image_model_type = config.image_model_type
        self.window_size = config.window_size
        self.list_bbox = ["startX_face","startY_face","endX_face","endY_face"]
        self.list_hand_crafted_features = [
                "head_turn_left", # values 0, 1
                "head_turn_right", # values 0, 1
                "head_tilt_left", # values 0, 1
                "head_tilt_right", # values 0, 1
                "head_down", # values 0, 1
                "head_up", # values 0, 1
                "head_lean_forward", # values 0, 1
                "shoulder_tilt_left", # values 0, 1
                "shoulder_tilt_right", # values 0, 1
                "shoulder_forward", # values 0, 1
                "shoulder_backward", # values 0, 1
                "head_pitch_angle", # углы
                "head_roll_angle", # углы
                "head_yaw_angle", # углы
                "shoulder_roll_angle", # углы
                "shoulder_asymmetry", # углы
                "shoulder_curve_left", # углы
                "shoulder_curve_right", # углы
                "right_elbow_above_shoulder", # values 0, 1
                "left_elbow_above_shoulder", # values 0, 1
                "left_hand_on_face", # values 0, 1
                "left_hand_above_shoulder", # values 0, 1
                "left_hand_below_shoulder", # values 0, 1
                "left_hand_near_left_ear", # values 0, 1
                "left_hand_in_frame", # values 0, 1
                "right_hand_on_face", # values 0, 1
                "right_hand_above_shoulder", # values 0, 1
                "right_hand_below_shoulder", # values 0, 1
                "right_hand_near_right_ear", # values 0, 1
                "right_hand_in_frame", # values 0, 1
                "hands_crossed", # values 0, 1
                "hands_crossed_above_shoulders", # values 0, 1
                "hands_crossed_below_shoulders", # values 0, 1
                "hands_above_head" # values 0, 1
                ]
        
        self.binary_features = [
                "head_turn_left", # values 0, 1
                "head_turn_right", # values 0, 1
                "head_tilt_left", # values 0, 1
                "head_tilt_right", # values 0, 1
                "head_down", # values 0, 1
                "head_up", # values 0, 1
                "head_lean_forward", # values 0, 1
                "shoulder_tilt_left", # values 0, 1
                "shoulder_tilt_right", # values 0, 1
                "shoulder_forward", # values 0, 1
                "shoulder_backward", # values 0, 1
                "right_elbow_above_shoulder", # values 0, 1
                "left_elbow_above_shoulder", # values 0, 1
                "left_hand_on_face", # values 0, 1
                "left_hand_above_shoulder", # values 0, 1
                "left_hand_below_shoulder", # values 0, 1
                "left_hand_near_left_ear", # values 0, 1
                "left_hand_in_frame", # values 0, 1
                "right_hand_on_face", # values 0, 1
                "right_hand_above_shoulder", # values 0, 1
                "right_hand_below_shoulder", # values 0, 1
                "right_hand_near_right_ear", # values 0, 1
                "right_hand_in_frame", # values 0, 1
                "hands_crossed", # values 0, 1
                "hands_crossed_above_shoulders", # values 0, 1
                "hands_crossed_below_shoulders", # values 0, 1
                "hands_above_head" # values 0, 1
        ]

        if  self.image_model_type == 'clip':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # elif self.image_model_type == 'body_movement':
        #     self.processor = PoseFeatureExtractor(list_hand_crafted_features=self.list_hand_crafted_features, angle_features=self.angle_features,window_size=self.counter_need_frames)

        if self.dataset_name == 'cmu_mosei':
            self.label_columns = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"]
        elif self.dataset_name == 'fiv2':
            self.label_columns = ["openness","conscientiousness","extraversion","agreeableness","non-neuroticism"]
        else:
            raise ValueError(f"Название датафрейма {self.dataset_name} не соотвествует целевому!")
    

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna() # убераем кадры где не найдено лица
        self.df = self.df.rename(columns={
            "video_name": "filename",
        })
        # self.len_labels = len(self.df.diagnosis.tolist())
        if self.subset_size > 0:
            self.need_segment_name = list(set(self.df.filename.tolist()))[:self.subset_size]
            self.df = self.df[self.df.filename.isin(self.need_segment_name)]
            logging.info(f"[DatasetVideo] Используем только {len(self.df)} записей (subset_size={self.subset_size}).")
        else:
            self.need_segment_name = list(set(self.df.filename.tolist()))

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Ошибка: директория с аудио {self.video_dir} не существует!")

        if self.save_prepared_data:
            self.meta = []
            meta_filename = 'task_{}_{}_{}_seed_{}_subset_size_{}_seg_{}_roi_{}_video_model_{}_feature_norm_{}.pickle'.format(
                task,
                self.dataset_name,
                self.split,
                self.seed,
                self.subset_size,
                self.counter_need_frames,
                self.roi_video,
                self.image_model_type,
                config.emb_normalize,
            )

            pickle_path = os.path.join(self.save_feature_path, meta_filename)
            self.load_data(pickle_path)

            if not self.meta:
                self.prepare_data()
                os.makedirs(self.save_feature_path, exist_ok=True)
                self.save_data(pickle_path)

    def save_data(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as handle:
                self.meta = pickle.load(handle)
        else:
            self.meta = []

    def __len__(self):
        if self.save_prepared_data:
            return len(self.meta)
        else:
            return len(self.need_segment_name)

    def select_uniform_frames(self, frames, N):
        if len(frames) <= N:
            return frames  # Если кадров меньше N, вернуть все
        else:
            np.random.seed(self.seed)
            indices = np.linspace(0, len(frames) - 1, num=N, dtype=int)
            return [frames[i] for i in indices]

    def pth_processing(self, fp):
        class PreprocessInput(torch.nn.Module):
            def init(self):
                super(PreprocessInput, self).init()

            def forward(self, x):
                x = x.to(torch.float32)
                x = torch.flip(x, dims=(0,))
                x[0, :, :] -= 91.4953
                x[1, :, :] -= 103.8827
                x[2, :, :] -= 131.0912
                return x

        def get_img_torch(img):

            if self.image_model_type == "emoresnet50" or self.image_model_type == "emo":
                ttransform = transforms.Compose(
                    [transforms.PILToTensor(), PreprocessInput()]
                )
            elif self.image_model_type == 'resnet18' or self.image_model_type == 'resnet50':
                ttransform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

            if self.image_model_type != 'clip':
                img = img.resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
                img = ttransform(img)
                img = torch.unsqueeze(img, 0).to("cuda")
            elif self.image_model_type == 'clip':
                img = self.processor(images=img, return_tensors="pt").to("cuda")
                img = img['pixel_values']
            return img

        return get_img_torch(fp)
    
    def draw_box(self, image, box, color=(255, 0, 255)):
        """Draw a rectangle on the image."""
        line_width = 2
        lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    def find_file_recursive(self, base_dir, filename):
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def get_data(self, segment_name):

        curr_data = self.df[self.df.filename==segment_name] # отбираем все строки нужного сегмента видео
        curr_data = curr_data.dropna() # убераем кадры где не найдено лица

        label_vec = curr_data[self.label_columns].values[0]
        curr_frames = list(set(curr_data.frame.tolist())) # считываем фреймы
        need_curr_frames = self.select_uniform_frames(curr_frames, self.counter_need_frames)
        # print(curr_frames, len(need_curr_frames))
        pass

        if self.dataset_name == 'fiv2':
            full_path_video = self.find_file_recursive(self.video_dir, curr_data.filename.unique()[0])
        else:
            full_path_video = os.path.join(self.video_dir, curr_data.filename.unique()[0])

        cap = cv2.VideoCapture(full_path_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        counter = 1

        feature_names = None

        all_frames = []
        frame_to_faces_indices = {}

        if self.roi_video == 'body_movement':
            all_frames = curr_data[curr_data.frame.isin(need_curr_frames)][self.list_hand_crafted_features].values
            # all_frames, feature_names = self.add_temporal_features(current_features, self.list_hand_crafted_features, self.binary_features)
        else:
            while True:
                ret, im0 = cap.read()
                if not ret:
                    break

                if counter in need_curr_frames:
                    idx = need_curr_frames.index(counter)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    if self.roi_video == 'body' or self.roi_video == 'face':
                        bboxs = curr_data[curr_data.frame==counter][self.list_bbox].values.astype('int')
                        frame_to_faces_indices[idx] = range(idx, idx + len(bboxs))
                        for bbox in bboxs:
                            curr_fr = im0[bbox[1]: bbox[3], bbox[0]: bbox[2]]
                            curr_fr = self.pth_processing(Image.fromarray(curr_fr))
                            all_frames.append(curr_fr)
                    elif self.roi_video == 'scene':
                        curr_fr = im0
                        curr_fr = self.pth_processing(Image.fromarray(curr_fr))
                        all_frames.append(curr_fr)

                counter += 1

            # plt.imshow(im0)
            # plt.show()
        cap.release()
        
        if self.roi_video == 'body' or self.roi_video == 'face':
            all_frames = torch.cat(all_frames, dim=0)
            video_features = self.image_feature_extractor.extract(all_frames).to('cpu')

            frame_features = []
            for idx in range(len(need_curr_frames)):
                if idx in frame_to_faces_indices:
                    face_indices = frame_to_faces_indices[idx]
                    if not face_indices:
                        # print(0) 
                        frame_features.append(torch.zeros(video_features.shape[1]))
                    else:
                        # print(1, face_indices)
                        frame_features.append(video_features[list(face_indices)].mean(dim=0))
            video_features = torch.stack(frame_features)
        elif self.roi_video == 'scene':
            all_frames = torch.cat(all_frames, dim=0)
            video_features = self.image_feature_extractor.extract(all_frames).to('cpu')
        elif self.roi_video == 'body_movement':
            # print(all_frames)
            video_features = torch.tensor(all_frames, dtype=torch.float32)
            # print(video_features.shape)


        torch.cuda.empty_cache()
        
        if self.roi_video == 'body_movement':
            return {
                "video_path": full_path_video,
                "feature_names": self.list_hand_crafted_features,
                "video": video_features,
                "label": torch.tensor(label_vec, dtype=torch.float32),
            }
        else:
            return {
                "video_path": full_path_video,
                "video": video_features,
                "label": torch.tensor(label_vec, dtype=torch.float32),
            }

    # def add_temporal_features(self, X, feature_names, binary_feature_names):
    #     """
    #     Добавляет временные признаки для каждого кадра.
        
    #     Args:
    #         X: np.ndarray, shape = (N, M)
    #         feature_names: list of str, длина M
    #         binary_feature_names: list of str — список бинарных признаков
            
    #     Returns:
    #         X_enhanced: np.ndarray, shape = (N, M_new)
    #         new_feature_names: list of str
    #     """
    #     N, M = X.shape

    #     new_columns = []
    #     new_feature_names = []

    #     for col_idx in range(M):
    #         feat_name = feature_names[col_idx]
    #         col_data = X[:, col_idx]

    #         # Сохраняем оригинальный признак
    #         new_columns.append(col_data.copy())
    #         new_feature_names.append(feat_name)

    #         # feat_prev
    #         col_prev = np.roll(col_data, shift=1)
    #         col_prev[0] = np.nan
    #         new_columns.append(col_prev)
    #         new_feature_names.append(f"{feat_name}_prev")

    #         # feat_next
    #         col_next = np.roll(col_data, shift=-1)
    #         col_next[-1] = np.nan
    #         new_columns.append(col_next)
    #         new_feature_names.append(f"{feat_name}_next")

    #         # feat_delta
    #         delta = col_data - col_prev
    #         delta[0] = np.nan
    #         new_columns.append(delta)
    #         new_feature_names.append(f"{feat_name}_delta")

    #         # feat_absdelta
    #         new_columns.append(np.abs(delta))
    #         new_feature_names.append(f"{feat_name}_absdelta")

    #         # feat_change — только для бинарных
    #         if feat_name in binary_feature_names:
    #             change_flag = (delta != 0).astype(float)
    #             change_flag[0] = np.nan
    #             new_columns.append(change_flag)
    #             new_feature_names.append(f"{feat_name}_change")

    #     # Объединяем все признаки
    #     X_enhanced = np.column_stack(new_columns)

    #     return X_enhanced, new_feature_names

    def prepare_data(self):
        """
        Загружает и обрабатывает один элемент датасета (он‑the‑fly).
        """

        for idx, segment_name in enumerate(tqdm(self.need_segment_name)):
            curr_dict = self.get_data(segment_name)

            self.meta.append(
            curr_dict
            )

    def __getitem__(self, index):
        if self.save_prepared_data:
            return self.meta[index]
        else:
            return self.get_data(self.need_segment_name[index])