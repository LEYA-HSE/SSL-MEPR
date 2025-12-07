from emonext_model import get_model

import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

import mediapipe as mp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 30
FRAME_SIZE = 224

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

emonext = get_model(num_classes=7, model_size="base", in_22k=False).to(DEVICE)
emonext.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_evenly_spaced_frames(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    while len(frames) < NUM_FRAMES:
        frames.append(frame)

    cap.release()
    return frames


def get_face(frame):
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        x2 = min(int((bbox.xmin + bbox.width) * w), w)
        y2 = min(int((bbox.ymin + bbox.height) * h), h)
        return frame[y1:y2, x1:x2]
    return None


def get_embeddings_from_video(video_path, num_frames):
    frames = get_evenly_spaced_frames(video_path, num_frames)
    face_tensors = []
    last_face = None

    for frame in frames:
        face = get_face(frame)
        if face is not None:
            last_face = face
        if last_face is not None:
            face_tensor = transform(last_face)
            face_tensors.append(face_tensor)

    if not face_tensors:
        return None

    batch = torch.stack(face_tensors).to(DEVICE)
    with torch.no_grad():
        aligned = emonext.stn(batch)
        embeddings = emonext.forward_features(aligned)
    return embeddings.cpu()


def process_video(video_path, output_dir):
    video_id = video_path.stem
    save_path = output_dir / f"{video_id}.pt"
    if save_path.exists():
        return

    emb = get_embeddings_from_video(video_path, NUM_FRAMES)
    if emb is None or torch.isnan(emb).any():
        return

    torch.save({"emb": emb, "length": emb.shape[0]}, save_path)


def process_folder(folder_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = list(folder_path.glob("*.mp4"))
    for video_path in tqdm(videos, desc=f"Processing {folder_path.name}"):
        process_video(video_path, output_dir)


def main(split, folders, base_path, output_root):
    input_root = Path(base_path) / split
    output_root = Path(output_root) / split

    selected_folders = [f for f in input_root.iterdir() if f.name in folders]

    for folder in selected_folders:
        output_dir = output_root / folder.name
        process_folder(folder, output_dir)


#Example usage: main("train", ["training80_01", "training80_02"], "FirstImpressionsV2", "FirstImpressionsV2_embeddings")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--base_path", default="FirstImpressionsV2")
    parser.add_argument("--output_root", default="FirstImpressionsV2_embeddings")
    args = parser.parse_args()

    main(args.split, args.folders, args.base_path, args.output_root)
