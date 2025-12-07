# core/media_utils.py
from __future__ import annotations
import os
import subprocess
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def get_oscilloscope(waveform, sr, path_save):
    fig = plt.figure(figsize=(10, 2), dpi=100, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.plot(np.linspace(0, len(waveform)/sr, len(waveform)), waveform, linewidth=0.5)
    plt.savefig(f'{path_save}/oscilloscope.jpg', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def convert_video_to_audio(file_path: str, file_save: str, sr: int = 16000) -> str:
    path_save = file_save + ".wav"
    if not os.path.exists(path_save):
        ffmpeg_command = f"ffmpeg -y -i {file_path} -async 1 -vn -acodec pcm_s16le -ar {sr} {path_save}"
        subprocess.call(ffmpeg_command, shell=True)
    return path_save

def transform_matrix(matrix):
    threshold1 = 1 - 1/7
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo):
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).numpy()
    pred_emo = transform_matrix(pred_emo)
    return pred_emo

_FACE_DET = None
_BODY_DET = None
try:
    from core.modalities.video.video_preprocessor import face_detector as _FACE_DET, body_detector as _BODY_DET
except Exception:
    _FACE_DET, _BODY_DET = None, None

BBox = Tuple[int, int, int, int]

def infer_keyframe_indices(
    result: dict,
    total_frames: int,
    n_default: int = 6,
    fps: Optional[float] = None,
) -> List[int]:

    idx_keys = ["key_frame_indices", "frame_indices", "used_frames", "steps_idx"]
    for k in idx_keys:
        if k in result:
            arr = np.asarray(result[k]).astype(int)
            arr = np.clip(arr, 0, max(0, total_frames - 1))
            return sorted(np.unique(arr).tolist())

    ts_keys = ["timestamps", "key_timestamps", "steps_ts"]
    for k in ts_keys:
        if k in result and fps is not None and fps > 0:
            ts = np.asarray(result[k], dtype=float)
            arr = np.clip((ts * fps).round().astype(int), 0, max(0, total_frames - 1))
            return sorted(np.unique(arr).tolist())

    n = min(n_default, max(1, total_frames))
    return np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()

def _get_bbox_from_result(result: dict, idx: int, kind: str) -> Optional[BBox]:
    b1 = result.get("bboxes", {})
    if isinstance(b1, dict):
        arr = b1.get(kind)
        if isinstance(arr, (list, tuple)) and idx < len(arr) and arr[idx] is not None:
            x1, y1, x2, y2 = map(int, arr[idx])
            return x1, y1, x2, y2

    b2 = result.get(f"{kind}_bboxes")
    if isinstance(b2, (list, tuple)) and idx < len(b2) and b2[idx] is not None:
        x1, y1, x2, y2 = map(int, b2[idx])
        return x1, y1, x2, y2

    det = result.get("detections", {})
    if isinstance(det, dict):
        arr = det.get(kind)
        if isinstance(arr, (list, tuple)) and idx < len(arr):
            item = arr[idx]
            if isinstance(item, dict) and item.get("bbox") is not None:
                x1, y1, x2, y2 = map(int, item["bbox"])
                return x1, y1, x2, y2
    return None

def _safe_crop(frame: np.ndarray, bbox: Optional[BBox], fallback: str) -> np.ndarray:

    h, w = frame.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]

    if fallback == "face":
        cw = int(w * 0.38)
        ch = int(h * 0.42)
        cx = w // 2
        x1 = max(0, cx - cw // 2)
        x2 = min(w, cx + cw // 2)
        y1 = 0
        y2 = min(h, ch)
        return frame[y1:y2, x1:x2]
    else:  # 'body'
        cw = int(w * 0.55)
        x1 = max(0, (w - cw) // 2)
        x2 = min(w, x1 + cw)
        y1 = int(h * 0.15)
        y2 = int(h * 0.90)
        return frame[y1:y2, x1:x2]

def _detect_face_body_on_frame(frame_bgr: np.ndarray) -> Tuple[Optional[BBox], Optional[BBox]]:

    if _FACE_DET is None and _BODY_DET is None:
        return None, None

    h, w = frame_bgr.shape[:2]
    im_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---- FACE (MediaPipe) ----
    best_face = None
    if _FACE_DET is not None:
        try:
            res = _FACE_DET.process(im_rgb)
            if res and res.detections:
                cands = []
                for det in res.detections:
                    rb = det.location_data.relative_bounding_box
                    x1 = max(int(rb.xmin * w), 0)
                    y1 = max(int(rb.ymin * h), 0)
                    x2 = min(int((rb.xmin + rb.width) * w),  w)
                    y2 = min(int((rb.ymin + rb.height) * h), h)
                    if x2 > x1 and y2 > y1:
                        area = (x2 - x1) * (y2 - y1)
                        score = float(det.score[0]) if getattr(det, "score", None) else 0.0
                        cands.append(((x1, y1, x2, y2), area * (0.5 + score)))
                if cands:
                    cands.sort(key=lambda t: t[1], reverse=True)
                    best_face = cands[0][0]
        except Exception:
            best_face = None

    # ---- BODY (YOLO) ----
    best_body = None
    if _BODY_DET is not None:
        try:
            yres = _BODY_DET.predict(im_rgb, imgsz=640, conf=0.05, iou=0.5, verbose=False)
            boxes = []
            if yres and len(yres[0].boxes):
                for b in yres[0].boxes:
                    bx = b.xyxy.int().cpu().numpy()[0]
                    x1, y1, x2, y2 = (
                        int(max(0, bx[0])),
                        int(max(0, bx[1])),
                        int(min(w - 1, bx[2])),
                        int(min(h - 1, bx[3])),
                    )
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
            if boxes:
                if best_face is not None:
                    cx = (best_face[0] + best_face[2]) // 2
                    cy = (best_face[1] + best_face[3]) // 2
                    for b in boxes:
                        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                            best_body = b
                            break
                if best_body is None:
                    best_body = max(boxes, key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]))
        except Exception:
            best_body = None

    return best_face, best_body

def extract_keyframes_from_result(
    video_path: str,
    result: dict,
    out_dir: str = "outputs/samples",
    n_default: int = 6,
) -> Dict[str, List[str]]:

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"indices": [], "scene": [], "face": [], "body": []}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = infer_keyframe_indices(result, total_frames, n_default=n_default, fps=fps)

    scene_paths, face_paths, body_paths = [], [], []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        p_scene = os.path.join(out_dir, f"scene_{i:02d}_frame{idx}.jpg")
        cv2.imwrite(p_scene, frame)
        scene_paths.append(p_scene)

        face_bbox = _get_bbox_from_result(result, idx, "face")
        body_bbox = _get_bbox_from_result(result, idx, "body")

        try:
            if (face_bbox is None or body_bbox is None) and ('_FACE_DET' in globals() or '_BODY_DET' in globals()):
                df, db = _detect_face_body_on_frame(frame)
                if face_bbox is None:
                    face_bbox = df
                if body_bbox is None:
                    body_bbox = db
        except NameError:
            pass

        face_crop = _safe_crop(frame, face_bbox, "face")
        p_face = os.path.join(out_dir, f"face_{i:02d}_frame{idx}.jpg")
        cv2.imwrite(p_face, face_crop)
        face_paths.append(p_face)

        body_crop = _safe_crop(frame, body_bbox, "body")
        p_body = os.path.join(out_dir, f"body_{i:02d}_frame{idx}.jpg")
        cv2.imwrite(p_body, body_crop)
        body_paths.append(p_body)

    cap.release()
    return {
        "indices": [int(v) for v in indices],
        "scene": scene_paths,
        "face": face_paths,
        "body": body_paths,
    }
