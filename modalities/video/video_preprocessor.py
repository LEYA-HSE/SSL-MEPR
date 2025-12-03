# coding: utf-8
import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.6
)

body_detector = YOLO("modalities/video/checkpoints/body/best_YOLO.pt")


def select_uniform_frames(frames, N):
    """Return N uniformly spaced frame indices (or the original list if shorter)."""
    if len(frames) <= N:
        return frames
    else:
        indices = np.linspace(0, len(frames) - 1, num=N, dtype=int)
        return [frames[i] for i in indices]


def image_processing(image, image_processor):
    """Run the given image through the processor and return tensor pixel values (CUDA)."""
    image = image_processor(images=image, return_tensors="pt").to("cuda")
    return image["pixel_values"]


def get_metadata(
    video_path: str,
    segment_length: int,
    image_processor,              # CLIPProcessor
    device: str = "cuda",         # not used yet; kept for future use
):
    """
    Returns:
        video_name (str),
        body_tensor  [N, 3, H, W] or None,
        face_tensor  [M, 3, H, W] or None,
        scene_tensor [K, 3, H, W] or None
    """
    # Reset YOLO tracker between videos
    if hasattr(body_detector.predictor, "trackers"):
        body_detector.predictor.trackers[0].reset()

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need_frames = select_uniform_frames(list(range(total_frames)), segment_length)

    body_list, face_list, scene_list = [], [], []
    counter = 0

    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        if counter in need_frames:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # SCENE: full frame without cropping
            scene_list.append(image_processing(im_rgb, image_processor))

            # Detections
            face_results = face_detector.process(im_rgb)
            body_results = body_detector.track(
                im_rgb, persist=True, imgsz=640, conf=0.01, iou=0.5,
                augment=False, device=0, verbose=False
            )

            # 1) Faces found
            if face_results.detections:
                w = im_rgb.shape[1]; h = im_rgb.shape[0]
                for det in face_results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1, y1 = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
                    x2 = min(int((bbox.xmin + bbox.width) * w), w)
                    y2 = min(int((bbox.ymin + bbox.height) * h), h)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Find a body box whose area contains the face center
                    body_bbox = None
                    if body_results and len(body_results[0].boxes):
                        for box in body_results[0].boxes:
                            bx = box.xyxy.int().cpu().numpy()[0]
                            if bx[0] <= face_center[0] <= bx[2] and bx[1] <= face_center[1] <= bx[3]:
                                body_bbox = bx
                                break

                    # Face ROI
                    face_roi = im_rgb[y1:y2, x1:x2]
                    if face_roi.size:
                        face_list.append(image_processing(face_roi, image_processor))

                    # Body ROI (if found)
                    if body_bbox is not None:
                        b = body_bbox
                        body_roi = im_rgb[b[1]:b[3], b[0]:b[2]]
                        if body_roi.size:
                            body_list.append(image_processing(body_roi, image_processor))

            # 2) No faces, but YOLO found bodies
            elif body_results and len(body_results[0].boxes):
                # Take the largest body
                largest = max(
                    body_results[0].boxes,
                    key=lambda b: (b.xyxy[0, 2] - b.xyxy[0, 0]) *
                                  (b.xyxy[0, 3] - b.xyxy[0, 1])
                )
                bx = largest.xyxy.int().cpu().numpy()[0]
                body_roi = im_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                if body_roi.size:
                    body_list.append(image_processing(body_roi, image_processor))
                    face_list.append(image_processing(body_roi, image_processor))

            # 3) Neither faces nor bodies — fall back to full frame for both
            else:
                body_list.append(image_processing(im_rgb, image_processor))
                face_list.append(image_processing(im_rgb, image_processor))

        counter += 1

    cap.release()

    body_tensor = torch.cat(body_list, dim=0) if body_list else None
    face_tensor = torch.cat(face_list, dim=0) if face_list else None
    scene_tensor = torch.cat(scene_list, dim=0) if scene_list else None

    return video_name, body_tensor, face_tensor, scene_tensor
