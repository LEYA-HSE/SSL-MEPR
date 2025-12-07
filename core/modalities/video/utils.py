import cv2
import numpy as np

def draw_box(image, box, color=(255, 0, 255)):
    line_width = 2
    lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    face_roi = cv2.resize(face_roi, (112, 112))
    face_roi = face_roi.astype('float32') / 255.0
    return face_roi

def preprocess_body(body_roi: np.ndarray) -> np.ndarray:
    body_roi = cv2.resize(body_roi, (224, 224))
    body_roi = body_roi.astype('float32') / 255.0
    return body_roi
