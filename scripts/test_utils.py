"""Shared helpers for saving detection test images."""
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "build", "test_out")
STATIC_WIDTH = 400


def data_subfolder_path(relative_folder):
    """e.g. 'construction signs' -> absolute data path."""
    return os.path.join(ROOT, "data", relative_folder)


def output_subfolder_path(relative_folder):
    """Mirror data/ layout under build/test_out/."""
    path = os.path.join(OUT_ROOT, relative_folder)
    os.makedirs(path, exist_ok=True)
    return path


def load_resized(image_path, width=STATIC_WIDTH):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))


def save_detection_result(frame, data_relative_folder, image_filename):
    """
    Save annotated frame only (no headers, no mask panels).
    Mirrors data/<folder>/ under build/test_out/<folder>/.
    """
    out_dir = output_subfolder_path(data_relative_folder)
    out_path = os.path.join(out_dir, image_filename)
    cv2.imwrite(out_path, frame)
    return out_path


def draw_label(frame, bounding_box, label, color, thickness=3, font_scale=0.8):
    x, y, w, h = bounding_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(
        frame,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
    )
