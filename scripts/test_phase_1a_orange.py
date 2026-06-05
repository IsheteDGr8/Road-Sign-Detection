"""Phase 1a: orange mask overlay on one image (saved to mirrored test_out folder)."""
import os
import sys

import cv2

from test_utils import ROOT, data_subfolder_path, load_resized, save_detection_result

DATA_FOLDER = "construction signs"
DEFAULT_IMAGE = "road_work_sign_1.jpg"


def get_static_orange_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (3, 100, 80), (25, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def main():
    image_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    path = os.path.join(data_subfolder_path(DATA_FOLDER), image_name)
    frame = load_resized(path)
    if frame is None:
        print(f"Could not load: {path}")
        return 1

    mask = get_static_orange_mask(frame)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 140, 255)

    out_path = save_detection_result(overlay, DATA_FOLDER, f"mask_{image_name}")
    print(f"Saved: {out_path}")
    print(f"Orange pixels: {cv2.countNonZero(mask)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
