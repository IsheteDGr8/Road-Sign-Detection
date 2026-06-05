"""Phase 1b: green mask overlay on data/guide signs/*.jpg."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from test_utils import data_subfolder_path, load_resized, save_detection_result

DATA_FOLDER = "guide signs"
GREEN_OVERLAY = (0, 200, 0)  # BGR
MIN_GREEN_PIXELS = 1000


def get_static_green_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 100, 50), (85, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def image_paths():
    folder = data_subfolder_path(DATA_FOLDER)
    patterns = [
        os.path.join(folder, "*.jpg"),
        os.path.join(folder, "*.jpeg"),
        os.path.join(folder, "*.png"),
    ]
    seen = set()
    paths = []
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def main():
    paths = image_paths()
    if not paths:
        print(f"No images found in data/{DATA_FOLDER}/")
        return 1

    passed = failed = 0
    saved = []
    for path in paths:
        name = os.path.basename(path)
        frame = load_resized(path)
        if frame is None:
            print(f"SKIP {name}: could not load")
            continue

        mask = get_static_green_mask(frame)
        green_px = cv2.countNonZero(mask)
        overlay = frame.copy()
        overlay[mask > 0] = GREEN_OVERLAY

        out_path = save_detection_result(overlay, DATA_FOLDER, name)
        saved.append(out_path)

        if green_px >= MIN_GREEN_PIXELS:
            passed += 1
            print(f"PASS {name}: {green_px} green px -> {out_path}")
        else:
            failed += 1
            print(f"FAIL {name}: only {green_px} green px -> {out_path}")

    print(f"\n{passed} passed, {failed} failed, {len(saved)} saved")
    if saved:
        try:
            os.startfile(os.path.dirname(saved[0]))
        except OSError:
            pass
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
