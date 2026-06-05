"""Phase 1c: white mask preview on speed limit images (matches ColorSegmenter.cpp)."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from test_utils import data_subfolder_path, load_resized, save_detection_result

DATA_FOLDER = "regulatory signs"
MIN_WHITE_PIXELS = 2000


def get_static_white_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(gray, 165, 255)
    red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    white = cv2.bitwise_and(white, cv2.bitwise_not(cv2.bitwise_or(red1, red2)))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)

    rows, cols = white.shape
    frame_area = rows * cols
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue
        ranked.append((area, cv2.boundingRect(c), c))
    ranked.sort(reverse=True)

    cleaned = np.zeros_like(white)
    if not ranked:
        return cleaned

    largest_area = ranked[0][0]
    if largest_area > frame_area * 0.20:
        cv2.drawContours(cleaned, [ranked[0][2]], -1, 255, cv2.FILLED)
        return cleaned

    for area, bb, c in ranked:
        if area < largest_area * 0.20:
            continue
        cy = bb[1] + bb[3] / 2
        if cy > rows * 0.72 or bb[2] > bb[3] * 2.0:
            continue
        cv2.drawContours(cleaned, [c], -1, 255, cv2.FILLED)
    return cleaned


def build_mask_preview(image, mask):
    tinted = image.copy()
    tinted[mask > 0] = (255, 200, 0)
    return cv2.addWeighted(image, 0.45, tinted, 0.55, 0)


def image_paths():
    folder = data_subfolder_path(DATA_FOLDER)
    patterns = [
        os.path.join(folder, "speed_limit*.jpg"),
        os.path.join(folder, "speed_limit*.jpeg"),
        os.path.join(folder, "speed_limit*.png"),
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
        print(f"No speed limit images found in data/{DATA_FOLDER}/")
        return 1

    passed = failed = 0
    saved = []
    for path in paths:
        name = os.path.basename(path)
        frame = load_resized(path)
        if frame is None:
            print(f"SKIP {name}: could not load")
            continue

        mask = get_static_white_mask(frame)
        white_px = cv2.countNonZero(mask)
        preview = build_mask_preview(frame, mask)

        out_path = save_detection_result(preview, DATA_FOLDER, name)
        saved.append(out_path)

        if white_px >= MIN_WHITE_PIXELS:
            passed += 1
            print(f"PASS {name}: {white_px} white px -> {out_path}")
        else:
            failed += 1
            print(f"FAIL {name}: only {white_px} white px -> {out_path}")

    print(f"\n{passed} passed, {failed} failed, {len(saved)} saved")
    if saved:
        try:
            os.startfile(os.path.dirname(saved[0]))
        except OSError:
            pass
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
