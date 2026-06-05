"""Test SERVICE SIGN detection on data/service signs/*.jpg."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from test_utils import data_subfolder_path, draw_label, load_resized, save_detection_result

DATA_FOLDER = "service signs"
SERVICE_COLOR = (255, 100, 0)  # BGR blue


def get_static_blue_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 80, 50), (130, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def is_nested(inner, outer):
    x, y, w, h = inner
    ox, oy, ow, oh = outer
    return x >= ox and y >= oy and x + w <= ox + ow and y + h <= oy + oh


def detect_service(mask, frame):
    rows, cols = frame.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_a = rows * cols * 0.95
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000 or area > max_a:
            continue
        bb = cv2.boundingRect(c)
        if bb[1] > rows * 0.85:
            continue
        ar = bb[2] / bb[3]
        if ar < 0.2 or ar > 5.0:
            continue
        fill = area / (bb[2] * bb[3])
        if fill < 0.55:
            continue
        if bb[1] <= 2 and fill < 0.65:
            continue
        if bb[1] <= 2 and bb[2] >= cols * 0.92:
            continue
        hull = cv2.convexHull(c)
        v = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
        if 4 <= v <= 6:
            candidates.append((area, bb))

    detected = 0
    for i, (area, bb) in enumerate(candidates):
        nested = any(
            j != i and candidates[j][0] > area and is_nested(bb, candidates[j][1])
            for j in range(len(candidates))
        )
        if nested:
            continue
        detected += 1
        draw_label(frame, bb, "SERVICE SIGN", SERVICE_COLOR)
    return detected


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

        det = detect_service(get_static_blue_mask(frame), frame)
        out_path = save_detection_result(frame, DATA_FOLDER, name)
        saved.append(out_path)

        if det:
            passed += 1
            print(f"PASS {name} ({det} detection(s)) -> {out_path}")
        else:
            failed += 1
            print(f"FAIL {name} (no detection) -> {out_path}")

    print(f"\n{passed} passed, {failed} failed, {len(saved)} saved")
    if saved:
        try:
            os.startfile(os.path.dirname(saved[0]))
        except OSError:
            pass
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
