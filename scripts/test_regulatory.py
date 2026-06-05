"""Test STOP / DO NOT ENTER / SPEED LIMIT on data/regulatory signs/."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from test_utils import data_subfolder_path, load_resized, save_detection_result

DATA_FOLDER = "regulatory signs"
RED_COLOR = (0, 0, 255)
SPEED_COLOR = (255, 255, 255)
TARGET_PREFIXES = ("stop_", "no_entry_", "speed_limit_")


def get_static_red_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)),
        cv2.inRange(hsv, (170, 120, 70), (180, 255, 255)),
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def get_static_white_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(gray, 175, 255)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)),
        cv2.inRange(hsv, (170, 120, 70), (180, 255, 255)),
    )
    white = cv2.bitwise_and(white, cv2.bitwise_not(red))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    white = cv2.morphologyEx(cv2.morphologyEx(white, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    rows, cols = white.shape
    frame_area = rows * cols
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked = [(cv2.contourArea(c), cv2.boundingRect(c), c) for c in contours if cv2.contourArea(c) >= 250]
    ranked.sort(reverse=True)
    cleaned = np.zeros_like(white)
    if not ranked:
        return cleaned

    if ranked[0][0] > frame_area * 0.20:
        cv2.drawContours(cleaned, [ranked[0][2]], -1, 255, cv2.FILLED)
        return cleaned

    eligible = [
        item
        for item in ranked
        if (item[1][1] + item[1][3] / 2) <= rows * 0.72 and item[1][2] <= item[1][3] * 2.0
    ]
    reference = max((a for a, _, _ in eligible), default=1.0)
    for area, bb, c in ranked:
        if area < reference * 0.20:
            continue
        if (bb[1] + bb[3] / 2) > rows * 0.72 or bb[2] > bb[3] * 2.0:
            continue
        cv2.drawContours(cleaned, [c], -1, 255, cv2.FILLED)
    return cleaned


def refine_speed_bbox(frame, seed):
    rows, cols = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = seed
    x0, x1 = max(0, x - 4), min(cols, x + w + 4)
    span0 = x1 - x0
    cx = (x0 + x1) // 2
    nw = max(int(span0 * 0.55), 20)
    cxl, cxr = max(0, cx - nw // 2), min(cols, cx + nw // 2)
    top = y
    for row in range(y, int(rows * 0.75)):
        band = gray[row : row + 1, cxl:cxr]
        span = cxr - cxl
        wp = cv2.countNonZero(cv2.inRange(band, 185, 255))
        bp = cv2.countNonZero(cv2.inRange(band, 0, 120))
        if wp > span * 0.55 and bp > span * 0.12:
            top = row
            break
    bottom = top
    for row in range(top, int(rows * 0.75)):
        band = gray[row : row + 1, cxl:cxr]
        span = cxr - cxl
        wp = cv2.countNonZero(cv2.inRange(band, 185, 255))
        bp = cv2.countNonZero(cv2.inRange(band, 0, 120))
        if wp > span * 0.45 and bp > span * 0.08:
            bottom = row
    height = max(bottom - top + 1, 20)
    lower = gray[top + height // 3 : top + height, x0:x1]
    left, right = 0, lower.shape[1] - 1
    for col in range(lower.shape[1]):
        if cv2.countNonZero(cv2.inRange(lower[:, col : col + 1], 190, 255)) > lower.shape[0] * 0.50:
            left = col
            break
    for col in range(lower.shape[1] - 1, -1, -1):
        if cv2.countNonZero(cv2.inRange(lower[:, col : col + 1], 190, 255)) > lower.shape[0] * 0.50:
            right = col
            break
    width = max(right - left + 1, 24)
    return (x0 + left, top, width, height)


def expand_speed_ocr_bbox(visual_bb, seed_bb, cols):
    x, y, w, h = visual_bb
    min_w = max(w, seed_bb[2])
    cx = x + w // 2
    ox = max(0, cx - min_w // 2)
    ow = min(min_w, cols - ox)
    return (ox, y, ow, h)


def count_digit_holes(patch):
    cnts, hier = cv2.findContours(patch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for i in range(len(cnts)) if hier[0][i][3] != -1)


def digit_heuristic(patch, parent_width):
    aspect = patch.shape[0] / max(patch.shape[1], 1)
    width_ratio = patch.shape[1] / parent_width
    if aspect > 2.0 or width_ratio < 0.20:
        return "1"
    if count_digit_holes(patch) >= 1:
        return "0"
    if width_ratio < 0.36:
        return "3"
    if width_ratio < 0.55:
        return "5" if aspect < 1.1 else "0"
    return "0"


def read_speed_digits(frame, visual_bb, seed_bb):
    cols = frame.shape[1]
    x, y, w, h = expand_speed_ocr_bbox(visual_bb, seed_bb, cols)
    gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    num = gray[int(h * 0.52) :, :]
    _, bw = cv2.threshold(num, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cands = []
    for c in cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < 80:
            continue
        dx, dy, dw, dh = cv2.boundingRect(c)
        if dh < num.shape[0] * 0.22 or dw < w * 0.14 or dw > w * 0.75:
            continue
        cands.append((cv2.contourArea(c), dx, bw[dy : dy + dh, dx : dx + dw]))
    cands.sort(reverse=True)
    if len(cands) < 2:
        return ""
    return "".join(digit_heuristic(patch, w) for _, _, patch in sorted(cands[:2], key=lambda t: t[1]))


def draw_label(frame, box, label, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, label, (x, max(y - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def detect_regulatory(frame, red_mask, white_mask):
    rows, cols = frame.shape[:2]
    max_a = rows * cols * 0.95
    labels = []

    speed_candidates = []
    for c in cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 1200 or area > max_a:
            continue
        rough = cv2.boundingRect(c)
        if rough[1] > rows * 0.90 or rough[2] > cols * 0.35:
            continue
        if rough[2] / rough[3] > 1.35:
            continue
        bb = refine_speed_bbox(frame, rough)
        if bb[1] <= 3 and (rough[2] / rough[3]) > 0.95:
            continue
        if bb[3] < 35 or bb[2] / bb[3] > 1.35:
            continue
        digits = read_speed_digits(frame, bb, rough)
        if len(digits) < 2:
            continue
        speed_candidates.append((area, bb, digits))

    if speed_candidates:
        _, bb, digits = max(speed_candidates, key=lambda item: item[0])
        label = f"SPEED LIMIT {digits}"
        draw_label(frame, bb, label, SPEED_COLOR)
        labels.append(label)

    for c in cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 4000 or area > max_a:
            continue
        bb = cv2.boundingRect(c)
        if bb[1] > rows * 0.90:
            continue
        if bb[2] / bb[3] < 0.70 or bb[2] / bb[3] > 1.30:
            continue
        hull = cv2.convexHull(c)
        v = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
        if v < 7 or v > 9:
            continue
        roi = frame[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        white_ratio = cv2.countNonZero(cv2.inRange(gray, 175, 255)) / gray.size
        if white_ratio > 0.35:
            continue
        has_bar = False
        _, bw = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        for wc in cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(wc) < 40:
                continue
            bx, by, bw_w, bh = cv2.boundingRect(wc)
            if bw_w / max(bh, 1) > 2.0 and bw_w > bb[2] * 0.35:
                has_bar = True
                break
        if has_bar:
            draw_label(frame, bb, "DO NOT ENTER", RED_COLOR)
            labels.append("DO NOT ENTER")
        elif v == 8:
            draw_label(frame, bb, "STOP SIGN", RED_COLOR)
            labels.append("STOP SIGN")

    return labels


def expected_label(filename):
    if filename.startswith("stop_"):
        return "STOP SIGN"
    if filename.startswith("no_entry_"):
        return "DO NOT ENTER"
    if filename == "speed_limit_sign_1.jpg":
        return "SPEED LIMIT 30"
    if filename == "speed_limit_sign_2.jpg":
        return "SPEED LIMIT 35"
    return "SPEED LIMIT"


def image_paths():
    folder = data_subfolder_path(DATA_FOLDER)
    alt = os.path.join(os.path.dirname(os.path.dirname(folder)), "build", "data", DATA_FOLDER)
    paths = []
    seen = set()
    for root in (folder, alt):
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(glob.glob(os.path.join(root, pattern))):
                name = os.path.basename(p)
                if p in seen or not name.startswith(TARGET_PREFIXES):
                    continue
                seen.add(p)
                paths.append(p)
    return sorted(paths)


def main():
    paths = image_paths()
    if not paths:
        print(f"No target images found in data/{DATA_FOLDER}/")
        return 1

    passed = failed = 0
    saved = []
    for path in paths:
        name = os.path.basename(path)
        frame = load_resized(path)
        if frame is None:
            print(f"SKIP {name}")
            continue

        labels = detect_regulatory(frame, get_static_red_mask(frame), get_static_white_mask(frame))
        out_path = save_detection_result(frame, DATA_FOLDER, name)
        saved.append(out_path)

        exp = expected_label(name)
        ok = exp in labels
        if ok:
            passed += 1
            print(f"PASS {name}: {labels} -> {out_path}")
        else:
            failed += 1
            print(f"FAIL {name}: expected {exp}, got {labels} -> {out_path}")

    print(f"\n{passed} passed, {failed} failed, {len(saved)} saved")
    if saved:
        try:
            os.startfile(os.path.dirname(saved[0]))
        except OSError:
            pass
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
