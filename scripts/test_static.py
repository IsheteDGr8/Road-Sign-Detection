"""Headless test for the three static demos in main.cpp (no video)."""
import cv2
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_WIDTH = 400
OUT = os.path.join(ROOT, "build", "test_out")
os.makedirs(OUT, exist_ok=True)


def get_static_red_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def get_static_yellow_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def get_static_blue_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 61, 87), (130, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def detect_red(mask, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_a = frame.shape[0] * frame.shape[1] * 0.95
    labels = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000 or area > max_a:
            continue
        bb = cv2.boundingRect(c)
        if bb[1] > frame.shape[0] * 0.85:
            continue
        ar = bb[2] / bb[3]
        if ar < 0.75 or ar > 1.25:
            continue
        v = len(cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True))
        if v > 10:
            labels.append("DO NOT ENTER")
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 0, 255), 3)
        elif 7 <= v <= 9:
            labels.append("STOP SIGN")
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 3)
    return labels


def detect_warning(mask, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_a = frame.shape[0] * frame.shape[1] * 0.99
    largest = max((cv2.contourArea(c) for c in contours if cv2.contourArea(c) < max_a), default=0)
    labels = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500 or area > max_a or area < largest * 0.30:
            continue
        bb = cv2.boundingRect(c)
        if bb[1] > frame.shape[0] * 0.90:
            continue
        ar = bb[2] / bb[3]
        if ar < 0.2 or ar > 3.0:
            continue
        hull = cv2.convexHull(c)
        v = len(cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True))
        if 3 <= v <= 10:
            labels.append("WARNING SIGN")
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 255), 3)
    return labels


def detect_blue(mask, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_a = frame.shape[0] * frame.shape[1] * 0.95
    labels = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000 or area > max_a:
            continue
        bb = cv2.boundingRect(c)
        if bb[1] > frame.shape[0] * 0.85:
            continue
        ar = bb[2] / bb[3]
        if ar < 0.4 or ar > 4.0:
            continue
        v = len(cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True))
        if 4 <= v <= 5:
            labels.append("INFO SIGN")
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (255, 100, 0), 3)
    return labels


def run_case(name, path, mask_fn, detect_fn):
    frame = cv2.imread(path)
    if frame is None:
        print(f"FAIL {name}: missing {path}")
        return False
    frame = cv2.resize(frame, (STATIC_WIDTH, int(frame.shape[0] * (STATIC_WIDTH / frame.shape[1]))))
    labels = detect_fn(mask_fn(frame), frame)
    out = os.path.join(OUT, f"{name}.jpg")
    cv2.imwrite(out, frame)
    ok = len(labels) > 0
    print(f"{'PASS' if ok else 'FAIL'} {name}: {labels or 'no detection'} -> {out}")
    return ok


cases = [
    ("red_stop", os.path.join(ROOT, "data", "stop_sign_2.jpg"), get_static_red_mask, detect_red),
    ("yellow_warning", os.path.join(ROOT, "data", "pedestrian_crossing_sign_2.jpg"), get_static_yellow_mask, detect_warning),
    ("blue_info", os.path.join(ROOT, "data", "disabled_parking_sign_1.jpg"), get_static_blue_mask, detect_blue),
]
results = [run_case(*c) for c in cases]
sys.exit(0 if all(results) else 1)
