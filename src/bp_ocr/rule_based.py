"""Nhan dang chu so LCD 7 doan bang rule-based (muc 5 trong thiet ke).

Anh dau vao cho moi ham la mot o chu so da qua preprocess_screen (nhi phan,
255 = pixel sang / net LCD, 0 = nen). Khong tu xu ly threshold trong module nay.
"""

from __future__ import annotations

import numpy as np

# Vi tri tuong doi (x, y, w, h) theo ty le [0, 1] cua tung doan trong o chu so.
# Xem so do 7 doan trong muc 5.1 cua thiet ke.
SEGMENT_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "a": (0.20, 0.00, 0.60, 0.20),  # thanh ngang tren (cao 20% thay vi 15% - o font nho,
                                     # 15% hut mat mot phan muc that, gay nham "7" thanh "1")
    "b": (0.75, 0.10, 0.25, 0.35),  # doc tren-phai
    "c": (0.75, 0.55, 0.25, 0.35),  # doc duoi-phai
    "d": (0.20, 0.85, 0.60, 0.15),  # thanh ngang duoi
    "e": (0.00, 0.55, 0.25, 0.35),  # doc duoi-trai
    "f": (0.00, 0.10, 0.25, 0.35),  # doc tren-trai
    "g": (0.20, 0.425, 0.60, 0.15),  # thanh ngang giua
}

# Mau cau hinh 7 doan cho tung chu so 0-9 (True = doan bat).
DIGIT_TEMPLATES: dict[int, dict[str, bool]] = {
    0: {"a": True, "b": True, "c": True, "d": True, "e": True, "f": True, "g": False},
    1: {"a": False, "b": True, "c": True, "d": False, "e": False, "f": False, "g": False},
    2: {"a": True, "b": True, "c": False, "d": True, "e": True, "f": False, "g": True},
    3: {"a": True, "b": True, "c": True, "d": True, "e": False, "f": False, "g": True},
    4: {"a": False, "b": True, "c": True, "d": False, "e": False, "f": True, "g": True},
    5: {"a": True, "b": False, "c": True, "d": True, "e": False, "f": True, "g": True},
    6: {"a": True, "b": False, "c": True, "d": True, "e": True, "f": True, "g": True},
    7: {"a": True, "b": True, "c": True, "d": False, "e": False, "f": False, "g": False},
    8: {"a": True, "b": True, "c": True, "d": True, "e": True, "f": True, "g": True},
    9: {"a": True, "b": True, "c": True, "d": True, "e": False, "f": True, "g": True},
}

SEGMENT_ON_THRESHOLD = 0.30  # ty le pixel sang toi thieu de coi doan la "bat" (giam tu
                              # 0.35 - da doi chieu voi toan bo du lieu that hien co, khong
                              # co doan nao "phai off" nam trong khoang 0.30-0.35)
BLANK_ON_RATIO_THRESHOLD = 0.05  # neu khong doan nao vuot qua muc nay -> blank


def measure_segment_ratios(digit_cell: np.ndarray) -> dict[str, float]:
    """Do ty le pixel sang (255) trong tung vung doan a-g."""
    height, width = digit_cell.shape[:2]
    ratios = {}
    for name, (fx, fy, fw, fh) in SEGMENT_REGIONS.items():
        x, y = int(fx * width), int(fy * height)
        w, h = max(1, int(fw * width)), max(1, int(fh * height))
        region = digit_cell[y : y + h, x : x + w]
        ratios[name] = float(np.count_nonzero(region)) / region.size if region.size else 0.0
    return ratios


def is_blank(segment_ratios: dict[str, float]) -> bool:
    return max(segment_ratios.values(), default=0.0) < BLANK_ON_RATIO_THRESHOLD


def match_digit(segment_ratios: dict[str, float]) -> tuple[int | None, float]:
    """So khop mem cau hinh doan do duoc voi bang mau 0-9.

    Tra ve (digit, confidence). confidence la ty le doan khop tren tong 7 doan,
    day la mot proxy hinh hoc chu khong phai xac suat da calibrate.
    """
    if is_blank(segment_ratios):
        return None, 1.0 - max(segment_ratios.values(), default=0.0)

    measured_on = {name: ratio >= SEGMENT_ON_THRESHOLD for name, ratio in segment_ratios.items()}

    best_digit = None
    best_score = -1.0
    for digit, template in DIGIT_TEMPLATES.items():
        matches = sum(1 for seg in SEGMENT_REGIONS if measured_on[seg] == template[seg])
        score = matches / len(SEGMENT_REGIONS)
        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit, best_score


def recognize_digit(digit_cell: np.ndarray) -> tuple[int | None, float]:
    """Entry point: anh mot o chu so da nhi phan hoa -> (digit hoac None, confidence)."""
    ratios = measure_segment_ratios(digit_cell)
    return match_digit(ratios)
