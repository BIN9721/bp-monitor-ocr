import cv2
import numpy as np

from bp_ocr.rule_based import DIGIT_TEMPLATES, SEGMENT_REGIONS

CELL_WIDTH = 96
CELL_HEIGHT = 64


def render_clean_digit(digit: int | None, width: int = CELL_WIDTH, height: int = CELL_HEIGHT) -> np.ndarray:
    """Ve mot chu so 'sach' (khong nhieu, khong bong mo) dung de unit test rule-based.

    Day KHONG phai bo sinh du lieu tong hop day du (xem scripts/generate_synthetic_dataset.py
    cho ban co augmentation), chi la fixture toi thieu de kiem tra logic so khop 7 doan.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    template = DIGIT_TEMPLATES.get(digit, {}) if digit is not None else {}
    for name, (fx, fy, fw, fh) in SEGMENT_REGIONS.items():
        if not template.get(name, False):
            continue
        x, y = int(fx * width), int(fy * height)
        w, h = max(1, int(fw * width)), max(1, int(fh * height))
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, thickness=-1)
    return image
