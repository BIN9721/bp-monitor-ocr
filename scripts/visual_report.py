#!/usr/bin/env python3
"""Sinh 1 anh bao cao truc quan cho moi lan inference: ANH DAU VAO o tren, KET QUA
NHAN DANG (so da trich xuat tu thuat toan) o duoi.

Muc dich: minh hoa ro rang gioi han cua thuat toan - pipeline chi thanh cong khi buoc
phat hien man hinh (screen_detection.py) khoanh DUNG duoc man hinh LCD trong anh. Neu
buoc nay that bai, khong co chu so nao duoc doc, bat ke chat luong chu so ben trong co
tot hay khong. Bao cao se the hien ro ca hai truong hop: thanh cong (co so + confidence)
va that bai (bao loi ro rang thay vi im lang tra ket qua sai).

Vi du:
    python scripts/visual_report.py --image data/samples/foo.jpg \
        --device-profile configs/device_profiles/omron_hem7121.yaml \
        --output report.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.preprocessing import preprocess_screen
from bp_ocr.roi import DeviceProfile
from bp_ocr.screen_detection import find_screen_corners, warp_to_frontal

PANEL_WIDTH = 640
BG = (24, 24, 24)
FG = (235, 235, 235)
MUTED = (150, 150, 150)
OK = (110, 210, 130)
FAIL = (80, 90, 235)


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, int(h * width / w))))


def bar(height: int, width: int = PANEL_WIDTH) -> np.ndarray:
    return np.full((height, width, 3), BG, dtype=np.uint8)


def label(canvas: np.ndarray, text: str, x: int, y: int, color=FG, scale: float = 0.6, thick: int = 1) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_digit_boxes(screen: np.ndarray, profile: DeviceProfile, fields) -> np.ndarray:
    """Ve digit_rois + chu so da nhan dang truc tiep len anh man hinh, de thay ro
    thuat toan 'trich' so tu dau.

    Luu y: digit=None nghia la 'blank' (o trong, vd khong co chu so hang tram khi gia
    tri < 100) - day la mot ket qua nhan dang HOP LE, khong phai loi. Mau sac duoc quyet
    dinh boi confidence (do tin cay), khong phai boi viec co doc duoc chu so hay khong.
    """
    vis = screen.copy()
    for field_name, field_profile in profile.fields.items():
        field_result = fields.get(field_name)
        for i, (x, y, w, h) in enumerate(field_profile.digit_rois):
            if field_result and i < len(field_result.digits):
                digit = field_result.digits[i].digit
                confidence = field_result.digits[i].confidence
            else:
                digit, confidence = None, 0.0
            color = OK if confidence >= profile.min_confidence else FAIL
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            text = str(digit) if digit is not None else "blank"
            cv2.putText(vis, text, (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def build_report(image_path: Path, profile: DeviceProfile) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Khong doc duoc anh: {image_path}")

    sections: list[np.ndarray] = []

    sections.append(bar(34))
    label(sections[-1], f"ANH DAU VAO: {image_path.name}", 10, 23)
    sections.append(resize_to_width(image, PANEL_WIDTH))

    corners = find_screen_corners(image)

    sections.append(bar(34))
    if corners is None:
        label(sections[-1], "BUOC 1 - PHAT HIEN MAN HINH: THAT BAI", 10, 23, color=FAIL)
        panel = bar(120)
        label(panel, "Khong tim thay tu giac man hinh LCD trong anh nay.", 10, 30, color=FAIL, scale=0.55)
        label(panel, "Thuat toan DUNG LAI o day - khong chu so nao duoc doc,", 10, 55, color=FAIL, scale=0.55)
        label(panel, "bat ke chat luong chu so ben trong co ro net hay khong.", 10, 78, color=FAIL, scale=0.55)
        label(panel, "Nguyen nhan thuong gap: vat the phu trong khung hinh (but", 10, 105, color=MUTED, scale=0.48)
        sections.append(panel)
        panel2 = bar(28)
        label(panel2, "lay mau...), goc chup nghieng, do phan giai anh qua thap.", 10, 20, color=MUTED, scale=0.48)
        sections.append(panel2)
        return np.vstack([resize_to_width(s, PANEL_WIDTH) for s in sections])

    label(sections[-1], "BUOC 1 - PHAT HIEN MAN HINH: THANH CONG", 10, 23, color=OK)

    screen = warp_to_frontal(image, corners, profile.screen_size)
    binary = preprocess_screen(screen, invert=profile.invert)

    pipeline = ReadingPipeline(profile)
    try:
        result = pipeline.run(image)
    except ScreenNotFoundError:
        result = None

    sections.append(bar(30))
    label(sections[-1], "BUOC 2 - MAN HINH DA WARP + O CHU SO DA TRICH XUAT (khung mau = so da doc):", 10, 20, scale=0.5)
    overlay = draw_digit_boxes(screen, profile, result.fields if result else {})
    sections.append(resize_to_width(overlay, PANEL_WIDTH))

    sections.append(bar(34))
    label(sections[-1], "KET QUA NHAN DANG (SO DA TRICH XUAT TU THUAT TOAN):", 10, 23)

    if result is None:
        panel = bar(50)
        label(panel, "Loi khong mong doi trong luc chay pipeline.", 10, 30, color=FAIL)
        sections.append(panel)
    else:
        row_h = 32
        panel = bar(row_h * len(result.fields) + 16)
        y = 24
        for name, field in result.fields.items():
            ok = field.value is not None and field.confidence >= profile.min_confidence
            color = OK if ok else FAIL
            value_str = str(field.value) if field.value is not None else "khong doc duoc"
            label(panel, f"{name.upper():<10s} = {value_str:<10s}  (confidence={field.confidence:.2f})", 24, y, color=color)
            y += row_h
        sections.append(panel)

        sections.append(bar(34))
        status_ok = result.valid
        status_text = "HOP LE" if status_ok else "KHONG HOP LE - can chup lai anh"
        label(sections[-1], f"TRANG THAI TONG THE: {status_text}", 10, 23, color=OK if status_ok else FAIL)

    return np.vstack([resize_to_width(s, PANEL_WIDTH) for s in sections])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Mac dinh: reports/<ten_anh>_report.png")
    args = parser.parse_args()

    profile = DeviceProfile.from_yaml(args.device_profile)
    canvas = build_report(args.image, profile)

    output = args.output or Path("reports") / f"{args.image.stem}_report.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    print(f"Da luu bao cao truc quan tai: {output}")


if __name__ == "__main__":
    main()
