#!/usr/bin/env python3
"""Quet nhanh 1 anh may do va in bang ket qua de kiem tra bang mat, truoc khi dua vao
danh gia hang loat (evaluate.py).

Ngoai ket qua in ra terminal, script luu anh man hinh da warp va tung o chu so da nhi
phan hoa vao thu muc debug de doi chieu bang mat khi gia tri doc sai (ROI lech, sai
nguong doan...) - giong cach kiem tra thu cong da lam khi do device profile.

Vi du:
    python scripts/quick_test.py --image data/samples/foo.jpg \
        --device-profile configs/device_profiles/omron_hem7121.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.preprocessing import preprocess_screen
from bp_ocr.roi import DeviceProfile, extract_all_digit_cells
from bp_ocr.schema import ReadingResult
from bp_ocr.screen_detection import find_screen_corners, warp_to_frontal


def save_debug_images(image, profile: DeviceProfile, debug_dir: Path) -> bool:
    """Luu anh warp + binary + tung o chu so. Tra ve False neu khong tim thay man hinh."""
    corners = find_screen_corners(image)
    if corners is None:
        return False

    debug_dir.mkdir(parents=True, exist_ok=True)
    screen = warp_to_frontal(image, corners, profile.screen_size)
    cv2.imwrite(str(debug_dir / "warped.png"), screen)

    binary = preprocess_screen(screen, invert=profile.invert)
    cv2.imwrite(str(debug_dir / "binary.png"), binary)

    cells = extract_all_digit_cells(binary, profile)
    for field_name, field_cells in cells.items():
        for i, cell in enumerate(field_cells):
            cv2.imwrite(str(debug_dir / f"{field_name}_{i}.png"), cell)
    return True


def print_table(result: ReadingResult, profile: DeviceProfile) -> None:
    name_w = max(len(n) for n in result.fields) + 2
    total_w = name_w + 34

    print(f"\nThiet bi: {profile.name}")
    print("-" * total_w)
    print(f"{'Truong':<{name_w}}{'Gia tri':<12}{'Confidence':<14}{'Trang thai'}")
    print("-" * total_w)
    for name, field in result.fields.items():
        value_str = str(field.value) if field.value is not None else "--"
        if field.value is None:
            status = "KHONG DOC DUOC"
        elif field.confidence < profile.min_confidence:
            status = "CAN KIEM TRA LAI"
        else:
            status = "OK"
        print(f"{name:<{name_w}}{value_str:<12}{field.confidence:<14.2f}{status}")
    print("-" * total_w)
    print(f"Tong the: {'HOP LE' if result.valid else 'KHONG HOP LE'}  (confidence={result.overall_confidence:.2f})")
    if result.errors:
        print("Loi:")
        for e in result.errors:
            print(f"  - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device-profile", type=Path, required=True)
    parser.add_argument(
        "--debug-dir", type=Path, default=None,
        help="Thu muc luu anh warp + tung o chu so de doi chieu bang mat (mac dinh: debug/<ten_anh>/)",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Khong doc duoc anh: {args.image}")

    profile = DeviceProfile.from_yaml(args.device_profile)
    debug_dir = args.debug_dir or Path("debug") / args.image.stem

    if save_debug_images(image, profile, debug_dir):
        print(f"Anh debug (warp + tung o chu so): {debug_dir}/")
    else:
        print("Khong tim thay man hinh LCD trong anh - khong luu duoc anh debug.")

    pipeline = ReadingPipeline(profile)

    try:
        result = pipeline.run(image)
    except ScreenNotFoundError as e:
        raise SystemExit(f"LOI: {e}")

    print_table(result, profile)


if __name__ == "__main__":
    main()
