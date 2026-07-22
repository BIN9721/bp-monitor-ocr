#!/usr/bin/env python3
"""Danh gia pipeline theo 3 cap do: chu so, chi so, toan anh (muc 13 trong thiet ke).

Danh sach truong duoc lay tu chinh device profile (khong hardcode) nen dung duoc
cho bat ky loai may do nao. Yeu cau mot file nhan (JSON) dang, vi du voi may do
huyet ap (3 truong sys/dia/pulse):
{
  "img_001.jpg": {"sys": 118, "dia": 76, "pulse": 71},
  "img_002.jpg": {"sys": 132, "dia": 85, "pulse": 68}
}
hoac voi may do duong huyet (1 truong glucose):
{
  "img_001.jpg": {"glucose": 135}
}

Vi du:
    python scripts/evaluate.py --images data/raw --labels data/raw/labels.json \
        --device-profile configs/device_profiles/example_device.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.roi import DeviceProfile


def digit_string(value: int | None) -> str:
    return "" if value is None else str(value)


def evaluate(images_dir: Path, labels: dict, pipeline: ReadingPipeline) -> None:
    field_names = list(pipeline.profile.fields.keys())
    total_images = 0
    correct_images = 0
    field_correct = {f: 0 for f in field_names}
    field_total = {f: 0 for f in field_names}
    digit_correct = 0
    digit_total = 0
    reshoot_requested = 0

    for filename, truth in labels.items():
        image_path = images_dir / filename
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[bo qua] khong doc duoc anh: {image_path}")
            continue

        total_images += 1
        try:
            result = pipeline.run(image)
        except ScreenNotFoundError:
            reshoot_requested += 1
            continue

        if not result.valid:
            reshoot_requested += 1

        all_fields_correct = True
        for f in field_names:
            predicted = result.fields[f].value
            expected = truth[f]
            field_total[f] += 1
            if predicted == expected:
                field_correct[f] += 1
            else:
                all_fields_correct = False

            pred_str, exp_str = digit_string(predicted), digit_string(expected)
            max_len = max(len(pred_str), len(exp_str))
            pred_str, exp_str = pred_str.rjust(max_len, "\0"), exp_str.rjust(max_len, "\0")
            for pc, ec in zip(pred_str, exp_str):
                digit_total += 1
                if pc == ec:
                    digit_correct += 1

        if all_fields_correct:
            correct_images += 1

    print(f"So anh danh gia: {total_images}")
    print(f"Ty le yeu cau chup lai: {reshoot_requested / total_images:.2%}" if total_images else "N/A")
    print(f"Do chinh xac chu so: {digit_correct}/{digit_total} = {digit_correct / digit_total:.2%}" if digit_total else "N/A")
    for f in field_names:
        if field_total[f]:
            print(f"Do chinh xac {f.upper()}: {field_correct[f]}/{field_total[f]} = {field_correct[f] / field_total[f]:.2%}")
    if total_images:
        fields_label = "+".join(f.upper() for f in field_names)
        print(f"Do chinh xac toan anh ({fields_label} deu dung): {correct_images}/{total_images} = {correct_images / total_images:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--device-profile", type=Path, required=True)
    args = parser.parse_args()

    profile = DeviceProfile.from_yaml(args.device_profile)
    pipeline = ReadingPipeline(profile)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    evaluate(args.images, labels, pipeline)


if __name__ == "__main__":
    main()
