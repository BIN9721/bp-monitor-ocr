#!/usr/bin/env python3
"""CLI chay pipeline tren mot anh don le.

Vi du:
    python scripts/run_inference.py --image data/samples/example.jpg \
        --device-profile configs/device_profiles/example_device.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.roi import DeviceProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device-profile", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Khong doc duoc anh: {args.image}")

    profile = DeviceProfile.from_yaml(args.device_profile)
    pipeline = ReadingPipeline(profile)

    try:
        result = pipeline.run(image)
    except ScreenNotFoundError as e:
        print(json.dumps({"valid": False, "errors": [str(e), "Vui long chup lai anh"]}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
