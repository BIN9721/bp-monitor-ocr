"""Orchestration toan bo pipeline, tu anh dau vao den ReadingResult (muc 4 trong thiet ke).

Tong quat cho bat ky loai may do dien tu nao hien thi so 7-net (huyet ap, duong huyet,
nhiet do...) - ten truong va gioi han hop le hoan toan do device profile (YAML) khai
bao, khong hardcode trong code.
"""

from __future__ import annotations

import numpy as np

from bp_ocr.postprocess import build_reading_result, combine_digits
from bp_ocr.preprocessing import preprocess_screen
from bp_ocr.roi import DeviceProfile, extract_all_digit_cells
from bp_ocr.rule_based import recognize_digit
from bp_ocr.schema import DigitPrediction, ReadingResult
from bp_ocr.screen_detection import find_screen_corners, warp_to_frontal


class ReadingPipeline:
    def __init__(self, device_profile: DeviceProfile):
        self.profile = device_profile

    def run(self, image: np.ndarray) -> ReadingResult:
        corners = find_screen_corners(image)
        if corners is None:
            raise ScreenNotFoundError("Khong phat hien duoc man hinh LCD trong anh")

        screen = warp_to_frontal(image, corners, self.profile.screen_size)
        binary = preprocess_screen(screen, invert=self.profile.invert)
        raw_cells = extract_all_digit_cells(binary, self.profile)

        fields = {}
        for field_name, cells in raw_cells.items():
            digit_predictions = [self._recognize_cell(cell) for cell in cells]
            decimal_digits = self.profile.fields[field_name].decimal_digits
            fields[field_name] = combine_digits(digit_predictions, decimal_digits=decimal_digits)

        valid_ranges = {
            name: fp.valid_range for name, fp in self.profile.fields.items() if fp.valid_range is not None
        }
        return build_reading_result(
            fields,
            valid_ranges=valid_ranges,
            greater_than=self.profile.greater_than,
            min_confidence=self.profile.min_confidence,
        )

    def _recognize_cell(self, cell: np.ndarray) -> DigitPrediction:
        digit, confidence = recognize_digit(cell)
        return DigitPrediction(digit=digit, confidence=confidence)


class ScreenNotFoundError(Exception):
    pass
