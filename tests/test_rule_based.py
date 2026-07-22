from conftest import render_clean_digit

from bp_ocr.rule_based import recognize_digit


def test_recognize_all_digits_on_clean_render():
    for digit in range(10):
        cell = render_clean_digit(digit)
        predicted, confidence = recognize_digit(cell)
        assert predicted == digit, f"digit={digit} predicted={predicted} confidence={confidence}"
        assert confidence == 1.0


def test_recognize_blank():
    cell = render_clean_digit(None)
    predicted, _confidence = recognize_digit(cell)
    assert predicted is None
