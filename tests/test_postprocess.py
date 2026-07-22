from bp_ocr.postprocess import combine_digits, validate_reading
from bp_ocr.schema import DigitPrediction, FieldResult


def digit(value, confidence=0.95):
    return DigitPrediction(digit=value, confidence=confidence)


def test_combine_digits_ignores_leading_blank():
    result = combine_digits([digit(None), digit(7), digit(6)])
    assert result.value == 76


def test_combine_digits_all_filled():
    result = combine_digits([digit(1), digit(1), digit(8)])
    assert result.value == 118


def test_combine_digits_unrecognized_middle_digit_is_none():
    result = combine_digits([digit(None), digit(None), digit(6)])
    assert result.value == 6

    result_broken = combine_digits([digit(1), digit(None), digit(8)])
    assert result_broken.value is None


def test_combine_digits_decimal_point():
    """May do duong huyet don vi mmol/L hien thi dang '5.8' (2 o, 1 o thap phan)."""
    result = combine_digits([digit(5), digit(8)], decimal_digits=1)
    assert result.value == 5.8


def test_combine_digits_decimal_point_with_leading_blank():
    """Blank o dau van duoc bo qua truoc, decimal_digits tinh tu ben phai nen khong doi."""
    result = combine_digits([digit(None), digit(5), digit(8)], decimal_digits=1)
    assert result.value == 5.8


def test_combine_digits_decimal_point_pads_missing_integer_part():
    """Chi co 1 chu so nhung decimal_digits=1 -> hieu la '0.x'."""
    result = combine_digits([digit(8)], decimal_digits=1)
    assert result.value == 0.8


def test_validate_reading_sys_must_exceed_dia():
    fields = {
        "sys": FieldResult(value=70, confidence=0.9),
        "dia": FieldResult(value=76, confidence=0.9),
        "pulse": FieldResult(value=71, confidence=0.9),
    }

    valid, errors = validate_reading(fields, greater_than=[("sys", "dia")])
    assert not valid
    assert any("SYS" in e for e in errors)


def test_validate_reading_valid_case():
    fields = {
        "sys": FieldResult(value=118, confidence=0.9),
        "dia": FieldResult(value=76, confidence=0.9),
        "pulse": FieldResult(value=71, confidence=0.9),
    }

    valid, errors = validate_reading(
        fields,
        valid_ranges={"sys": (60, 260), "dia": (30, 150), "pulse": (30, 220)},
        greater_than=[("sys", "dia")],
    )
    assert valid
    assert errors == []


def test_validate_reading_single_field_device():
    """Thiet bi chi co 1 truong (vd may do duong huyet) - khong can greater_than."""
    fields = {"glucose": FieldResult(value=135, confidence=0.9)}

    valid, errors = validate_reading(fields, valid_ranges={"glucose": (20, 600)})
    assert valid
    assert errors == []


def test_validate_reading_out_of_range_without_declared_range_is_valid():
    """Truong khong khai bao valid_range thi khong bi kiem tra gioi han."""
    fields = {"glucose": FieldResult(value=9999, confidence=0.9)}

    valid, errors = validate_reading(fields)
    assert valid
    assert errors == []
