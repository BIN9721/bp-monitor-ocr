"""Ghep chu so thanh gia tri va kiem tra logic (muc 10 trong thiet ke).

Cac quy tac o day chi dung de phat hien loi nhan dang, KHONG duoc dung de tu
dong sua gia tri hoac chan doan tinh trang suc khoe. Gioi han hop le va quan
he giua cac truong (vd SYS phai lon hon DIA) do device profile khai bao, khong
hardcode ten truong de dung chung cho bat ky loai may do nao.
"""

from __future__ import annotations

from bp_ocr.schema import DigitPrediction, FieldResult, ReadingResult

DEFAULT_MIN_CONFIDENCE = 0.6


def combine_digits(digits: list[DigitPrediction], decimal_digits: int = 0) -> FieldResult:
    """Ghep danh sach digit (co the co blank o dau) thanh mot gia tri so.

    Vi du: [blank, 7, 6] -> 76. Neu bat ky digit khong-blank nao la None
    (khong nhan dang duoc) thi value = None.

    decimal_digits: so o TINH TU BEN PHAI la phan thap phan (vd [5, 8] voi
    decimal_digits=1 -> 5.8, dung cho may hien thi don vi mmol/L). Mac dinh 0
    (gia tri nguyen, khong doi hanh vi cu).
    """
    significant = []
    seen_nonblank = False
    for d in digits:
        if d.digit is None and not seen_nonblank:
            continue  # bo qua blank o dau
        seen_nonblank = True
        significant.append(d)

    if not significant or any(d.digit is None for d in significant):
        confidence = min((d.confidence for d in digits), default=0.0)
        return FieldResult(value=None, confidence=confidence, digits=digits)

    digit_str = "".join(str(d.digit) for d in significant)
    confidence = min(d.confidence for d in digits)

    if decimal_digits > 0:
        digit_str = digit_str.rjust(decimal_digits + 1, "0")
        split = len(digit_str) - decimal_digits
        value = float(f"{digit_str[:split]}.{digit_str[split:]}")
    else:
        value = int(digit_str)

    return FieldResult(value=value, confidence=confidence, digits=digits)


def validate_reading(
    fields: dict[str, FieldResult],
    valid_ranges: dict[str, tuple[float, float]] | None = None,
    greater_than: list[tuple[str, str]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[bool, list[str]]:
    """Kiem tra logic tren ket qua da ghep. Tra ve (valid, danh sach loi).

    valid_ranges: gioi han ky thuat hop ly cho tung truong (khong bat buoc khai bao het).
    greater_than: danh sach (a, b) yeu cau gia tri truong a phai lon hon truong b
    (vd [("sys", "dia")] cho may do huyet ap).
    """
    valid_ranges = valid_ranges or {}
    greater_than = greater_than or []
    errors: list[str] = []

    for name, result in fields.items():
        if result.value is None:
            errors.append(f"Khong nhan dang chac chan chi so {name.upper()}")
            continue
        if name in valid_ranges:
            low, high = valid_ranges[name]
            if not (low <= result.value <= high):
                errors.append(f"Gia tri {name.upper()}={result.value} nam ngoai gioi han hop ly [{low}, {high}]")
        if result.confidence < min_confidence:
            errors.append(f"Do tin cay {name.upper()} qua thap ({result.confidence:.2f})")

    for greater_name, lesser_name in greater_than:
        greater_field, lesser_field = fields.get(greater_name), fields.get(lesser_name)
        if not greater_field or not lesser_field:
            continue
        if greater_field.value is not None and lesser_field.value is not None:
            if greater_field.value <= lesser_field.value:
                errors.append(
                    f"{greater_name.upper()} ({greater_field.value}) phai lon hon "
                    f"{lesser_name.upper()} ({lesser_field.value})"
                )

    return (len(errors) == 0, errors)


def build_reading_result(
    fields: dict[str, FieldResult],
    valid_ranges: dict[str, tuple[float, float]] | None = None,
    greater_than: list[tuple[str, str]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ReadingResult:
    valid, errors = validate_reading(fields, valid_ranges, greater_than, min_confidence)
    confidences = [f.confidence for f in fields.values()]
    overall_confidence = min(confidences) if valid else min(confidences + [0.0])

    if not valid:
        errors = errors + ["Vui long chup lai anh"]

    return ReadingResult(
        fields=fields,
        overall_confidence=overall_confidence,
        valid=valid,
        errors=errors,
    )
