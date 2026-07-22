"""Kieu du lieu dung chung cho toan bo pipeline (xem muc 11 trong tai lieu thiet ke)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DigitPrediction:
    """Ket qua nhan dang mot chu so tai mot vi tri (co the la 'blank')."""

    digit: int | None  # None khi la blank hoac khong nhan dang duoc
    confidence: float


@dataclass
class FieldResult:
    """Ket qua cua mot truong (SYS/DIA/PULSE) sau khi ghep cac chu so."""

    value: int | float | None  # float khi truong co decimal_digits > 0 (vd mmol/L)
    confidence: float
    digits: list[DigitPrediction] = field(default_factory=list)


@dataclass
class ReadingResult:
    """Ket qua cuoi cung tra ve cho nguoi dung / he thong theo doi suc khoe.

    fields: ten truong -> ket qua, khai bao tu device profile (vd {"sys":..,"dia":..,
    "pulse":..} cho may do huyet ap, hoac {"glucose":..} cho may do duong huyet) - khong
    hardcode ten truong de dung chung cho bat ky loai may do hien thi so 7-net nao.
    """

    fields: dict[str, FieldResult]
    overall_confidence: float
    valid: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": {
                name: {"value": result.value, "confidence": result.confidence}
                for name, result in self.fields.items()
            },
            "overall_confidence": self.overall_confidence,
            "valid": self.valid,
            "errors": self.errors,
        }
