"""Tach vung SYS/DIA/PULSE va tung o chu so theo device profile (muc 9).

Moi o chu so co ROI rieng (khong chia deu tu ROI field) vi tren nhieu may do
huyet ap thuc te, khoang cach giua cac chu so trong cung mot truong khong deu
(vd: chu so hang tram tach rieng, co khoang trong lon truoc 2 chu so cuoi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class FieldProfile:
    digit_rois: list[tuple[int, int, int, int]]  # [(x, y, w, h), ...] cho tung o, tinh tren anh da warp
    valid_range: tuple[float, float] | None = None  # gioi han ky thuat hop ly cua truong nay (khong bat buoc)
    decimal_digits: int = 0  # so o chu so TINH TU BEN PHAI la phan thap phan (vd "5.8" voi 2 o -> 1)


@dataclass
class DeviceProfile:
    """Khai bao mot loai may do cu the (huyet ap, duong huyet, nhiet do...).

    Cac truong du lieu (fields) va gioi han hop le hoan toan do YAML khai bao,
    khong hardcode ten truong trong code de dung chung cho bat ky loai may do
    hien thi so 7-net nao.
    """

    name: str
    screen_size: tuple[int, int]  # width, height
    invert: bool
    fields: dict[str, FieldProfile]
    greater_than: list[tuple[str, str]] = field(default_factory=list)  # [(a, b), ...]: truong a phai > truong b
    min_confidence: float = 0.6

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeviceProfile":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        fields = {}
        for name, cfg in data["fields"].items():
            valid_range = tuple(cfg["valid_range"]) if "valid_range" in cfg else None
            fields[name] = FieldProfile(
                digit_rois=[tuple(roi) for roi in cfg["digit_rois"]],
                valid_range=valid_range,
                decimal_digits=cfg.get("decimal_digits", 0),
            )
        return cls(
            name=data["name"],
            screen_size=tuple(data["screen_size"]),
            invert=data.get("invert", False),
            fields=fields,
            greater_than=[tuple(pair) for pair in data.get("greater_than", [])],
            min_confidence=data.get("min_confidence", 0.6),
        )


def extract_digit_cells(screen: np.ndarray, field: FieldProfile) -> list[np.ndarray]:
    """Cat tung o chu so cua mot field theo digit_rois."""
    cells = []
    for x, y, w, h in field.digit_rois:
        cells.append(screen[y : y + h, x : x + w])
    return cells


def extract_all_digit_cells(
    screen: np.ndarray, profile: DeviceProfile
) -> dict[str, list[np.ndarray]]:
    """Tra ve {"sys": [cell0, cell1, cell2], "dia": [...], "pulse": [...]}."""
    return {
        field_name: extract_digit_cells(screen, field_profile)
        for field_name, field_profile in profile.fields.items()
    }
