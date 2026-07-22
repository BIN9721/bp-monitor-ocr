"""Tien xu ly anh man hinh: xam hoa, tang tuong phan, threshold, morphology (muc 5.2)."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE de tang tuong phan cuc bo, chiu duoc anh sang khong deu."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """Threshold thich nghi. invert=True khi chu so sang tren nen toi."""
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return binary


def clean_morphology(binary: np.ndarray) -> np.ndarray:
    """Loai bo nhieu nho va lap day khe ho nho trong net LCD."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def preprocess_screen(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Pipeline day du: xam hoa -> tuong phan -> threshold -> morphology."""
    gray = to_grayscale(image)
    contrasted = enhance_contrast(gray)
    binary = binarize(contrasted, invert=invert)
    return clean_morphology(binary)
