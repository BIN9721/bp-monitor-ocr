"""Phat hien man hinh LCD va chinh ve goc nhin chinh dien (muc 8 trong thiet ke)."""

from __future__ import annotations

import cv2
import numpy as np


def find_screen_corners(image: np.ndarray) -> np.ndarray | None:
    """Tim 4 goc man hinh LCD bang contour hinh chu nhat lon nhat trong anh.

    Tra ve mang (4, 2) float32 theo thu tu [top-left, top-right, bottom-right,
    bottom-left], hoac None neu khong tim thay contour tu giac phu hop.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # RETR_LIST (khong chi RETR_EXTERNAL): tren nhieu may do, vien than may co the tao
    # thanh 1 contour khep kin lon hon va bao quanh vien man hinh LCD - RETR_EXTERNAL se
    # bo qua contour man hinh vi coi no la "long ben trong". RETR_LIST giu lai ca hai,
    # de vong lap ben duoi tu chon ung vien hinh chu nhat phu hop theo dien tich.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in candidates[:10]:
        area = cv2.contourArea(contour)
        if area < 0.05 * image_area:
            break  # contours da sap xep giam dan, cac contour sau con nho hon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))

    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sap xep 4 diem theo thu tu top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered[0] = pts[np.argmin(s)]       # top-left: x+y nho nhat
    ordered[2] = pts[np.argmax(s)]       # bottom-right: x+y lon nhat
    ordered[1] = pts[np.argmin(diff)]    # top-right: x-y nho nhat
    ordered[3] = pts[np.argmax(diff)]    # bottom-left: x-y lon nhat
    return ordered


def warp_to_frontal(image: np.ndarray, corners: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Ap dung perspective transform de dua man hinh ve nhin chinh dien.

    output_size: (width, height) cua anh dau ra.
    """
    width, height = output_size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (width, height))
