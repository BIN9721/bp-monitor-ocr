"""Demo Streamlit: upload anh may do, chon device profile, xem ket qua quet so 7-doan.

Chay:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.preprocessing import preprocess_screen
from bp_ocr.roi import DeviceProfile, extract_all_digit_cells
from bp_ocr.screen_detection import find_screen_corners, warp_to_frontal

DEVICE_PROFILE_DIR = Path(__file__).parent / "configs" / "device_profiles"
UPLOAD_DEBUG_DIR = Path(__file__).parent / "debug" / "streamlit_uploads"


def load_profiles() -> dict[str, DeviceProfile]:
    return {path.name: DeviceProfile.from_yaml(path) for path in sorted(DEVICE_PROFILE_DIR.glob("*.yaml"))}


def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def uploaded_debug_dir(uploaded_file) -> Path:
    """Thu muc rieng cho moi anh upload, dat ten theo hash noi dung de khong bi
    ghi de khi upload nhieu anh trung ten nhung khac noi dung (hoac nguoc lai)."""
    digest = hashlib.sha1(uploaded_file.getvalue()).hexdigest()[:8]
    return UPLOAD_DEBUG_DIR / f"{Path(uploaded_file.name).stem}_{digest}"


def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Ve tu giac va 4 goc da phat hien len anh goc - de kiem tra bang mat xem
    screen_detection.py co bat dung vien man hinh LCD hay khong."""
    vis = image.copy()
    pts = corners.astype(int)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 255), 3)
    for pt in pts:
        cv2.circle(vis, tuple(pt), 8, (0, 255, 255), -1)
    return vis


st.set_page_config(page_title="BP Monitor OCR", page_icon="🩺", layout="wide")
st.title("Quet so 7-doan tren man hinh may do dien tu")
st.caption(
    "Upload anh chup man hinh may do (huyet ap, duong huyet, nhiet do...), chon device "
    "profile phu hop, xem ket qua nhan dang tung truong."
)

profiles = load_profiles()
if not profiles:
    st.error(f"Khong tim thay device profile nao trong {DEVICE_PROFILE_DIR}")
    st.stop()

col_upload, col_config = st.columns([2, 1])
with col_config:
    profile_file = st.selectbox("Device profile", list(profiles.keys()))
    profile = profiles[profile_file]
    st.markdown(
        f"**Ten thiet bi:** `{profile.name}`  \n"
        f"**Cac truong:** {', '.join(profile.fields.keys())}  \n"
        f"**screen_size:** {list(profile.screen_size)} &nbsp;&nbsp; **invert:** {profile.invert}"
    )
    with st.expander("digit_rois dang dung (de doi chieu neu nghi ngo profile cu)"):
        for field_name, field_profile in profile.fields.items():
            st.write(f"`{field_name}`: {field_profile.digit_rois}")
    show_debug = st.checkbox("Hien anh debug (warp / binary / tung o chu so)", value=True)

with col_upload:
    uploaded = st.file_uploader("Anh may do", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload mot anh de bat dau.")
    st.stop()

image = decode_uploaded_image(uploaded)
if image is None:
    st.error("Khong doc duoc anh, thu lai voi file khac.")
    st.stop()

session_dir = uploaded_debug_dir(uploaded)
session_dir.mkdir(parents=True, exist_ok=True)
(session_dir / uploaded.name).write_bytes(uploaded.getvalue())

corners = find_screen_corners(image)

input_cols = st.columns(2)
with input_cols[0]:
    st.subheader("Anh dau vao")
    st.image(bgr_to_rgb(image), width=360)
with input_cols[1]:
    st.subheader("Goc man hinh da phat hien")
    if corners is None:
        st.image(bgr_to_rgb(image), width=360)
        st.caption("Khong phat hien duoc tu giac nao - xem loi ben duoi.")
    else:
        st.image(bgr_to_rgb(draw_corners(image, corners)), width=360)
        st.caption("Neu khung do KHONG bao dung vien man hinh LCD, ket qua o duoi se sai do ROI bi lech.")

if corners is None:
    st.error("Khong phat hien duoc man hinh LCD trong anh. Thu anh khac hoac kiem tra device profile.")
    st.stop()

screen = warp_to_frontal(image, corners, profile.screen_size)
binary = preprocess_screen(screen, invert=profile.invert)

cv2.imwrite(str(session_dir / "corners.png"), draw_corners(image, corners))
cv2.imwrite(str(session_dir / "warped.png"), screen)
cv2.imwrite(str(session_dir / "binary.png"), binary)
st.caption(f"Da luu anh dau vao + anh debug tai: `{session_dir}`")

pipeline = ReadingPipeline(profile)
try:
    result = pipeline.run(image)
except ScreenNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("Ket qua")
status_label = "HOP LE" if result.valid else "KHONG HOP LE - can chup lai"
status_color = "green" if result.valid else "red"
st.markdown(f"**Trang thai:** :{status_color}[{status_label}]  &nbsp; (confidence tong the: {result.overall_confidence:.2f})")

rows = []
for name, field in result.fields.items():
    if field.value is None:
        status = "Khong doc duoc"
    elif field.confidence < profile.min_confidence:
        status = "Can kiem tra lai"
    else:
        status = "OK"
    rows.append(
        {
            "Truong": name,
            "Gia tri": field.value if field.value is not None else "--",
            "Confidence": round(field.confidence, 2),
            "Trang thai": status,
        }
    )
st.table(rows)

if result.errors:
    st.warning("\n".join(f"- {e}" for e in result.errors))

if show_debug:
    st.subheader("Anh debug")
    debug_cols = st.columns(2)
    with debug_cols[0]:
        st.image(bgr_to_rgb(screen), caption="Man hinh da warp chinh dien", width=320)
    with debug_cols[1]:
        st.image(binary, caption="Anh nhi phan sau preprocess", width=320)

    cells = extract_all_digit_cells(binary, profile)
    for field_name, field_cells in cells.items():
        st.markdown(f"**{field_name}**")
        cell_cols = st.columns(len(field_cells))
        for i, cell in enumerate(field_cells):
            with cell_cols[i]:
                st.image(cell, caption=f"o {i}", width=100)
