"""REST API de tich hop bp_ocr vao he thong phan mem khac (bat ky ngon ngu nao, qua HTTP).

Chay:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000

Vi du goi:
    curl -X POST http://localhost:8000/infer \
        -F "device_profile=omron_hem7121" \
        -F "image=@data/samples/example.jpg"

Xem INTEGRATION.md de biet chi tiet hop dong dau vao/dau ra va cach xu ly loi.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.roi import DeviceProfile

DEVICE_PROFILE_DIR = Path(__file__).parent / "configs" / "device_profiles"

app = FastAPI(
    title="BP Monitor OCR API",
    description="Nhan dang chi so 7-segment tren man hinh may do dien tu. "
    "KHONG phai thiet bi y te da kiem dinh - xem INTEGRATION.md truoc khi dung.",
    version="1.0.0",
)


def load_profiles() -> dict[str, DeviceProfile]:
    return {p.stem: DeviceProfile.from_yaml(p) for p in sorted(DEVICE_PROFILE_DIR.glob("*.yaml"))}


def decode_image(contents: bytes) -> np.ndarray | None:
    arr = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/device-profiles")
def list_device_profiles() -> dict:
    """Danh sach device profile hien co, de he thong goi biet truyen gia tri nao cho
    device_profile trong POST /infer."""
    profiles = load_profiles()
    return {
        name: {
            "fields": list(profile.fields.keys()),
            "screen_size": list(profile.screen_size),
        }
        for name, profile in profiles.items()
    }


@app.post("/infer")
async def infer(device_profile: str = Form(...), image: UploadFile = File(...)) -> dict:
    """Nhan 1 anh + ten device profile, tra ve ket qua nhan dang.

    200: nhan dang thanh cong (co the valid=false neu gia tri ngoai gioi han hop ly -
    van la HTTP 200 vi day la ket qua hop le cua thuat toan, khong phai loi he thong).
    404: device_profile khong ton tai.
    422: khong doc duoc anh, hoac khong phat hien duoc man hinh LCD trong anh (day la
    gioi han da biet cua thuat toan - xem INTEGRATION.md muc "Xu ly loi").
    """
    profiles = load_profiles()
    if device_profile not in profiles:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay device profile '{device_profile}'. Xem GET /device-profiles.",
        )

    contents = await image.read()
    img = decode_image(contents)
    if img is None:
        raise HTTPException(status_code=422, detail="Khong doc duoc anh - kiem tra dinh dang file.")

    pipeline = ReadingPipeline(profiles[device_profile])
    try:
        result = pipeline.run(img)
    except ScreenNotFoundError as e:
        return JSONResponse(
            status_code=422,
            content={
                "valid": False,
                "errors": [str(e), "Vui long chup lai anh: dam bao man hinh LCD ro net, khong bi vat khac che"],
            },
        )

    return result.to_dict()
