# Hướng dẫn tích hợp vào hệ thống phần mềm

Tài liệu này dành cho người muốn **gọi bp_ocr từ một hệ thống khác** (backend, app di
động, hệ thống theo dõi sức khỏe...), không phải chạy độc lập. Đọc trước
[README.md](README.md) mục "Cách hoạt động & Giới hạn" — các con số và cảnh báo ở đó áp
dụng y hệt khi tích hợp, không riêng gì khi chạy CLI.

## ⚠️ Trước khi tích hợp

- **Không phải thiết bị y tế đã kiểm định.** Đây là công cụ hỗ trợ nhập liệu (giảm thao
  tác gõ tay), không phải nguồn dữ liệu y tế đáng tin cậy tuyệt đối. Hệ thống tích hợp
  nên luôn cho người dùng xem lại/sửa số trước khi lưu vào hồ sơ sức khỏe.
- **Tỷ lệ phát hiện màn hình thực tế ~58%** trên ảnh sản phẩm đa dạng (xem README). Hệ
  thống gọi API/thư viện **phải** xử lý rõ nhánh thất bại (yêu cầu chụp lại ảnh), không
  được coi im lặng/exception là điều hiếm gặp.
- **Mỗi model máy đo cần 1 file device profile** đã hiệu chỉnh tay (`configs/device_profiles/*.yaml`).
  Máy đo chưa có profile sẽ không dùng được — không có cơ chế tự nhận diện model.

## 3 cách tích hợp

| Cách | Phù hợp khi | Chi phí tích hợp |
|---|---|---|
| [Thư viện Python](#1-thư-viện-python-tích-hợp-chặt) | Hệ thống backend viết bằng Python | Thấp nhất, import trực tiếp |
| [CLI / subprocess](#2-cli--subprocess) | Hệ thống ngôn ngữ khác, không muốn chạy service riêng | Trung bình, gọi qua subprocess + parse JSON |
| [REST API](#3-rest-api) | Hệ thống ngôn ngữ khác, kiến trúc microservice | Cần chạy 1 service riêng (`api.py`), nhưng ngôn ngữ gọi bất kỳ |

---

## 1. Thư viện Python (tích hợp chặt)

```bash
pip install -e /duong/dan/toi/bp-monitor-ocr
# hoac: pip install git+https://github.com/BIN9721/bp-monitor-ocr.git
```

```python
import cv2
from bp_ocr.pipeline import ReadingPipeline, ScreenNotFoundError
from bp_ocr.roi import DeviceProfile

profile = DeviceProfile.from_yaml("configs/device_profiles/omron_hem7121.yaml")
pipeline = ReadingPipeline(profile)

image = cv2.imread("anh_nguoi_dung_upload.jpg")  # hoac cv2.imdecode() tu bytes trong memory

try:
    result = pipeline.run(image)
except ScreenNotFoundError:
    # BAT BUOC xu ly nhanh nay - xay ra ~42% tren anh da dang thuc te
    ...  # bao nguoi dung chup lai anh

if not result.valid:
    # gia tri doc duoc nhung ngoai gioi han hop ly hoac confidence thap
    print(result.errors)
else:
    print(result.to_dict())
    # {"fields": {"sys": {"value": 118, "confidence": 1.0}, ...}, "overall_confidence": 0.86, "valid": true, "errors": []}
```

`ReadingPipeline` không giữ trạng thái giữa các lần gọi (`run()` không mutate `self`) —
an toàn dùng lại 1 instance cho nhiều ảnh liên tiếp, kể cả trong môi trường đa luồng
(mỗi thread nên tạo `ReadingPipeline` riêng nếu chạy song song, vì OpenCV/numpy operations
bên trong không được thiết kế thread-safe qua chung 1 object).

## 2. CLI / subprocess

Dùng khi hệ thống gọi không viết bằng Python. `scripts/run_inference.py` in JSON ra
stdout, exit code khác 0 khi thất bại:

```bash
python scripts/run_inference.py --image anh.jpg --device-profile configs/device_profiles/omron_hem7121.yaml
echo "exit code: $?"   # 0 = co ket qua (co the valid=false), 1 = khong phat hien man hinh
```

Ví dụ gọi từ Node.js:

```javascript
const { execFile } = require("child_process");

execFile("python3", [
  "scripts/run_inference.py",
  "--image", "anh.jpg",
  "--device-profile", "configs/device_profiles/omron_hem7121.yaml",
], { cwd: "/duong/dan/bp-monitor-ocr" }, (err, stdout) => {
  const result = JSON.parse(stdout);
  // err != null khi exit code = 1 (khong phat hien man hinh) - stdout van la JSON hop le
  console.log(result);
});
```

Chi phí: mỗi lần gọi khởi động lại Python interpreter (~0.3-0.5s overhead) — không phù
hợp nếu cần xử lý số lượng lớn ảnh/giây, dùng REST API (mục 3) cho trường hợp đó.

## 3. REST API

Chạy service (xem thêm `Dockerfile` để đóng gói container):

```bash
pip install fastapi uvicorn python-multipart   # da co san trong requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

### `GET /health`
Health check cho load balancer / orchestrator.
```json
{"status": "ok"}
```

### `GET /device-profiles`
Danh sách profile hiện có và các trường mỗi profile hỗ trợ — hệ thống gọi nên hiển thị
danh sách này cho người dùng chọn đúng model máy, thay vì hardcode tên profile.
```json
{"omron_hem7121": {"fields": ["sys", "dia", "pulse"], "screen_size": [640, 480]}, ...}
```

### `POST /infer`
Multipart form: `device_profile` (tên profile, string) + `image` (file).

```bash
curl -X POST http://localhost:8000/infer \
  -F "device_profile=omron_hem7121" \
  -F "image=@anh.jpg"
```

**Mã trạng thái HTTP — hệ thống gọi cần xử lý cả 3:**

| Mã | Khi nào | Body |
|---|---|---|
| `200` | Chạy xong pipeline (kể cả khi `valid: false` do ngoài giới hạn hợp lý) | `ReadingResult.to_dict()` — xem cấu trúc bên dưới |
| `404` | `device_profile` không tồn tại | `{"detail": "..."}` |
| `422` | Không đọc được file ảnh, HOẶC không phát hiện được màn hình LCD | `{"valid": false, "errors": [...]}` |

**Cấu trúc body khi 200:**
```json
{
  "fields": {
    "sys": {"value": 118, "confidence": 1.0},
    "dia": {"value": 78, "confidence": 0.86},
    "pulse": {"value": 70, "confidence": 0.86}
  },
  "overall_confidence": 0.86,
  "valid": true,
  "errors": []
}
```
`value` có thể là `int`, `float` (trường có `decimal_digits` trong profile, vd đơn vị
mmol/L), hoặc `null` (không nhận dạng chắc chắn được chữ số).

Ví dụ gọi từ hệ thống khác (bất kỳ ngôn ngữ nào hỗ trợ HTTP + multipart), minh họa bằng
Python `requests`:

```python
import requests

with open("anh.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/infer",
        data={"device_profile": "omron_hem7121"},
        files={"image": f},
    )

if resp.status_code == 422:
    # yeu cau nguoi dung chup lai anh
    ...
elif resp.status_code == 404:
    # loi cau hinh o phia he thong goi (sai ten profile)
    ...
else:
    result = resp.json()
    if not result["valid"]:
        # doc duoc nhung can nguoi dung xac nhan lai (ngoai gioi han / confidence thap)
        ...
```

### Đóng gói container

```bash
docker build -t bp-monitor-ocr-api .
docker run -p 8000:8000 bp-monitor-ocr-api
```

## Thêm model máy đo mới cho hệ thống đang tích hợp

Xem README mục "Thêm một loại máy đo mới" — quy trình giống hệt dù dùng qua thư viện,
CLI hay REST API, vì cả 3 đều dùng chung `configs/device_profiles/`. Thêm 1 file YAML
mới, không cần sửa code tích hợp.
