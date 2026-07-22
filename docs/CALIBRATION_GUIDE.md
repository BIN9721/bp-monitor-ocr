# Hướng dẫn tạo Device Profile cho model máy mới

Hướng dẫn từng bước, dành cho người mới, để thêm 1 model máy đo chưa có profile. Đọc
[ALGORITHM.md](../ALGORITHM.md) trước nếu bạn chưa hiểu vì sao `digit_rois` phải đo theo
đúng quy tắc ở đây — hiểu lý do sẽ giúp bạn tự sửa được khi đo sai, thay vì đoán mò.

Quy trình này chính là quy trình thật đã dùng để tạo tất cả các file trong
`configs/device_profiles/` — không có bước nào bị lược bớt.

## Bạn cần gì trước khi bắt đầu

- 1 ảnh chụp/ảnh sản phẩm rõ màn hình LCD của model máy đó.
- Biết **giá trị thật** đang hiển thị trên màn hình (để đối chiếu, không phải để "mớm"
  cho thuật toán).
- Môi trường đã cài xong (`pip install -r requirements.txt`).

## Bước 1 — Xác nhận `find_screen_corners` hoạt động

```python
import cv2, sys
sys.path.insert(0, "src")
from bp_ocr.screen_detection import find_screen_corners

img = cv2.imread("duong/dan/anh.jpg")
corners = find_screen_corners(img)
print(corners)  # None => dung lai o day, xem "Khi screen detection that bai" ben duoi
```

Nếu `corners` không phải `None`, vẽ thử lên ảnh gốc để xác nhận bằng mắt khung đó **có
ôm đúng viền màn hình LCD** hay không (không phải viền thân máy):

```python
vis = img.copy()
pts = corners.astype(int)
for i in range(4):
    cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 255), 3)
cv2.imwrite("check_corners.png", vis)
```

Nếu khung đỏ sai vị trí (ôm thân máy thay vì màn hình, hoặc lệch hẳn) — đây là lỗi Bước 1
trong `ALGORITHM.md`, không phải lỗi tọa độ ROI, đừng đo `digit_rois` cho tới khi bước
này đúng.

## Bước 2 — Warp thử + chọn `screen_size`

```python
from bp_ocr.screen_detection import warp_to_frontal

screen_size = (640, 480)  # chon 1 kich thuoc chuan, du lon de doc duoc chi tiet
warped = warp_to_frontal(img, corners, screen_size)
cv2.imwrite("warped.png", warped)
```

Mở `warped.png` lên xem — đây sẽ là "hệ tọa độ" bạn dùng cho toàn bộ `digit_rois` sau
này. Nếu ảnh nguồn có độ phân giải thấp (vùng màn hình gốc dưới ~100px), phóng to lên
640×480 có thể tạo viền/góc nhiễu do nội suy — xem mục "Ảnh độ phân giải thấp" bên dưới.

## Bước 3 — Xác định `invert`

```python
from bp_ocr.preprocessing import preprocess_screen

for invert in (True, False):
    binary = preprocess_screen(warped, invert=invert)
    cv2.imwrite(f"binary_invert_{invert}.png", binary)
```

Mở cả 2 ảnh, chọn cái mà **chữ số hiện thành màu trắng rõ nét trên nền đen** (đúng quy
ước 255=nét chữ mà `rule_based.py` giả định). Hầu hết màn hình LCD chữ số tối trên nền
sáng → `invert: true`.

## Bước 4 — Đo `digit_rois`

**Quy tắc bắt buộc (nhắc lại từ ALGORITHM.md):** mỗi ROI phải rộng bằng **cả ô chữ số**
(như đang chứa số "8"), không phải bounding box sát nét chữ đang hiển thị.

Có 2 cách đo, dùng cách nào tùy chất lượng ảnh:

### Cách A — Contour tự động (nhanh, dùng khi ảnh nét, không nhiễu viền)

```python
h, w = binary.shape
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = [cv2.boundingRect(c) for c in contours]
boxes = [b for b in boxes if b[2]*b[3] > 150]          # loc nhieu nho
boxes.sort(key=lambda b: (b[1], b[0]))                  # sap theo tren-xuong, trai-qua-phai
for x, y, ww, hh in boxes:
    print(f"x={x} y={y} w={ww} h={hh} (x2={x+ww} y2={y+hh})")
```

In ra tọa độ từng mảng sáng — đối chiếu bằng mắt xem mảng nào ứng với chữ số nào (một
chữ số có thể tách thành 2 contour, vd "0"/"8" tách thành vòng trên + vòng dưới do đường
chéo giữa — gộp lại bằng cách lấy min(x,y) / max(x2,y2) của các mảng cùng vị trí).

**Sau khi có tọa độ tight (bó sát) từng chữ số, phải nới rộng thành "cả ô":**
- Nếu vài chữ số trong field đều "đầy đủ" (0, 2, 3, 5, 6, 8, 9 — có thanh ngang trên/dưới)
  thì bounding box của chúng thường **đã** xấp xỉ đúng kích thước 1 ô — chỉ cần đệm thêm
  ~8-10px mỗi cạnh.
- Riêng chữ số **"1"** hầu như luôn có bounding box hẹp hơn hẳn (chỉ 2 nét dọc) — **không
  suy ra "ô" của nó bằng cách đệm quanh bounding box hẹp này**. Thay vào đó, dùng cùng độ
  rộng với chữ số "đầy đủ" bên cạnh (0/8...) trong cùng field, căn sao cho cạnh phải của
  hộp "1" thẳng hàng với cạnh phải bounding box thật của nó (vì số "1" thường in lệch về
  bên phải ô, khớp với vị trí đoạn b/c trong `SEGMENT_REGIONS`).

### Cách B — Lưới tọa độ thủ công (chậm hơn nhưng đáng tin cậy hơn, dùng khi ảnh nhiễu/độ phân giải thấp)

```python
vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if len(binary.shape) == 2 else binary.copy()
h, w = binary.shape[:2]
for x in range(0, w, 40):
    cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    cv2.putText(vis, str(x), (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
for y in range(0, h, 40):
    cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    cv2.putText(vis, str(y), (2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.imwrite("grid.png", vis)
```

Mở `grid.png`, đọc trực tiếp tọa độ pixel của từng ô chữ số bằng mắt (cộng thêm lưới
tham chiếu 40px một để ước lượng nhanh). Chậm hơn Cách A nhưng không bị ảnh hưởng bởi
nhiễu contour — dùng cách này nếu Cách A cho ra 1 contour khổng lồ bao trùm cả ảnh (dấu
hiệu ảnh bị nhiễu viền, xem mục dưới).

## Bước 5 — Viết file device profile

Tạo `configs/device_profiles/ten_may_moi.yaml`:

```yaml
name: ten_may_moi
screen_size: [640, 480]   # dung dung gia tri da chon o Buoc 2
invert: true              # dung dung gia tri da chon o Buoc 3

fields:
  glucose:                # doi ten truong cho phu hop (sys/dia/pulse, glucose, temperature...)
    digit_rois:
      - [x0, y0, w0, h0]   # o dau tien (tu trai qua phai)
      - [x1, y1, w1, h1]
      - [x2, y2, w2, h2]
    valid_range: [20, 600]     # gioi han ky thuat hop ly (khong bat buoc)
    decimal_digits: 0          # > 0 neu man hinh hien thi so thap phan (vd don vi mmol/L)
```

## Bước 6 — Kiểm thử

```bash
python scripts/visual_report.py --image duong/dan/anh.jpg \
    --device-profile configs/device_profiles/ten_may_moi.yaml \
    --output test_report.png
```

Mở `test_report.png` — mục "BƯỚC 2" sẽ vẽ khung màu quanh từng ô: **xanh** = confidence
đạt ngưỡng, **đỏ** = không đạt. Đối chiếu số hiển thị trong khung với giá trị thật.

Nếu đúng hết → xong. Nếu sai, xem tiếp mục dưới.

## Khi kết quả sai — debug theo bảng trong ALGORITHM.md

Tra bảng "Lỗi thực tế → bước nào gây ra" trong
[ALGORITHM.md](../ALGORITHM.md#lỗi-thực-tế--bước-nào-gây-ra). 2 trường hợp hay gặp nhất
khi mới tạo profile:

**Số vô nghĩa hoàn toàn (vd đọc ra vài trăm trong khi máy chỉ hiện 2 chữ số):** gần như
chắc chắn `digit_rois` sai vị trí — kiểm tra lại có nhầm ảnh/profile không, hoặc tọa độ đo
sai từ Bước 4.

**Đọc gần đúng nhưng sai đúng 1 chữ số (thường là số "1" hoặc số có nét mảnh):** dùng
đoạn script sau để xem chính xác đoạn a-g nào đo sai, so với ví dụ tính tay trong
`ALGORITHM.md`:

```python
from bp_ocr.rule_based import measure_segment_ratios, recognize_digit

cell = cv2.imread("duong/dan/o_chu_so_sai.png", cv2.IMREAD_GRAYSCALE)  # tu debug_dir cua quick_test.py
print(measure_segment_ratios(cell))
print(recognize_digit(cell))
```

Nếu 1 đoạn "phải bật" đo được ratio thấp hơn nhưng gần ngưỡng 0.30 → khung đo đoạn đó
(`SEGMENT_REGIONS` trong `rule_based.py`) có thể cần điều chỉnh **cho toàn bộ dự án**
(ảnh hưởng mọi device profile) — cân nhắc kỹ, đối chiếu với dữ liệu của các profile khác
đang đúng trước khi đổi (xem cách đã làm trong lịch sử commit khi sửa khung đoạn "a").
Thường thì sửa lại `digit_rois` (chỉ ảnh hưởng riêng profile đang làm) là lựa chọn an
toàn hơn.

## Ảnh độ phân giải thấp

Nếu vùng màn hình trong ảnh gốc dưới ~100px, phóng to lên `screen_size` (thường 640×480)
là phóng ~5-9 lần — có thể sinh viền/góc đen do nội suy ở `warpPerspective`, khiến Cách A
(contour tự động) trả về 1 contour khổng lồ bao trùm gần hết ảnh thay vì từng chữ số
riêng. Dấu hiệu nhận biết: kiểm tra tỷ lệ pixel trắng và giá trị 4 góc/viền ảnh nhị phân:

```python
import numpy as np
print("ty le trang:", np.count_nonzero(binary) / binary.size)   # bat thuong neu > 0.3-0.4
print("vien tren:", binary[0, ::80])
```

Nếu viền toàn giá trị 255 bất thường → chuyển sang **Cách B** (lưới tọa độ thủ công), bỏ
qua vùng viền khi đọc tọa độ.

## Font chữ in nghiêng (italic)

`SEGMENT_REGIONS` hiện giả định đoạn ngang nằm ngang, đoạn dọc thẳng đứng. Một số máy
dùng font 7-đoạn in nghiêng (đoạn `a`/`d` là hình bình hành chéo thay vì hình chữ nhật
ngang) — trường hợp này **không có cách khắc phục chỉ bằng chỉnh `digit_rois`**, vì vấn đề
nằm ở hình dạng vùng đo, không phải vị trí ô. Đây là giới hạn đã biết của thuật toán (xem
README mục "Cách hoạt động & Giới hạn") — nếu gặp, báo cáo lại kèm ảnh mẫu thay vì cố đoán
tọa độ, vì khả năng cao sẽ không hội tụ về kết quả đúng dù đo lại bao nhiêu lần.

## Checklist tổng kết

- [ ] `find_screen_corners` trả về khung đúng vị trí màn hình (không phải thân máy)
- [ ] Đã chọn `invert` đúng (chữ trắng trên nền đen ở ảnh nhị phân)
- [ ] Mỗi `digit_rois` rộng bằng "cả ô" (đặc biệt số "1", không đo bó sát)
- [ ] Đã khai báo `decimal_digits` nếu máy hiện số thập phân
- [ ] `scripts/visual_report.py` cho khung xanh + số đúng ở toàn bộ các ô
- [ ] Đã thử `min_confidence`/`valid_range` hợp lý cho model đó (không bắt buộc, nhưng
      nên có để hệ thống tự phát hiện được kết quả đáng ngờ)
