# CSDLDPT — Tra cứu video theo nội dung

Trích rút 324 đặc trưng chuyển động và hình dáng (nhóm A–F) từ video KTH, sau đó so sánh video qua dashboard chạy cục bộ.

Repository chứa mã nguồn, cấu hình, kiểm thử và mô hình PP-HumanSeg cần cho phân đoạn người. Dataset, kết quả trích rút, cơ sở dữ liệu dashboard và báo cáo cục bộ không được đưa lên GitHub.

## Cấu trúc

- `extract_videos.py`: trích rút đặc trưng, hỗ trợ chạy thử (`pilot`) và toàn bộ dữ liệu (`full`).
- `features_af.py`: tính 324 đặc trưng thuộc sáu nhóm A–F.
- `segmentation.py`: phân đoạn người và tinh chỉnh silhouette.
- `verify_results.py`: kiểm tra kết quả trích rút.
- `extraction_config.json`: cấu hình xử lý, đường dẫn và SHA-256 của mô hình.
- `video_dashboard.py`, `video_dashboard/`: máy chủ Python và giao diện tra cứu video.
- `models/`: mô hình PP-HumanSeg ONNX và giấy phép Apache 2.0 của mô hình.
- `tests/`: kiểm thử công thức và xử lý đặc trưng.

## Cài đặt trên Windows / PowerShell

Môi trường hiện tại sử dụng Python 3.13. Cài các phiên bản thư viện trong `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Mô hình `models/pphumanseg.onnx` được cung cấp kèm repository (khoảng 6 MB), có nguồn từ [OpenCV Zoo — PP-HumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg). Giấy phép của mô hình nằm trong `models/LICENSE`; checksum được kiểm tra khi trích rút.

## Chuẩn bị dataset riêng

Đặt video KTH đã chuẩn hóa vào thư mục `Dataset/`, theo cấu trúc:

```text
Dataset/
  boxing/person01_boxing_d1.mp4
  handclapping/...
  handwaving/...
  jogging/...
  running/...
  walking/...
```

Video đầu vào phải có kích thước **160 × 120**, tốc độ **25 fps**, định dạng MP4 và tên theo mẫu `personXX_<action>_dY.mp4`. Chế độ `pilot` sử dụng danh sách 12 video cố định trong `extract_videos.py`, vì vậy cần có đủ các tệp tương ứng. Dataset được giữ cục bộ và bị loại khỏi Git bằng `.gitignore`.

## Trích rút và kiểm tra

Chạy thử trước, sau đó xử lý toàn bộ dữ liệu:

```powershell
.\.venv\Scripts\python.exe extract_videos.py --mode pilot --workers 4
.\.venv\Scripts\python.exe verify_results.py --mode pilot
.\.venv\Scripts\python.exe extract_videos.py --mode full --workers 4
.\.venv\Scripts\python.exe verify_results.py --mode full
```

Các lệnh hỗ trợ `--dataset` và `--out` để thay đổi thư mục. Kết quả mặc định nằm trong `results/pilot/` và `results/full/`, gồm vector đặc trưng, manifest, schema, chỉ báo chất lượng và checkpoint. Kiểm tra chế độ `full` còn đối chiếu với kết quả `pilot`, nên cần chạy `pilot` trước.

## Chạy dashboard

Dashboard sử dụng dataset trong `Dataset/` và kết quả tại `results/full/`. Sau khi trích rút đầy đủ:

```powershell
.\.venv\Scripts\python.exe video_dashboard.py
```

Mở <http://127.0.0.1:8061>. Có thể đổi cổng bằng `--port 8080`. Cơ sở dữ liệu dashboard được tạo tự động trong `results/`.

## Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các kiểm thử công thức dùng dữ liệu tổng hợp, không yêu cầu tải dataset.
