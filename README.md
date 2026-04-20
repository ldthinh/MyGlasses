# MyGlasses — Face Shape Classifier

Mô hình phân loại hình dáng khuôn mặt (Face Shape Classification) sử dụng **MobileNetV3-Small**, tối ưu cho inference nhanh trên CPU.

## Classes

| # | Face Shape |
|---|-----------|
| 0 | Heart     |
| 1 | Oblong    |
| 2 | Oval      |
| 3 | Round     |
| 4 | Square    |

## Cấu trúc dự án

```
MyGlasses/
├── data/                           # Dataset gốc & dataset đã crop
├── outputs/                        # Checkpoints (.pth) và model ONNX
├── src/
│   ├── preprocess_dataset.py       # [Bước 1] MediaPipe face detection + crop
│   ├── dataset.py                  # Dataset & DataLoader (letterbox 150×150)
│   ├── model.py                    # EfficientNetV2S với 2-phase freeze/unfreeze
│   ├── train.py                    # 2-Phase fine-tuning training loop
│   ├── evaluate.py                 # Confusion matrix & classification report
│   ├── export_onnx.py              # Xuất model sang ONNX
│   └── infer_onnx.py               # Inference trên CPU (có MediaPipe crop)
├── download_data.py                # Tải dataset từ Kaggle
├── pyproject.toml
└── requirements.txt
```

## Cài đặt

```bash
# Clone project
git clone <repo-url> && cd MyGlasses

# Cài dependencies bằng uv
uv sync
```

## Chuẩn bị dữ liệu

### Cách 1: Tải tự động qua Kaggle API

```bash
# Đặt Kaggle API token vào ~/.kaggle/kaggle.json
# (Lấy tại: Kaggle.com → Settings → Create New API Token)
chmod 600 ~/.kaggle/kaggle.json

uv run python download_data.py
```

### Cách 2: Tải thủ công

Tải dataset tại [Face Shape Dataset (Kaggle)](https://www.kaggle.com/datasets/niten19/face-shape-dataset) và giải nén vào `data/face-shape-dataset/`.

Cấu trúc sau khi giải nén:
```
data/face-shape-dataset/FaceShape Dataset/
├── training_set/
│   ├── Heart/
│   ├── Oblong/
│   ├── Oval/
│   ├── Round/
│   └── Square/
└── testing_set/
    ├── Heart/
    ├── Oblong/
    ├── Oval/
    ├── Round/
    └── Square/
```

## Preprocessing — MediaPipe Face Crop (Bước quan trọng theo paper)

Sau khi tải data, chạy lệnh dưới đây **1 lần duy nhất** để detect và crop khuôn mặt bằng **MediaPipe**. Việc này giúp model chỉ "nhìn" vào phần mặt, loại bỏ tóc/nền ảnh làm nhiễu quá trình học:

```bash
cd src
uv run python preprocess_dataset.py \
    --input_dir "../data/face-shape-dataset/FaceShape Dataset" \
    --output_dir "../data/face-shape-cropped"
```


## Huấn luyện (2-Phase Fine-tuning)

Theo phương pháp của paper, training được chia làm 2 phase:
- **Phase 1**: Freeze toàn bộ backbone EfficientNetV2S, chỉ train classifier head với lr cao.
- **Phase 2**: Unfreeze N block cuối của backbone, fine-tune với lr rất nhỏ (để tránh catastrophic forgetting).

```bash
cd src

# Train với EfficientNetV2S (mặc định theo paper)
uv run python train.py

# Hoặc tuỳ chỉnh
uv run python train.py \
    --data_dir "../data/face-shape-cropped" \
    --model efficientnet_v2_s \
    --phase1_epochs 10 \
    --phase2_epochs 20 \
    --phase1_lr 1e-3 \
    --phase2_lr 1e-5 \
    --unfreeze_blocks 2 \
    --batch_size 32
```

Model tốt nhất được lưu tại `outputs/best_efficientnet_v2_s.pth`.

## Đánh giá

```bash
cd src
uv run python evaluate.py \
    --weights ../outputs/best_mobilenet_v3_small.pth
```

Kết quả gồm **Classification Report** (precision, recall, F1) và **Confusion Matrix** được lưu tại `outputs/confusion_matrix.png`.

## Xuất ONNX & Inference trên CPU

```bash
cd src

# Xuất model sang ONNX
uv run python export_onnx.py \
    --weights ../outputs/best_mobilenet_v3_small.pth \
    --output ../outputs/face_shape_model.onnx

# Chạy inference trên 1 ảnh
uv run python infer_onnx.py --image path/to/face.jpg
```

## Ghi chú kỹ thuật

- **Letterbox Padding**: Ảnh được resize giữ nguyên tỉ lệ (aspect ratio) và thêm viền đen, tránh biến dạng khuôn mặt ảnh hưởng đến việc nhận dạng shape.
- **Data Augmentation**: HorizontalFlip, ShiftScaleRotate, ColorJitter giúp tăng tính đa dạng dữ liệu training.
- **Pretrained weights**: Sử dụng ImageNet pretrained weights để transfer learning, giảm thời gian hội tụ.
