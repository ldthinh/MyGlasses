"""
Preprocessing script: Dùng MediaPipe Face Detector (Tasks API, v0.10+)
để crop khuôn mặt từ dataset gốc và lưu lại vào thư mục mới.

Chạy script này 1 lần duy nhất trước khi training.

Usage:
    uv run python preprocess_dataset.py \
        --input_dir "../data/face-shape-dataset/FaceShape Dataset" \
        --output_dir "../data/face-shape-cropped"
"""

import os
import sys
import cv2
import argparse
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ─── Download model tflite nếu chưa có ────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")

def _download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe face detector model (~1 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"  Saved to {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Cannot download model: {e}")
            print("  Please download manually from:")
            print(f"  {MODEL_URL}")
            sys.exit(1)

# ─── Khởi tạo detector (Tasks API) ────────────────────────────────────────────
def _build_detector(min_confidence: float = 0.4):
    _download_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def crop_faces_mediapipe(input_dir: str, output_dir: str, padding: float = 0.25):
    """
    Duyệt qua toàn bộ ảnh trong input_dir, detect face bằng MediaPipe Tasks API,
    crop và lưu vào output_dir với cùng cấu trúc thư mục.

    Args:
        padding: Tỉ lệ padding thêm vào quanh bbox để không cắt sát mặt (0.25 = 25%)
    """
    detector = _build_detector()

    splits   = ["training_set", "testing_set"]
    skipped  = 0
    processed = 0

    for split in splits:
        split_input  = os.path.join(input_dir,  split)
        split_output = os.path.join(output_dir, split)

        if not os.path.exists(split_input):
            print(f"[SKIP] {split_input} not found")
            continue

        classes = sorted(os.listdir(split_input))
        for cls_name in classes:
            cls_input  = os.path.join(split_input,  cls_name)
            cls_output = os.path.join(split_output, cls_name)

            if not os.path.isdir(cls_input):
                continue

            os.makedirs(cls_output, exist_ok=True)
            img_files = [f for f in os.listdir(cls_input)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

            for fname in tqdm(img_files, desc=f"{split}/{cls_name}", leave=False):
                src_path = os.path.join(cls_input,  fname)
                dst_path = os.path.join(cls_output, fname)

                if os.path.exists(dst_path):
                    continue  # đã xử lý rồi, bỏ qua

                bgr = cv2.imread(src_path)
                if bgr is None:
                    skipped += 1
                    continue

                h, w = bgr.shape[:2]
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = detector.detect(mp_img)

                if result.detections:
                    # Lấy detection có score cao nhất
                    det  = max(result.detections, key=lambda d: d.categories[0].score)
                    bbox = det.bounding_box

                    # bbox từ Tasks API là pixel-absolute
                    x1 = int(bbox.origin_x - padding * bbox.width)
                    y1 = int(bbox.origin_y - padding * bbox.height)
                    x2 = int(bbox.origin_x + bbox.width  * (1 + padding))
                    y2 = int(bbox.origin_y + bbox.height * (1 + padding))

                    # Clamp vào trong biên ảnh
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    cropped = bgr[y1:y2, x1:x2]
                    if cropped.size == 0:
                        cropped = bgr   # fallback
                else:
                    # Không detect được mặt → giữ ảnh gốc
                    cropped = bgr
                    skipped += 1

                cv2.imwrite(dst_path, cropped)
                processed += 1

    detector.close()
    print(f"\nDone!  Processed: {processed} | No face detected (used original): {skipped}")
    print(f"Cropped dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str,
                        default="../data/")
    parser.add_argument("--output_dir", type=str,
                        default="../data/face-shape-cropped")
    parser.add_argument("--padding",    type=float, default=0.25,
                        help="Padding ratio around face bounding box")
    args = parser.parse_args()

    crop_faces_mediapipe(args.input_dir, args.output_dir, args.padding)
