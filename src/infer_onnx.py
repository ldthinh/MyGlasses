import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import argparse
import time
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "blaze_face_short_range.tflite")

def _get_detector():
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Face detector model not found at {_MODEL_PATH}.\n"
            "Run preprocess_dataset.py first — it will auto-download the model."
        )
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.4,
    )
    return mp_vision.FaceDetector.create_from_options(options)

_detector = _get_detector()

def letterbox_image(image, expected_size):
    h, w = image.shape[:2]
    ih, iw = expected_size, expected_size
    scale = min(iw/w, ih/h)
    nw, nh = int(w * scale), int(h * scale)
    
    image = cv2.resize(image, (nw, nh))
    new_image = np.zeros((ih, iw, 3), dtype=np.uint8)
    
    # Tính toán padding
    pad_y = (ih - nh) // 2
    pad_x = (iw - nw) // 2
    new_image[pad_y:pad_y + nh, pad_x:pad_x + nw, :] = image
    return new_image

def preprocess_image(image_path, img_size=150):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe face detection (Tasks API) — crop trước khi đưa vào model
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result  = _detector.detect(mp_img)
    if result.detections:
        det     = max(result.detections, key=lambda d: d.categories[0].score)
        bbox    = det.bounding_box
        padding = 0.25
        x1 = max(0, int(bbox.origin_x - padding * bbox.width))
        y1 = max(0, int(bbox.origin_y - padding * bbox.height))
        x2 = min(w, int(bbox.origin_x + bbox.width  * (1 + padding)))
        y2 = min(h, int(bbox.origin_y + bbox.height * (1 + padding)))
        image = image[y1:y2, x1:x2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox_image(image, img_size)

    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def infer(onnx_path, image_path, classes):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    input_tensor = preprocess_image(image_path)
    
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end_time = time.time()
    
    preds = outputs[0][0] # first batch
    pred_idx = np.argmax(preds)
    
    print(f"Prediction: {classes[pred_idx]}")
    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="../outputs/face_shape_model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    # Needs to match the order from training exactly
    classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    infer(args.onnx, args.image, classes)
