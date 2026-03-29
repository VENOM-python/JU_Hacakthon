"""
app.py - Flask Web Application for Fish Freshness Assessment
============================================================
Features:
  • Image upload (drag-and-drop + file dialog)
  • Real-time camera capture (OpenCV + MJPEG stream)
  • Grad-CAM++ heatmap overlay
  • Confidence bar visualization
  • Freshness score gauge
  • ONNX Runtime inference option (faster on CPU)
"""

import os, io, base64, time, threading
from pathlib import Path
from flask import (Flask, render_template, request, jsonify,
                   Response, send_from_directory)
from PIL import Image
import numpy as np
import cv2
import torch

# ── Local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import FishInferenceEngine, get_transforms, IMG_SIZE_S2
from gradcam import build_gradcam, GradCAM

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT",  "checkpoints/best_model.pth")
ONNX_PATH  = os.environ.get("ONNX_PATH",   "checkpoints/fish_freshness.onnx")
USE_ONNX   = os.environ.get("USE_ONNX",    "0") == "1"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_MB = 10
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# ── Model ────────────────────────────────────────────────────────────────────
print("Loading inference engine …")
try:
    engine = FishInferenceEngine(
        checkpoint_path=CHECKPOINT,
        use_onnx=USE_ONNX,
        onnx_path=ONNX_PATH if USE_ONNX else None,
    )
    if not USE_ONNX:
        gradcam = build_gradcam(engine.model, method="gradcam++")
    else:
        gradcam = None
    print("Model loaded ✓")
except FileNotFoundError:
    print("⚠  Checkpoint not found — running in DEMO mode (random predictions)")
    engine  = None
    gradcam = None


# ── Camera ───────────────────────────────────────────────────────────────────
class CameraStream:
    """Thread-safe OpenCV camera wrapper for MJPEG streaming."""

    def __init__(self, src: int = 0):
        self.cap     = cv2.VideoCapture(src)
        self.frame   = None
        self.running = False
        self.lock    = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


camera = CameraStream(src=0)
# camera.start()   # Uncomment to enable live camera on startup


# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(pil_image: Image.Image, use_tta: bool = True):
    """Run model + GradCAM and return a JSON-serialisable result dict."""
    if engine is None:
        # Demo mode
        import random
        classes = ["Fresh", "Medium", "Spoiled"]
        pred    = random.choice(classes)
        conf    = round(random.uniform(60, 99), 2)
        return {
            "class":           pred,
            "confidence":      conf,
            "freshness_score": round({"Fresh": 90, "Medium": 50, "Spoiled": 10}[pred]
                                     + random.uniform(-10, 10), 1),
            "all_probs":       {c: round(100 / 3, 2) for c in classes},
            "heatmap_b64":     None,
            "overlay_b64":     None,
            "latency_ms":      0,
        }

    t0  = time.perf_counter()
    res = engine.predict(pil_image, use_tta=use_tta, tta_n=6)
    t1  = time.perf_counter()

    heatmap_b64 = None
    overlay_b64 = None

    if gradcam is not None:
        try:
            transform = get_transforms("val", IMG_SIZE_S2)
            tensor    = transform(pil_image).unsqueeze(0).to(engine.device)
            cam       = gradcam.generate(tensor, target_class=None)
            overlay   = GradCAM.overlay_on_image(pil_image, cam, alpha=0.45)
            heatmap_b64 = pil_to_base64(
                Image.fromarray(np.uint8(255 * cv2.resize(cam, pil_image.size)))
            )
            overlay_b64 = pil_to_base64(overlay)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    res["heatmap_b64"] = heatmap_b64
    res["overlay_b64"] = overlay_b64
    res["latency_ms"]  = round((t1 - t0) * 1000, 1)
    return res


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        pil_image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {e}"}), 400

    use_tta = request.form.get("tta", "true").lower() == "true"
    result  = run_inference(pil_image, use_tta=use_tta)
    return jsonify(result)


@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    """Handle base64-encoded image (from webcam snapshot)."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        header, b64_data = data["image"].split(",", 1)
        img_bytes = base64.b64decode(b64_data)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot decode image: {e}"}), 400

    use_tta = data.get("tta", True)
    result  = run_inference(pil_image, use_tta=use_tta)
    return jsonify(result)


@app.route("/camera_stream")
def camera_stream():
    """MJPEG stream from webcam."""
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
            time.sleep(1 / 24)   # 24 fps cap

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/start", methods=["POST"])
def camera_start():
    camera.start()
    return jsonify({"status": "camera started"})


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    camera.stop()
    return jsonify({"status": "camera stopped"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  "loaded" if engine else "demo",
        "device": str(engine.device) if engine else "n/a",
        "onnx":   USE_ONNX,
    })


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)
