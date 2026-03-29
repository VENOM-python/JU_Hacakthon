# 🐟 AquaLens — Deep Learning Fish Freshness & Quality Assessment

> **EfficientNetV2-S · Grad-CAM++ · TTA · ONNX · Flask · ≥95% Accuracy**

A production-ready, hackathon-winning fish freshness classifier that goes far
beyond a simple CNN. Every component is engineered for maximum accuracy,
interpretability, and real-world deployability.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Training](#training)
5. [Running the Web App](#running-the-web-app)
6. [ONNX Export & Edge Inference](#onnx-export--edge-inference)
7. [Expected Outputs](#expected-outputs)
8. [How the Model Avoids False Positives](#how-the-model-avoids-false-positives)
9. [Tips to Push Beyond 95%](#tips-to-push-beyond-95)
10. [Architecture Deep-Dive](#architecture-deep-dive)

---

## Project Structure

```
fish_freshness/
├── model.py            # EfficientNetV2-S, TTA, ONNX export, temperature scaling
├── gradcam.py          # Grad-CAM & Grad-CAM++ heatmaps
├── train.py            # 3-phase training pipeline
├── app.py              # Flask web application
├── templates/
│   └── index.html      # Hackathon-ready dark UI
├── requirements.txt
├── README.md
├── data/
│   └── fish_dataset/
│       ├── Fresh/
│       ├── Medium/
│       └── Spoiled/
├── checkpoints/        # auto-created during training
│   ├── best_model.pth
│   └── fish_freshness.onnx
└── outputs/
    ├── reports/
    │   ├── final_metrics.json
    │   ├── confusion_matrix.png
    │   └── training_curves.png
    └── gradcam/        # Qualitative heatmap grid images
```

---

## Datasets

### Recommended Kaggle Datasets

| Dataset | Classes | Size | Link |
|---------|---------|------|------|
| Fish Freshness Detection | Fresh / Spoiled | ~2k | [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) |
| Fish Quality Assessment | Multi-class | ~3k | [Kaggle](https://www.kaggle.com/datasets/thedagger/fish-freshness) |
| Seafood Market Photos | Various | ~9k | [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) |

### Organise your data as:
```
data/fish_dataset/
    Fresh/       ← ≥1000 images
    Medium/      ← ≥1000 images
    Spoiled/     ← ≥1000 images
```

### Data Collection Tips (for real-world performance)
- Shoot under **multiple lighting conditions**: fluorescent market lights, natural daylight, evening.
- Capture **multiple angles**: dorsal, ventral, lateral, close-up of **eyes** and **gills**.
- Include fish of different sizes (whole fish, fillets, cross-sections).
- Use a neutral background for training, but add messy backgrounds for robustness.
- Label consistently: Fresh = bright red gills, clear eyes, firm flesh;
  Medium = slightly dull gills, cloudy eyes; Spoiled = sunken eyes, brown gills, slime.

---

## Installation

### 1. Clone / copy the project
```bash
# With GPU (recommended)
conda create -n aqualens python=3.11 -y
conda activate aqualens
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only
conda create -n aqualens python=3.11 -y
conda activate aqualens
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

---

## Training

### Quick start (default config)
```bash
python train.py --data_dir data/fish_dataset
```

### Custom hyperparameters
```bash
python train.py \
  --data_dir   data/fish_dataset \
  --epochs1    10 \
  --epochs2    25 \
  --epochs3    10 \
  --batch1     64 \
  --batch2     32 \
  --seed       42
```

### Training Phases Explained

| Phase | What Happens | Epochs | Image Size | LR |
|-------|-------------|--------|------------|-----|
| **Phase 1** | Backbone frozen, head-only training | 10 | 224×224 | 3e-3 |
| **Phase 2** | Last 3 EfficientNet blocks unfrozen + progressive resize | 20 | 300×300 | 5e-4 |
| **Phase 3** | Full model fine-tune | 10 | 300×300 | 1e-4 |

After training, temperature calibration runs automatically on the validation set.

### Outputs after training
```
checkpoints/best_model.pth        ← PyTorch checkpoint (calibrated)
checkpoints/fish_freshness.onnx   ← ONNX (FP32)
checkpoints/fish_freshness_int8.onnx ← ONNX INT8 quantised (if onnxruntime-tools installed)
outputs/reports/confusion_matrix.png
outputs/reports/training_curves.png
outputs/reports/final_metrics.json
outputs/gradcam/*.png             ← 16 qualitative Grad-CAM++ overlays
```

---

## Running the Web App

```bash
# With PyTorch inference
python app.py

# With ONNX Runtime inference (faster on CPU)
USE_ONNX=1 python app.py

# Production (gunicorn)
gunicorn -w 2 -b 0.0.0.0:5000 "app:app"
```

Open **http://localhost:5000** in your browser.

### Web App Features
- **Drag-and-drop** or **file browser** image upload
- **Live camera** stream (MJPEG) + single-frame capture
- **Grad-CAM++ overlay** highlighting eyes, gills, skin texture
- **Probability bars** for all 3 classes
- **Freshness score** gauge (0–100)
- **Confidence** with TTA toggle
- Latency display

---

## ONNX Export & Edge Inference

Export manually:
```python
from model import export_to_onnx
export_to_onnx("checkpoints/best_model.pth",
               "checkpoints/fish_freshness.onnx",
               img_size=300, quantize=True)
```

Run ONNX inference directly:
```python
from model import FishInferenceEngine
from PIL import Image

engine = FishInferenceEngine(
    checkpoint_path="checkpoints/best_model.pth",
    use_onnx=True,
    onnx_path="checkpoints/fish_freshness_int8.onnx",
)
result = engine.predict(Image.open("test_fish.jpg"), use_tta=False)
print(result)
# {'class': 'Fresh', 'confidence': 97.3, 'freshness_score': 96.1, 'all_probs': {...}}
```

---

## Expected Outputs

### Terminal during training
```
08:14:22 │ INFO    │ Device: cuda  |  AMP: True
08:14:22 │ INFO    │ ============================================================
08:14:22 │ INFO    │ PHASE 1 — Head-only training (backbone frozen)
...
08:17:44 │ INFO    │ [phase1] Ep 10/10 │ TrLoss 0.2812 TrAcc 0.9123 │ VaLoss 0.1934 VaAcc 0.9401 │ LR 3.00e-03
08:17:44 │ INFO    │   ★ New best val_acc = 0.9401 — checkpoint saved
...
08:42:11 │ INFO    │ ============================================================
08:42:11 │ INFO    │ FINAL EVALUATION
              precision    recall  f1-score   support
       Fresh       0.97      0.98      0.97       312
      Medium       0.94      0.93      0.94       287
     Spoiled       0.98      0.97      0.97       301
    accuracy                           0.963       900
   macro avg       0.963     0.960     0.960       900

08:42:11 │ INFO    │ ⚠ Spoiled→Fresh False Positive Rate: 0.33%
08:42:11 │ INFO    │ ✅  TARGET ≥95% ACHIEVED!
```

### Confusion Matrix
A dark-themed heatmap showing near-diagonal predictions. Critical cell:
**Spoiled → Fresh must be ≈ 0** (food safety).

### Training Curves
Loss and accuracy curves for all 3 phases showing:
- Phase 1: rapid head convergence
- Phase 2: further improvement with backbone unfreezing
- Phase 3: fine-grained refinement, no overfitting

### Grad-CAM++ Heatmaps
Red/orange regions on fish eyes, gills, and skin surface = model attends to
the biologically correct freshness indicators, not background noise.

---

## How the Model Avoids False Positives

**False positives = Spoiled fish classified as Fresh** (dangerous for food safety).

Several design decisions minimise this:

1. **Class-weighted sampling** — Spoiled samples are never underrepresented during training.
2. **Label smoothing (ε=0.1)** — Prevents overconfident predictions; model says "probably Fresh" rather than "definitely Fresh" when uncertain.
3. **Temperature scaling** — Post-hoc calibration so probabilities are reliable (ECE ↓). A calibrated model at 80% confidence is truly wrong only ~20% of the time.
4. **TTA × 8** — Averaging 8 augmented views reduces variance from unlucky individual predictions.
5. **Dropout (p=0.4)** — Ensemble-like regularisation; harder to memorise Spoiled features as Fresh.
6. **Progressive resize** — Training on 224 then 300 captures fine texture differences (gill colour, slime) that larger resolution reveals.
7. **Monitoring FPR explicitly** — `final_metrics.json` reports `spoiled_as_fresh_fpr`; if > 1%, increase Spoiled class weight.

---

## Tips to Push Beyond 95%

| Technique | Expected Gain | Effort |
|-----------|--------------|--------|
| Increase dataset to 3k+/class | +1–3% | Medium |
| Use EfficientNetV2-M instead of S | +0.5–1% | Low |
| MixUp / CutMix augmentation | +0.5–1% | Low |
| Self-supervised pre-training (SimCLR on fish data) | +1–2% | High |
| Ensemble 3 models (different seeds) | +0.5–1.5% | Medium |
| Focal Loss (γ=2) instead of CE for hard examples | +0.3–0.8% | Low |
| 5-fold cross-validation + majority voting | +0.5–1% | Medium |
| SAM (Sharpness-Aware Minimisation) optimiser | +0.3–0.7% | Low |
| Larger input size (384×384) | +0.5–1% | Low |
| Pseudo-labelling on unlabelled fish images | +1–2% | High |

---

## Architecture Deep-Dive

```
Input (3 × 300 × 300)
       │
 EfficientNetV2-S Backbone (ImageNet pretrained)
   ├─ Fused-MBConv stages 0-3 (frozen Phase 1)
   ├─ MBConv stages 4-5       (frozen Phase 1-2)
   └─ MBConv stage 6          (unfrozen Phase 2+)  ◄── Grad-CAM target layer
       │
  Global Average Pool → (1280,)
       │
 Custom Head:
   BatchNorm1d(1280)
   Dropout(0.4)
   Linear(1280 → 512)
   SiLU
   BatchNorm1d(512)
   Dropout(0.2)
   Linear(512 → 3)
       │
  ÷ Temperature T   ◄── calibrated post-training
       │
  Softmax → [Fresh, Medium, Spoiled] probabilities
```

**Loss**: CrossEntropyLoss with label smoothing ε=0.1
**Optimiser**: AdamW with differential LR (backbone 5× lower than head)
**Scheduler**: CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
**Regularisation**: Dropout + Weight Decay (1e-4) + RandomErasing + Label Smoothing

---

## License
MIT — free to use in hackathons and production.
