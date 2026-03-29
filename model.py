"""
model.py - EfficientNetV2 Model for Fish Freshness Classification
================================================================
Handles model architecture, loading, inference, ONNX export,
temperature scaling calibration, and Test-Time Augmentation (TTA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np
from pathlib import Path
import json
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Fresh", "Medium", "Spoiled"]
NUM_CLASSES  = len(CLASS_NAMES)
IMG_SIZE_S1  = 224   # Phase-1 training (smaller)
IMG_SIZE_S2  = 300   # Phase-2 progressive resize

# Freshness score mapping: Fresh=100, Medium=50, Spoiled=0 (blended by confidence)
FRESHNESS_ANCHORS = {"Fresh": 100.0, "Medium": 50.0, "Spoiled": 0.0}


# ─── Model Architecture ──────────────────────────────────────────────────────
class FishFreshnessModel(nn.Module):
    """
    EfficientNetV2-S backbone with a custom classification head.

    Design choices:
      • Dropout (p=0.4) before the final linear layer → regularisation
      • BatchNorm in head for stable fine-tuning
      • Label-smoothing loss is applied externally during training
      • Supports staged unfreezing: freeze backbone → unfreeze last N blocks
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)

        # Remove original classifier; keep feature extractor
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(512, num_classes),
        )

        # Temperature for calibration (post-hoc)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits   = self.head(features)
        return logits / self.temperature

    # ── Staged Unfreezing ────────────────────────────────────────────────────
    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase-1: head-only training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen. Training head only.")

    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Unfreeze the last *n* blocks of EfficientNetV2 for fine-tuning.
        EfficientNetV2-S has 7 fused/MBConv stages (indices 0-6).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = list(self.backbone.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze last {n} blocks. Trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze the entire network for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Full model unfrozen. Trainable params: {trainable:,}")


# ─── Temperature Scaling Calibration ────────────────────────────────────────
class TemperatureScaler:
    """
    Post-hoc confidence calibration via temperature scaling.
    Learns a single scalar T on the validation set so that
    softmax(logits / T) is better calibrated (ECE minimised).
    """

    def __init__(self, model: FishFreshnessModel, device: torch.device):
        self.model  = model
        self.device = device

    def calibrate(self, val_loader, max_iter: int = 50):
        self.model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(self.device)
                # Raw logits before temperature division
                feats  = self.model.backbone(imgs)
                logits = self.model.head(feats)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        temperature = nn.Parameter(torch.ones(1))
        optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        nll_loss    = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = nll_loss(logits_all / temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        optimal_T = temperature.item()
        self.model.temperature.data = torch.tensor([optimal_T])
        logger.info(f"Calibrated temperature: {optimal_T:.4f}")
        return optimal_T


# ─── Transforms ──────────────────────────────────────────────────────────────
def get_transforms(phase: str = "val", img_size: int = IMG_SIZE_S1):
    """
    Return torchvision transform pipelines.

    phase = 'train'  → heavy augmentation
    phase = 'val'    → deterministic preprocessing
    phase = 'tta'    → one TTA variant (used multiple times externally)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout-style
        ])

    if phase == "tta":
        return T.Compose([
            T.Resize(int(img_size * 1.1)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # val / test
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ─── Inference Engine ────────────────────────────────────────────────────────
class FishInferenceEngine:
    """
    High-level inference wrapper supporting:
      • PyTorch native inference
      • ONNX Runtime inference
      • Test-Time Augmentation (TTA)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto",
                 use_onnx: bool = False, onnx_path: str = None):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_onnx = use_onnx and (onnx_path is not None)
        self.class_names = CLASS_NAMES

        if self.use_onnx:
            logger.info(f"Loading ONNX model from {onnx_path}")
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if torch.cuda.is_available() else ["CPUExecutionProvider"])
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        else:
            logger.info(f"Loading PyTorch model from {checkpoint_path}")
            self.model = FishFreshnessModel(pretrained=False)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

        self.val_transform = get_transforms("val", IMG_SIZE_S2)
        self.tta_transform = get_transforms("tta", IMG_SIZE_S2)

    # ── Core predict ─────────────────────────────────────────────────────────
    def predict(self, pil_image, use_tta: bool = True, tta_n: int = 8):
        """
        Run inference on a PIL image.
        Returns dict with class, confidence, freshness_score, all_probs.
        """
        if self.use_onnx:
            return self._predict_onnx(pil_image)

        with torch.no_grad():
            if use_tta:
                probs = self._tta_probs(pil_image, tta_n)
            else:
                tensor = self.val_transform(pil_image).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx       = int(np.argmax(probs))
        pred_class     = self.class_names[pred_idx]
        confidence     = float(probs[pred_idx]) * 100.0
        freshness_score = self._freshness_score(probs)

        return {
            "class":           pred_class,
            "confidence":      round(confidence, 2),
            "freshness_score": round(freshness_score, 1),
            "all_probs": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(self.class_names, probs)
            },
        }

    # ── TTA ──────────────────────────────────────────────────────────────────
    def _tta_probs(self, pil_image, n: int = 8) -> np.ndarray:
        """Average predictions over n augmented views + 1 canonical view."""
        all_probs = []

        # Canonical
        tensor = self.val_transform(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        all_probs.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())

        # TTA augmented
        for _ in range(n):
            tensor = self.tta_transform(pil_image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            all_probs.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())

        return np.mean(all_probs, axis=0)

    # ── ONNX ─────────────────────────────────────────────────────────────────
    def _predict_onnx(self, pil_image) -> dict:
        tensor = self.val_transform(pil_image).unsqueeze(0).numpy()
        ort_inputs  = {self.ort_session.get_inputs()[0].name: tensor}
        logits_np   = self.ort_session.run(None, ort_inputs)[0]
        logits_t    = torch.from_numpy(logits_np)
        probs       = F.softmax(logits_t, dim=1).squeeze(0).numpy()

        pred_idx       = int(np.argmax(probs))
        pred_class     = self.class_names[pred_idx]
        confidence     = float(probs[pred_idx]) * 100.0
        freshness_score = self._freshness_score(probs)

        return {
            "class":           pred_class,
            "confidence":      round(confidence, 2),
            "freshness_score": round(freshness_score, 1),
            "all_probs": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(self.class_names, probs)
            },
        }

    # ── Freshness score ───────────────────────────────────────────────────────
    @staticmethod
    def _freshness_score(probs: np.ndarray) -> float:
        """
        Weighted blend of class anchor scores by predicted probability.
        Fresh=100, Medium=50, Spoiled=0
        """
        anchors = [100.0, 50.0, 0.0]
        return float(np.dot(probs, anchors))


# ─── ONNX Export ─────────────────────────────────────────────────────────────
def export_to_onnx(checkpoint_path: str, onnx_path: str,
                   img_size: int = IMG_SIZE_S2, quantize: bool = False):
    """
    Export trained PyTorch model to ONNX (opset 17).
    Optionally applies dynamic INT8 quantisation via ONNX Runtime tools.
    """
    device = torch.device("cpu")
    model  = FishFreshnessModel(pretrained=False)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    state  = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info(f"ONNX model saved to {onnx_path}  ({Path(onnx_path).stat().st_size/1e6:.1f} MB)")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
            logger.info(f"INT8 quantised model saved to {q_path}  ({Path(q_path).stat().st_size/1e6:.1f} MB)")
        except ImportError:
            logger.warning("onnxruntime-tools not installed; skipping quantisation.")

    return onnx_path


if __name__ == "__main__":
    # Quick smoke-test
    model = FishFreshnessModel(pretrained=True)
    model.freeze_backbone()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # (2, 3)
    print("model.py OK ✓")
