"""
gradcam.py - Gradient-weighted Class Activation Mapping for Fish Freshness
==========================================================================
Implements Grad-CAM and Grad-CAM++ to produce saliency heatmaps that
highlight *why* the model made its decision (eyes, gills, skin texture).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ─── Hook-based Grad-CAM ────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation using forward/backward hooks.

    Usage:
        gcam = GradCAM(model, target_layer=model.backbone.features[-1])
        heatmap = gcam.generate(tensor_input, target_class=None)
        overlay = gcam.overlay_on_image(pil_image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor : (1, C, H, W) preprocessed tensor
            target_class : class index to explain; None → argmax(logits)

        Returns:
            heatmap (np.ndarray, H×W, float32, range [0,1])
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)            # (1, num_classes)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward w.r.t. target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Pool gradients over spatial dims → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)                                      # (1, 1, h, w)

        # Upsample to input size
        h, w    = input_tensor.shape[2:]
        cam     = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    # ── Overlay helpers ───────────────────────────────────────────────────────
    @staticmethod
    def heatmap_to_rgb(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Convert float[0,1] heatmap to BGR uint8 using OpenCV colormap."""
        heatmap_uint8 = np.uint8(255 * cam)
        heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, colormap)
        return heatmap_bgr  # BGR, uint8

    @staticmethod
    def overlay_on_image(pil_image: Image.Image, cam: np.ndarray,
                         alpha: float = 0.45,
                         colormap: int = cv2.COLORMAP_JET) -> Image.Image:
        """
        Blend Grad-CAM heatmap with the original image.

        Returns:
            PIL.Image (RGB)
        """
        img_np = np.array(pil_image.convert("RGB"))          # H×W×3, uint8
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Resize cam to match image
        cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)

        blended_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended_rgb)

    @staticmethod
    def save_heatmap_grid(pil_image: Image.Image, cam: np.ndarray,
                          save_path: str, pred_class: str,
                          confidence: float, freshness_score: float):
        """
        Save a 3-panel figure: Original | Heatmap | Overlay with annotations.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0f1117")

        titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
        colors = {"Fresh": "#22c55e", "Medium": "#f59e0b", "Spoiled": "#ef4444"}
        accent = colors.get(pred_class, "#6366f1")

        # Panel 1: Original
        axes[0].imshow(pil_image)
        axes[0].set_title(titles[0], color="white", fontsize=13, pad=10)
        axes[0].axis("off")

        # Panel 2: Heatmap only
        cam_resized = cv2.resize(cam, pil_image.size)
        axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(titles[1], color="white", fontsize=13, pad=10)
        axes[1].axis("off")

        # Panel 3: Overlay
        overlay = GradCAM.overlay_on_image(pil_image, cam_resized)
        axes[2].imshow(overlay)
        axes[2].set_title(titles[2], color="white", fontsize=13, pad=10)
        axes[2].axis("off")

        # Annotation bar
        fig.text(
            0.5, 0.02,
            f"Prediction: {pred_class}  |  Confidence: {confidence:.1f}%  |  "
            f"Freshness Score: {freshness_score:.0f}/100",
            ha="center", fontsize=13, color=accent,
            fontweight="bold",
        )

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(accent)
                spine.set_linewidth(1.5)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Grad-CAM grid saved → {save_path}")


# ─── Grad-CAM++ ──────────────────────────────────────────────────────────────
class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ (Chattopadhyay et al., 2018) — improved localisation for
    multiple object instances and finer details (e.g., fish gills/eyes).
    """

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        grads = self.gradients          # (1, C, h, w)
        acts  = self.activations        # (1, C, h, w)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        sum_acts  = acts.sum(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        eps       = 1e-7

        alpha_num   = grads_sq
        alpha_denom = 2 * grads_sq + sum_acts * grads_cu + eps
        alpha       = alpha_num / alpha_denom

        weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        h, w = input_tensor.shape[2:]
        cam  = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam  = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)


# ─── Factory Helper ──────────────────────────────────────────────────────────
def build_gradcam(model: torch.nn.Module,
                  method: str = "gradcam++") -> GradCAM:
    """
    Build Grad-CAM (or Grad-CAM++) attached to the last conv block of
    EfficientNetV2-S (features[-1]).
    """
    target_layer = model.backbone.features[-1]  # Last MBConv/FusedMBConv block
    if method == "gradcam++":
        return GradCAMPlusPlus(model, target_layer)
    return GradCAM(model, target_layer)


# ─── Batch visualisation utility ─────────────────────────────────────────────
def visualise_batch(model, dataloader, device, save_dir: str = "outputs/gradcam",
                    n_samples: int = 12, method: str = "gradcam++"):
    """
    Generate and save Grad-CAM overlays for the first n_samples from a loader.
    Useful for qualitative evaluation of what the model attends to.
    """
    from model import get_transforms, CLASS_NAMES
    import torchvision.transforms.functional as TF

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gcam     = build_gradcam(model, method=method)
    inv_mean = torch.tensor([0.485, 0.456, 0.406])
    inv_std  = torch.tensor([0.229, 0.224, 0.225])

    count = 0
    for imgs, labels in dataloader:
        for i in range(imgs.size(0)):
            if count >= n_samples:
                return
            tensor = imgs[i:i+1].to(device)

            # Reconstruct PIL for overlay
            img_t  = imgs[i].cpu() * inv_std[:, None, None] + inv_mean[:, None, None]
            pil_img = TF.to_pil_image(img_t.clamp(0, 1))

            cam = gcam.generate(tensor, target_class=None)
            pred_class = CLASS_NAMES[model(tensor).argmax(dim=1).item()]
            true_class = CLASS_NAMES[labels[i].item()]

            out_path = save_dir / f"sample_{count:03d}_{pred_class}_true{true_class}.png"
            GradCAM.save_heatmap_grid(pil_img, cam, str(out_path),
                                      pred_class, confidence=0.0, freshness_score=0.0)
            count += 1


if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model import FishFreshnessModel, get_transforms

    model = FishFreshnessModel(pretrained=False)
    model.eval()

    gcam   = build_gradcam(model, "gradcam++")
    dummy  = torch.randn(1, 3, 300, 300)
    hm     = gcam.generate(dummy)
    print(f"Heatmap shape: {hm.shape}, min={hm.min():.3f}, max={hm.max():.3f}")
    print("gradcam.py OK ✓")
