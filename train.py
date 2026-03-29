"""
train.py - Full Training Pipeline for Fish Freshness Classification
===================================================================
Features:
  • EfficientNetV2-S with staged unfreezing (head-only → partial → full)
  • Progressive resizing (224 → 300)
  • Advanced augmentation (torchvision + albumentation-style via torch)
  • Weighted random sampler for class imbalance
  • CosineAnnealingWarmRestarts LR scheduler
  • Early stopping with best-model checkpoint
  • Mixed-precision (AMP) training
  • Label smoothing loss
  • Stratified train/val split
  • Comprehensive metrics: accuracy, precision, recall, F1, confusion matrix
  • Temperature calibration post-training
  • Grad-CAM qualitative evaluation
  • ONNX export
"""

import os, sys, json, time, logging, random
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.datasets as datasets

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
)

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from model import (
    FishFreshnessModel, FishInferenceEngine,
    get_transforms, TemperatureScaler,
    export_to_onnx, CLASS_NAMES,
    IMG_SIZE_S1, IMG_SIZE_S2,
)
from gradcam import build_gradcam, visualise_batch

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    # Data
    "data_dir":          "data/fish_dataset",
    "val_split":         0.15,
    "num_workers":       4,
    "seed":              42,

    # Model
    "num_classes":       3,
    "dropout":           0.4,
    "label_smoothing":   0.1,

    # Phase-1: train head only (frozen backbone, small LR)
    "phase1_epochs":     10,
    "phase1_lr":         3e-3,
    "phase1_batch":      64,
    "phase1_img_size":   IMG_SIZE_S1,

    # Phase-2: unfreeze last 3 blocks + progressive resize
    "phase2_epochs":     20,
    "phase2_lr":         5e-4,
    "phase2_batch":      32,
    "phase2_img_size":   IMG_SIZE_S2,
    "phase2_unfreeze_n": 3,

    # Phase-3: full fine-tune (optional, lower LR)
    "phase3_epochs":     10,
    "phase3_lr":         1e-4,

    # Scheduler
    "t0":                10,    # CosineAnnealingWarmRestarts period
    "t_mult":            2,
    "eta_min":           1e-6,

    # Early stopping
    "patience":          8,
    "min_delta":         1e-4,

    # Paths
    "checkpoint_dir":    "checkpoints",
    "best_ckpt":         "checkpoints/best_model.pth",
    "onnx_path":         "checkpoints/fish_freshness.onnx",
    "report_dir":        "outputs/reports",
    "gradcam_dir":       "outputs/gradcam",
}


# ─── Reproducibility ─────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Dataset Utilities ───────────────────────────────────────────────────────
class SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a Subset and applies a per-phase transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders(data_dir: str, img_size: int, batch_size: int,
                      val_split: float = 0.15, num_workers: int = 4,
                      seed: int = 42):
    """
    Build stratified train/val DataLoaders with:
      • Weighted random sampling to handle class imbalance
      • Per-split transforms
    """
    # Load with no transform first (PIL images)
    full_dataset = datasets.ImageFolder(root=data_dir)
    targets      = np.array(full_dataset.targets)
    n            = len(targets)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(n), targets))

    # Class weights for WeightedRandomSampler
    train_labels  = targets[train_idx]
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_idx),
        replacement=True,
    )

    # Transforms
    train_tf = get_transforms("train", img_size)
    val_tf   = get_transforms("val",   img_size)

    train_ds = SubsetWithTransform(full_dataset, train_idx, train_tf)
    val_ds   = SubsetWithTransform(full_dataset, val_idx,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    # Class distribution log
    dist = {CLASS_NAMES[i]: int(c) for i, c in enumerate(class_counts)}
    logger.info(f"Dataset split → train: {len(train_idx)}  val: {len(val_idx)}")
    logger.info(f"Class distribution (train): {dist}")

    return train_loader, val_loader, full_dataset.class_to_idx


# ─── Metrics Utilities ───────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion):
    """Run one evaluation epoch; return loss, accuracy, all preds & labels."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(labels, preds, class_names, save_path: str, title: str = ""):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", color="white", fontsize=12, labelpad=8)
    ax.set_ylabel("True",      color="white", fontsize=12, labelpad=8)
    ax.set_title(title or "Confusion Matrix", color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")

    # Normalised row (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.72, f"{cm_norm[i, j]:.1%}",
                    ha="center", va="center",
                    fontsize=9, color="#94a3b8")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def save_training_curves(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color="#6366f1", linewidth=2, label="Train")
    ax1.plot(epochs, history["val_loss"],   color="#22c55e", linewidth=2, label="Val")
    ax1.set_title("Loss",     color="white", fontsize=13)
    ax1.set_xlabel("Epoch",   color="white")
    ax1.set_ylabel("Loss",    color="white")
    ax1.legend(facecolor="#1e2130", labelcolor="white")

    ax2.plot(epochs, history["train_acc"], color="#6366f1", linewidth=2, label="Train")
    ax2.plot(epochs, history["val_acc"],   color="#22c55e", linewidth=2, label="Val")
    ax2.axhline(0.95, color="#f59e0b", linestyle="--", linewidth=1, label="95% target")
    ax2.set_title("Accuracy", color="white", fontsize=13)
    ax2.set_xlabel("Epoch",   color="white")
    ax2.set_ylabel("Accuracy", color="white")
    ax2.legend(facecolor="#1e2130", labelcolor="white")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ─── Early Stopping ──────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─── One Training Phase ──────────────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, criterion, optimizer,
              scheduler, scaler, device, epochs, phase_name,
              checkpoint_dir, history, patience, min_delta):
    """Generic training loop for one phase. Returns best_val_acc."""

    best_val_acc  = 0.0
    best_state    = None
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    ckpt_dir      = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        t_preds, t_labels = [], []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            t_loss  += loss.item() * imgs.size(0)
            t_preds.extend(logits.argmax(dim=1).cpu().numpy())
            t_labels.extend(labels.cpu().numpy())

        scheduler.step()

        train_loss = t_loss / len(train_loader.dataset)
        train_acc  = accuracy_score(t_labels, t_preds)

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[{phase_name}] Ep {epoch:02d}/{epochs} │ "
            f"TrLoss {train_loss:.4f} TrAcc {train_acc:.4f} │ "
            f"VaLoss {val_loss:.4f} VaAcc {val_acc:.4f} │ "
            f"LR {lr_now:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = deepcopy(model.state_dict())
            torch.save(
                {"model_state_dict": best_state, "val_acc": val_acc,
                 "epoch": epoch, "phase": phase_name},
                ckpt_dir / "best_model.pth",
            )
            logger.info(f"  ★ New best val_acc = {val_acc:.4f} — checkpoint saved")

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                ckpt_dir / f"{phase_name}_epoch{epoch:02d}.pth",
            )

        if early_stopper(val_acc):
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc


# ─── Full Training Pipeline ──────────────────────────────────────────────────
def train(cfg: dict = CFG):
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device}  |  AMP: {use_amp}")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["report_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = FishFreshnessModel(
        num_classes=cfg["num_classes"],
        dropout_p=cfg["dropout"],
        pretrained=True,
    ).to(device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler    = GradScaler() if use_amp else None
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── PHASE 1: Train head only at img_size=224 ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Head-only training (backbone frozen)")
    logger.info("=" * 60)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        cfg["data_dir"], cfg["phase1_img_size"],
        cfg["phase1_batch"], cfg["val_split"],
        cfg["num_workers"], cfg["seed"],
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase1_lr"], weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["t0"], T_mult=cfg["t_mult"], eta_min=cfg["eta_min"])

    run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              scaler, device, cfg["phase1_epochs"], "phase1",
              cfg["checkpoint_dir"], history,
              patience=cfg["patience"], min_delta=cfg["min_delta"])

    # ── PHASE 2: Unfreeze last N blocks + progressive resize ──────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Unfreeze last {cfg['phase2_unfreeze_n']} blocks + resize to {cfg['phase2_img_size']}")
    logger.info("=" * 60)

    model.unfreeze_last_n_blocks(cfg["phase2_unfreeze_n"])

    train_loader, val_loader, _ = build_dataloaders(
        cfg["data_dir"], cfg["phase2_img_size"],
        cfg["phase2_batch"], cfg["val_split"],
        cfg["num_workers"], cfg["seed"],
    )

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["phase2_lr"] / 5},
        {"params": model.head.parameters(),     "lr": cfg["phase2_lr"]},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["t0"], T_mult=cfg["t_mult"], eta_min=cfg["eta_min"])

    run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              scaler, device, cfg["phase2_epochs"], "phase2",
              cfg["checkpoint_dir"], history,
              patience=cfg["patience"], min_delta=cfg["min_delta"])

    # ── PHASE 3: Full fine-tune ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — Full model fine-tuning")
    logger.info("=" * 60)

    model.unfreeze_all()

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["phase3_lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=cfg["eta_min"])

    best_acc = run_phase(model, train_loader, val_loader, criterion, optimizer,
                         scheduler, scaler, device, cfg["phase3_epochs"],
                         "phase3", cfg["checkpoint_dir"], history,
                         patience=cfg["patience"], min_delta=cfg["min_delta"])

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_ckpt = torch.load(cfg["best_ckpt"], map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info(f"Loaded best checkpoint (val_acc={best_ckpt['val_acc']:.4f})")

    # ── Temperature calibration ───────────────────────────────────────────────
    logger.info("Running temperature scaling calibration …")
    scaler_cal = TemperatureScaler(model, device)
    scaler_cal.calibrate(val_loader)

    # Re-save with calibrated temperature
    torch.save({"model_state_dict": model.state_dict(),
                "val_acc": best_ckpt["val_acc"],
                "temperature": model.temperature.item(),
                "class_to_idx": class_to_idx},
               cfg["best_ckpt"])

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    _, final_acc, preds, labels = evaluate(model, val_loader, device, criterion)

    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2])
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro")

    report = classification_report(labels, preds, target_names=CLASS_NAMES)
    logger.info("\n" + report)

    # Key metric: false-positive rate for "Spoiled classified as Fresh"
    cm = confusion_matrix(labels, preds)
    # Row=Spoiled(2), Col=Fresh(0)
    spoiled_as_fresh = cm[2, 0] if cm.shape[0] > 2 else 0
    total_spoiled    = cm[2].sum() if cm.shape[0] > 2 else 1
    false_positive_rate = spoiled_as_fresh / max(total_spoiled, 1)
    logger.info(f"⚠ Spoiled→Fresh False Positive Rate: {false_positive_rate:.2%}")

    # Save results
    results = {
        "final_val_accuracy": round(final_acc, 4),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(p[i]), 4),
                "recall":    round(float(r[i]), 4),
                "f1":        round(float(f1[i]), 4),
            } for i in range(len(CLASS_NAMES))
        },
        "spoiled_as_fresh_fpr": round(false_positive_rate, 4),
        "temperature": model.temperature.item(),
    }

    report_path = Path(cfg["report_dir"]) / "final_metrics.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved → {report_path}")

    # Confusion matrix plot
    save_confusion_matrix(
        labels, preds, CLASS_NAMES,
        str(Path(cfg["report_dir"]) / "confusion_matrix.png"),
        title=f"Fish Freshness Confusion Matrix  (Acc={final_acc:.2%})",
    )

    # Training curves
    save_training_curves(history,
                         str(Path(cfg["report_dir"]) / "training_curves.png"))

    # ── Grad-CAM qualitative evaluation ──────────────────────────────────────
    logger.info("Generating Grad-CAM visualisations …")
    try:
        visualise_batch(model, val_loader, device,
                        save_dir=cfg["gradcam_dir"], n_samples=16)
    except Exception as e:
        logger.warning(f"Grad-CAM visualisation failed: {e}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    logger.info("Exporting to ONNX …")
    export_to_onnx(cfg["best_ckpt"], cfg["onnx_path"],
                   img_size=cfg["phase2_img_size"], quantize=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best Val Accuracy : {final_acc:.2%}")
    logger.info(f"  Macro F1          : {f1_macro:.4f}")
    logger.info(f"  Spoiled→Fresh FPR : {false_positive_rate:.2%}")
    logger.info(f"  Temperature (T)   : {model.temperature.item():.4f}")
    logger.info(f"  Checkpoint        : {cfg['best_ckpt']}")
    logger.info(f"  ONNX model        : {cfg['onnx_path']}")
    logger.info("=" * 60)

    if final_acc >= 0.95:
        logger.info("✅  TARGET ≥95% ACHIEVED!")
    else:
        logger.info(f"⚠ Accuracy {final_acc:.2%} — see tips in README to push higher.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fish Freshness Classifier – Training")
    parser.add_argument("--data_dir",  type=str, default=CFG["data_dir"])
    parser.add_argument("--epochs1",   type=int, default=CFG["phase1_epochs"])
    parser.add_argument("--epochs2",   type=int, default=CFG["phase2_epochs"])
    parser.add_argument("--epochs3",   type=int, default=CFG["phase3_epochs"])
    parser.add_argument("--batch1",    type=int, default=CFG["phase1_batch"])
    parser.add_argument("--batch2",    type=int, default=CFG["phase2_batch"])
    parser.add_argument("--seed",      type=int, default=CFG["seed"])
    args = parser.parse_args()

    CFG.update({
        "data_dir":        args.data_dir,
        "phase1_epochs":   args.epochs1,
        "phase2_epochs":   args.epochs2,
        "phase3_epochs":   args.epochs3,
        "phase1_batch":    args.batch1,
        "phase2_batch":    args.batch2,
        "seed":            args.seed,
    })

    train(CFG)
