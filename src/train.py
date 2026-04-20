"""
2-Phase Fine-tuning training script (theo paper EfficientNetV2S Face Shape Classification)

Phase 1: Freeze toàn bộ backbone, chỉ train classifier head (lr cao, ổn định weights mới)
Phase 2: Unfreeze N top blocks của backbone, fine-tune với lr rất nhỏ

Usage:
    cd src
    uv run python train.py

    # Dùng data đã preprocess (recommended)
    uv run python train.py --data_dir "../data/face-shape-cropped"

    # Tuỳ chỉnh
    uv run python train.py \\
        --data_dir "../data/face-shape-cropped" \\
        --model efficientnet_v2_s \\
        --phase1_epochs 10 \\
        --phase2_epochs 20 \\
        --phase1_lr 1e-3 \\
        --phase2_lr 1e-5 \\
        --unfreeze_blocks 2 \\
        --batch_size 32
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse

from dataset import get_dataloader
from model import FaceShapeModel


class EarlyStopping:
    """Dừng training sớm khi val_loss không cải thiện sau `patience` epochs."""
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for inputs, labels in tqdm(loader, desc="Train" if is_train else "Val  ", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def print_header(phase_name):
    print(f"\n{'='*65}")
    print(f"  {phase_name}")
    print(f"{'='*65}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print(f"{'-'*65}")


def train_phase(phase_name, model, train_loader, val_loader, criterion,
                optimizer, scheduler, early_stopping, epochs, device, best_val_acc, best_model_path):
    print_header(phase_name)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>8.2e}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"         → Saved best  val_acc={best_val_acc:.4f}")

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    return best_val_acc


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dir = os.path.join(args.data_dir, "training_set")
    test_dir  = os.path.join(args.data_dir, "testing_set")

    if not os.path.exists(train_dir):
        print(f"[ERROR] {train_dir} not found. Run preprocess_dataset.py first, or check --data_dir")
        return

    train_loader, classes = get_dataloader(train_dir, batch_size=args.batch_size, is_train=True)
    val_loader, _         = get_dataloader(test_dir,  batch_size=args.batch_size, is_train=False)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = FaceShapeModel(
        num_classes=num_classes,
        model_name=args.model,
        dropout=args.dropout
    ).to(device)

    # Label smoothing — tránh model quá tự tin
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs("../outputs", exist_ok=True)
    best_model_path = f"../outputs/best_{args.model}.pth"
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=args.patience)

    # ─────────────────────────────────────────────
    # PHASE 1: Train chỉ classifier head
    # Backbone đang frozen (từ model.__init__)
    # ─────────────────────────────────────────────
    print("\n[Phase 1] Training classifier head (backbone frozen)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=1e-4
    )
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.phase1_epochs, eta_min=1e-6)

    best_val_acc = train_phase(
        "Phase 1 — Head only", model, train_loader, val_loader,
        criterion, optimizer_p1, scheduler_p1, early_stopping,
        args.phase1_epochs, device, best_val_acc, best_model_path
    )

    # ─────────────────────────────────────────────
    # PHASE 2: Unfreeze N top blocks + Fine-tune
    # ─────────────────────────────────────────────
    print(f"\n[Phase 2] Unfreezing top {args.unfreeze_blocks} blocks of backbone")
    model.unfreeze_top_layers(num_blocks=args.unfreeze_blocks)

    # Dùng lr rất nhỏ để tránh catastrophic forgetting
    optimizer_p2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr, weight_decay=1e-4
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=args.phase2_epochs, eta_min=1e-7)

    early_stopping.reset()
    best_val_acc = train_phase(
        f"Phase 2 — Fine-tune (top {args.unfreeze_blocks} blocks)", model, train_loader, val_loader,
        criterion, optimizer_p2, scheduler_p2, early_stopping,
        args.phase2_epochs, device, best_val_acc, best_model_path
    )

    print(f"\n{'='*65}")
    print(f"Training complete! Best val_acc: {best_val_acc:.4f}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Phase fine-tuning for face shape classification")
    parser.add_argument("--data_dir", type=str,
                        default="../data/face-shape-cropped",
                        help="Path to pre-cropped dataset (run preprocess_dataset.py first)")
    parser.add_argument("--model", type=str, default="efficientnet_v2_s",
                        choices=["efficientnet_v2_s", "efficientnet_b0", "mobilenet_v3_small"])
    parser.add_argument("--phase1_epochs", type=int, default=10,
                        help="Epochs for Phase 1 (head only)")
    parser.add_argument("--phase2_epochs", type=int, default=40,
                        help="Epochs for Phase 2 (fine-tune top blocks)")
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-4,
                        help="LR for fine-tuning unfrozen blocks (higher than 1e-5 to converge faster)")
    parser.add_argument("--unfreeze_blocks", type=int, default=3,
                        help="Number of top backbone blocks to unfreeze in Phase 2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train(args)
