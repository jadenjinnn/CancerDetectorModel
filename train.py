"""
Standard PyTorch training loop with CUDA mixed precision (AMP) and Weights & Biases.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
    log_interval: int,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                logits = model(data)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = data.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += bs

        if batch_idx % log_interval == 0:
            global_step = epoch * num_batches + batch_idx
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/epoch": epoch,
                    "train/batch": global_step,
                },
                step=global_step,
            )

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                logits = model(data)
                loss = criterion(logits, targets)
        else:
            logits = model(data)
            loss = criterion(logits, targets)

        bs = data.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += bs

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with AMP + W&B")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--wandb-project", type=str, default="CancerDetectorModel")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (CUDA only)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Dry run without logging to W&B")
    p.add_argument("--save-freq", type=int, default=3,
                   help="Save checkpoint every N epochs")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                   help="Directory to save checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconf):
        try:
            _reconf(line_buffering=True)
        except OSError:
            pass

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} epochs={args.epochs} batch_size={args.batch_size} "
        f"data_dir={args.data_dir}",
        flush=True,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler: GradScaler | None = GradScaler() if use_amp else None

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    print("wandb.init done", flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.PCAM(args.data_dir, split="train",
                             download=True, transform=transform)
    val_ds = datasets.PCAM(
        args.data_dir, split="val", download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    wandb.watch(model, log="gradients", log_freq=500)

    num_batches = len(train_loader)
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.log_interval,
            use_amp,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        epoch_end_step = (epoch + 1) * num_batches - 1
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/acc": train_metrics["acc"],
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch_end_step,
        )

        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

    wandb.finish()


if __name__ == "__main__":
    main()
