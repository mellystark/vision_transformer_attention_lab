
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16


# ============================================================
# CONFIG HELPERS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oxford-IIIT Pet üzerinde ViT-B/16 fine-tuning"
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)

    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--finetune-blocks", type=int, default=2)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Son N blok yerine tüm ViT'i fine-tune et.",
    )

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-root",
        default="vit_finetune_outputs",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_data_root():
    project_dir = Path(__file__).resolve().parent

    sibling = (
        project_dir.parent
        / "cnn_architecture_benchmark"
        / "data"
    )

    if sibling.exists():
        return sibling

    return project_dir / "data"


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.75, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )

    return train_tf, eval_tf


def get_labels(dataset):
    labels = getattr(dataset, "_labels", None)

    if labels is None:
        raise RuntimeError(
            "OxfordIIITPet _labels alanı bulunamadı."
        )

    return np.asarray(labels)


def build_loaders(
    data_root,
    batch_size,
    num_workers,
    seed,
):
    train_tf, eval_tf = build_transforms()

    trainval_train = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=True,
    )

    trainval_eval = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=eval_tf,
        download=True,
    )

    test_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        transform=eval_tf,
        download=True,
    )

    labels = get_labels(trainval_train)
    indices = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    train_dataset = Subset(
        trainval_train,
        train_idx,
    )

    val_dataset = Subset(
        trainval_eval,
        val_idx,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )

    classes = list(
        getattr(trainval_train, "classes", [])
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        classes,
    )


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes, device):
    weights = ViT_B_16_Weights.DEFAULT

    model = vit_b_16(
        weights=weights
    )

    in_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        in_features,
        num_classes,
    )

    return model.to(device)


def freeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_head_only(model):
    freeze_all(model)

    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def enable_last_blocks(
    model,
    num_blocks,
    full_finetune=False,
):
    freeze_all(model)

    # Classification head is always trainable.
    for parameter in model.heads.parameters():
        parameter.requires_grad = True

    if full_finetune:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    # ViT-B/16 has 12 encoder blocks.
    blocks = list(
        model.encoder.layers.children()
    )

    num_blocks = max(
        1,
        min(num_blocks, len(blocks)),
    )

    for block in blocks[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Final LayerNorm
    for parameter in model.encoder.ln.parameters():
        parameter.requires_grad = True

    # CLS token and position embedding can adapt too.
    model.class_token.requires_grad = True
    model.encoder.pos_embedding.requires_grad = True


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    targets_all = []
    preds_all = []
    probs_all = []

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = criterion(
                logits,
                targets,
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        targets_all.append(
            targets.cpu().numpy()
        )

        preds_all.append(
            predictions.cpu().numpy()
        )

        probs_all.append(
            probabilities.cpu().numpy()
        )

    targets = np.concatenate(
        targets_all
    )

    predictions = np.concatenate(
        preds_all
    )

    probabilities = np.concatenate(
        probs_all
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    top5 = np.mean(
        [
            target in row
            for target, row in zip(
                targets,
                np.argpartition(
                    probabilities,
                    -5,
                    axis=1,
                )[:, -5:],
            )
        ]
    )

    return {
        "loss": (
            total_loss
            / len(loader.dataset)
        ),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top5_accuracy": float(top5),
        "targets": targets,
        "predictions": predictions,
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accumulation_steps,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, targets) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            scaled_loss = (
                loss
                / accumulation_steps
            )

        scaler.scale(
            scaled_loss
        ).backward()

        if (
            step % accumulation_steps == 0
            or step == len(loader)
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def run_phase(
    *,
    phase_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    accumulation_steps,
    start_epoch,
    best_state,
    best_val_f1,
    patience_counter,
    patience_limit,
):
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    history = []

    for local_epoch in range(
        1,
        epochs + 1,
    ):
        global_epoch = (
            start_epoch
            + local_epoch
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            accumulation_steps,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        lr = optimizer.param_groups[0]["lr"]

        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        improved = (
            val_metrics["macro_f1"]
            > best_val_f1
        )

        if improved:
            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0
        else:
            patience_counter += 1

        print()
        print(
            f"Epoch {global_epoch:02d} | "
            f"{phase_name}"
        )
        print("-" * 76)

        print(
            f"Train Loss       : "
            f"{train_metrics['loss']:.5f}"
        )

        print(
            f"Train Accuracy   : "
            f"{train_metrics['accuracy'] * 100:6.2f} %"
        )

        print(
            f"Val Loss         : "
            f"{val_metrics['loss']:.5f}"
        )

        print(
            f"Val Accuracy     : "
            f"{val_metrics['accuracy'] * 100:6.2f} %"
        )

        print(
            f"Val Macro-F1     : "
            f"{val_metrics['macro_f1']:.5f}"
        )

        print(
            f"Val Top-5 Acc    : "
            f"{val_metrics['top5_accuracy'] * 100:6.2f} %"
        )

        print(
            f"Learning Rate    : "
            f"{lr:.8f}"
        )

        print(
            f"Epoch Time       : "
            f"{epoch_seconds:.2f} s"
        )

        if device.type == "cuda":
            print(
                f"GPU Peak Memory  : "
                f"{peak_memory_gb:.2f} GB"
            )

        if improved:
            print(
                "✓ New best checkpoint "
                "(validation Macro-F1)"
            )

        history.append(
            {
                "epoch": global_epoch,
                "phase": phase_name,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_top5_accuracy": val_metrics["top5_accuracy"],
                "learning_rate": lr,
                "epoch_seconds": epoch_seconds,
                "gpu_peak_memory_gb": peak_memory_gb,
            }
        )

        if scheduler is not None:
            scheduler.step()

        if (
            patience_limit > 0
            and patience_counter >= patience_limit
        ):
            print(
                f"Early stopping: "
                f"{patience_counter} epoch "
                f"iyileşme yok."
            )
            break

    return (
        history,
        best_state,
        best_val_f1,
        patience_counter,
        start_epoch + len(history),
    )


# ============================================================
# BENCHMARK + PLOTS
# ============================================================

@torch.no_grad()
def benchmark_latency(
    model,
    device,
):
    model.eval()

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    for _ in range(10):
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []

    for _ in range(50):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings.append(
            (time.perf_counter() - start)
            * 1000.0
        )

    latency_ms = float(
        np.median(timings)
    )

    return (
        latency_ms,
        1000.0 / latency_ms,
    )


def save_training_plots(
    history_df,
    output_root,
):
    output_root = Path(output_root)

    # Loss
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Train Loss",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="o",
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "ViT-B/16 - Training / Validation Loss"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_root
        / "01_loss_curve.png",
        dpi=180,
    )
    plt.close()

    # Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"] * 100,
        marker="o",
        label="Train Accuracy",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_accuracy"] * 100,
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "ViT-B/16 - Accuracy"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_root
        / "02_accuracy_curve.png",
        dpi=180,
    )
    plt.close()

    # Macro-F1
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro-F1")
    plt.title(
        "ViT-B/16 - Validation Macro-F1"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_root
        / "03_macro_f1_curve.png",
        dpi=180,
    )
    plt.close()


def save_confusion_matrix(
    targets,
    predictions,
    classes,
    output_path,
):
    cm = confusion_matrix(
        targets,
        predictions,
    )

    plt.figure(
        figsize=(15, 13)
    )

    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90,
        fontsize=6,
    )

    plt.yticks(
        range(len(classes)),
        classes,
        fontsize=6,
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )

    plt.title(
        "ViT-B/16 - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()


def save_resnet_comparison(
    vit_metrics,
    output_root,
):
    # Previous experiment result from the user's CNN benchmark.
    rows = [
        {
            "model": "ResNet18",
            "test_accuracy": 0.8926,
            "macro_f1": 0.8905,
            "params_million": 11.20,
            "latency_ms": 3.50,
            "fps": 285.7,
        },
        {
            "model": "ViT-B/16",
            "test_accuracy": (
                vit_metrics["test_accuracy"]
            ),
            "macro_f1": (
                vit_metrics["test_macro_f1"]
            ),
            "params_million": (
                vit_metrics["total_parameters"]
                / 1e6
            ),
            "latency_ms": (
                vit_metrics["latency_ms"]
            ),
            "fps": (
                vit_metrics["fps"]
            ),
        },
    ]

    df = pd.DataFrame(rows)

    df.to_csv(
        Path(output_root)
        / "resnet18_vs_vit_b16.csv",
        index=False,
    )

    # Accuracy
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        df["model"],
        df["test_accuracy"] * 100,
    )
    plt.ylabel("Test Accuracy (%)")
    plt.title(
        "ResNet18 vs ViT-B/16"
    )
    for bar, val in zip(
        bars,
        df["test_accuracy"] * 100,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        Path(output_root)
        / "05_resnet18_vs_vit_accuracy.png",
        dpi=180,
    )
    plt.close()

    # Latency
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        df["model"],
        df["latency_ms"],
    )
    plt.ylabel("Latency (ms / image)")
    plt.title(
        "ResNet18 vs ViT-B/16 - Latency"
    )
    for bar, val in zip(
        bars,
        df["latency_ms"],
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        Path(output_root)
        / "06_resnet18_vs_vit_latency.png",
        dpi=180,
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if (
        args.require_cuda
        and device.type != "cuda"
    ):
        raise RuntimeError(
            "CUDA bulunamadı."
        )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_root = detect_data_root()

    print("=" * 76)
    print("ViT-B/16 - OXFORD-IIIT PET FINE-TUNING")
    print("=" * 76)

    print(f"Device             : {device}")

    if device.type == "cuda":
        print(
            f"GPU                : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Data root          : "
        f"{data_root}"
    )

    print(
        f"Batch size         : "
        f"{args.batch_size}"
    )

    print(
        f"Gradient accumulate: "
        f"{args.accumulation_steps}"
    )

    print(
        f"Effective batch    : "
        f"{args.batch_size * args.accumulation_steps}"
    )

    print()

    (
        train_loader,
        val_loader,
        test_loader,
        classes,
    ) = build_loaders(
        data_root,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    print("DATASET")
    print("-" * 76)

    print(
        f"Classes            : "
        f"{len(classes)}"
    )

    print(
        f"Train              : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation         : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test               : "
        f"{len(test_loader.dataset)}"
    )

    print()

    model = build_model(
        len(classes),
        device,
    )

    total_params, _ = count_parameters(
        model
    )

    criterion = nn.CrossEntropyLoss()

    history = []
    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    current_epoch = 0

    total_training_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # PHASE 1 - HEAD ONLY
    # --------------------------------------------------------
    enable_head_only(model)

    _, trainable = count_parameters(
        model
    )

    print("PHASE 1 - CLASSIFIER WARM-UP")
    print("-" * 76)

    print(
        f"Total parameters    : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Learning rate       : "
        f"{args.warmup_lr}"
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
    )

    (
        rows,
        best_state,
        best_val_f1,
        patience_counter,
        current_epoch,
    ) = run_phase(
        phase_name="head_warmup",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epochs=args.warmup_epochs,
        accumulation_steps=args.accumulation_steps,
        start_epoch=current_epoch,
        best_state=best_state,
        best_val_f1=best_val_f1,
        patience_counter=patience_counter,
        patience_limit=0,
    )

    history.extend(rows)

    # --------------------------------------------------------
    # PHASE 2 - LAST BLOCKS OR FULL
    # --------------------------------------------------------
    enable_last_blocks(
        model,
        args.finetune_blocks,
        full_finetune=args.full_finetune,
    )

    _, trainable = count_parameters(
        model
    )

    print()
    print("PHASE 2 - TRANSFORMER FINE-TUNING")
    print("-" * 76)

    mode_text = (
        "FULL MODEL"
        if args.full_finetune
        else f"LAST {args.finetune_blocks} ENCODER BLOCKS + HEAD"
    )

    print(
        f"Mode                : "
        f"{mode_text}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Learning rate       : "
        f"{args.finetune_lr}"
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                args.finetune_epochs,
            ),
        )
    )

    patience_counter = 0

    (
        rows,
        best_state,
        best_val_f1,
        patience_counter,
        current_epoch,
    ) = run_phase(
        phase_name="transformer_finetune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.finetune_epochs,
        accumulation_steps=args.accumulation_steps,
        start_epoch=current_epoch,
        best_state=best_state,
        best_val_f1=best_val_f1,
        patience_counter=patience_counter,
        patience_limit=args.patience,
    )

    history.extend(rows)

    training_seconds = (
        time.perf_counter()
        - total_training_start
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------
    model.load_state_dict(
        best_state
    )

    checkpoint_path = (
        output_root
        / "best_vit_b16.pth"
    )

    torch.save(
        {
            "model": "ViT-B/16",
            "state_dict": (
                model.state_dict()
            ),
            "classes": classes,
            "best_val_macro_f1": (
                best_val_f1
            ),
            "finetune_blocks": (
                args.finetune_blocks
            ),
            "full_finetune": (
                args.full_finetune
            ),
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------
    print()
    print("FINAL TEST")
    print("-" * 76)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    latency_ms, fps = benchmark_latency(
        model,
        device,
    )

    checkpoint_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    final_metrics = {
        "model": "ViT-B/16",
        "total_parameters": total_params,
        "best_val_macro_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_top5_accuracy": test_metrics["top5_accuracy"],
        "training_seconds": training_seconds,
        "checkpoint_mb": checkpoint_mb,
        "latency_ms": latency_ms,
        "fps": fps,
        "finetune_blocks": args.finetune_blocks,
        "full_finetune": args.full_finetune,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
    }

    print(
        f"Test Loss          : "
        f"{test_metrics['loss']:.5f}"
    )

    print(
        f"Test Accuracy      : "
        f"{test_metrics['accuracy'] * 100:6.2f} %"
    )

    print(
        f"Macro Precision    : "
        f"{test_metrics['macro_precision']:.5f}"
    )

    print(
        f"Macro Recall       : "
        f"{test_metrics['macro_recall']:.5f}"
    )

    print(
        f"Macro F1           : "
        f"{test_metrics['macro_f1']:.5f}"
    )

    print(
        f"Top-5 Accuracy     : "
        f"{test_metrics['top5_accuracy'] * 100:6.2f} %"
    )

    print(
        f"Training Time      : "
        f"{training_seconds / 60:.2f} min"
    )

    print(
        f"Checkpoint Size    : "
        f"{checkpoint_mb:.2f} MB"
    )

    print(
        f"Latency            : "
        f"{latency_ms:.3f} ms / image"
    )

    print(
        f"Estimated FPS      : "
        f"{fps:.2f}"
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_root
        / "history.csv",
        index=False,
    )

    with open(
        output_root
        / "final_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metrics,
            f,
            indent=2,
        )

    save_training_plots(
        history_df,
        output_root,
    )

    save_confusion_matrix(
        test_metrics["targets"],
        test_metrics["predictions"],
        classes,
        output_root
        / "04_confusion_matrix.png",
    )

    save_resnet_comparison(
        final_metrics,
        output_root,
    )

    print()
    print("SAVED OUTPUTS")
    print("-" * 76)

    for path in [
        "best_vit_b16.pth",
        "history.csv",
        "final_metrics.json",
        "01_loss_curve.png",
        "02_accuracy_curve.png",
        "03_macro_f1_curve.png",
        "04_confusion_matrix.png",
        "resnet18_vs_vit_b16.csv",
        "05_resnet18_vs_vit_accuracy.png",
        "06_resnet18_vs_vit_latency.png",
    ]:
        print(
            output_root / path
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
