
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import vit_b_16


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuned ViT-B/16 prediction and patch-importance inspection"
    )

    parser.add_argument(
        "--checkpoint",
        default="vit_finetune_final_outputs/best_vit_b16.pth",
    )

    parser.add_argument(
        "--output-root",
        default="vit_visual_inspection",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--occlusion-index",
        type=int,
        default=0,
        help="Test dataset index for patch occlusion visualization.",
    )

    return parser.parse_args()


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


def eval_transform():
    return transforms.Compose(
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


def display_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )


def denormalize(tensor):
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=tensor.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=tensor.device,
    ).view(3, 1, 1)

    image = tensor * std + mean

    return image.clamp(0, 1)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    classes = checkpoint["classes"]

    model = vit_b_16(
        weights=None
    )

    in_features = (
        model.heads.head.in_features
    )

    model.heads.head = nn.Linear(
        in_features,
        len(classes),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, classes


@torch.no_grad()
def predict(model, image_tensor):
    logits = model(
        image_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    top_prob, top_idx = torch.topk(
        probabilities,
        k=min(
            5,
            probabilities.shape[1],
        ),
        dim=1,
    )

    return (
        probabilities,
        top_prob,
        top_idx,
    )


def evaluate_all(
    model,
    dataset,
    device,
):
    records = []

    with torch.no_grad():
        for index in range(
            len(dataset)
        ):
            image, target = dataset[index]

            image_batch = (
                image
                .unsqueeze(0)
                .to(device)
            )

            probabilities, _, _ = predict(
                model,
                image_batch,
            )

            confidence, pred = (
                probabilities.max(dim=1)
            )

            records.append(
                {
                    "index": index,
                    "target": int(target),
                    "prediction": int(
                        pred.item()
                    ),
                    "confidence": float(
                        confidence.item()
                    ),
                    "correct": bool(
                        pred.item()
                        == target
                    ),
                }
            )

    return records


def save_sample_grid(
    model,
    tensor_dataset,
    raw_dataset,
    classes,
    indices,
    output_path,
    device,
    title,
):
    cols = 3
    rows = math.ceil(
        len(indices) / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 4.6 * rows),
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, index in zip(
        axes,
        indices,
    ):
        image_tensor, target = (
            tensor_dataset[index]
        )

        raw_image, _ = (
            raw_dataset[index]
        )

        image_batch = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        (
            probabilities,
            top_prob,
            top_idx,
        ) = predict(
            model,
            image_batch,
        )

        pred = int(
            top_idx[0, 0].item()
        )

        confidence = float(
            top_prob[0, 0].item()
        )

        ax.imshow(raw_image)
        ax.axis("off")

        status = (
            "CORRECT"
            if pred == target
            else "WRONG"
        )

        ax.set_title(
            f"{status}\n"
            f"True: {classes[target]}\n"
            f"Pred: {classes[pred]} "
            f"({confidence * 100:.1f}%)",
            fontsize=10,
        )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_top5_for_sample(
    model,
    tensor_dataset,
    raw_dataset,
    classes,
    index,
    output_path,
    device,
):
    image_tensor, target = (
        tensor_dataset[index]
    )

    raw_image, _ = (
        raw_dataset[index]
    )

    image_batch = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    (
        _,
        top_prob,
        top_idx,
    ) = predict(
        model,
        image_batch,
    )

    labels = [
        classes[int(i)]
        for i in top_idx[0].cpu()
    ]

    values = (
        top_prob[0]
        .cpu()
        .numpy()
        * 100
    )

    fig = plt.figure(
        figsize=(12, 5)
    )

    ax1 = fig.add_subplot(
        1,
        2,
        1,
    )

    ax1.imshow(raw_image)
    ax1.axis("off")
    ax1.set_title(
        f"True class: {classes[target]}"
    )

    ax2 = fig.add_subplot(
        1,
        2,
        2,
    )

    bars = ax2.barh(
        labels[::-1],
        values[::-1],
    )

    ax2.set_xlabel(
        "Probability (%)"
    )

    ax2.set_title(
        "Top-5 predictions"
    )

    for bar, value in zip(
        bars,
        values[::-1],
    ):
        ax2.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


@torch.no_grad()
def patch_occlusion_map(
    model,
    image_tensor,
    device,
    patch_size=16,
):
    image = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    baseline_logits = model(
        image
    )

    baseline_probs = torch.softmax(
        baseline_logits,
        dim=1,
    )

    baseline_conf, pred_class = (
        baseline_probs.max(dim=1)
    )

    pred_class = int(
        pred_class.item()
    )

    baseline_conf = float(
        baseline_conf.item()
    )

    grid_size = (
        image.shape[-1]
        // patch_size
    )

    importance = torch.zeros(
        grid_size,
        grid_size,
        device=device,
    )

    # Normalized "neutral" pixel roughly corresponds
    # to mean RGB after ImageNet normalization = 0.
    for row in range(
        grid_size
    ):
        for col in range(
            grid_size
        ):
            masked = image.clone()

            y1 = row * patch_size
            y2 = y1 + patch_size

            x1 = col * patch_size
            x2 = x1 + patch_size

            masked[
                :,
                :,
                y1:y2,
                x1:x2,
            ] = 0.0

            logits = model(
                masked
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            masked_prob = float(
                probs[
                    0,
                    pred_class,
                ].item()
            )

            importance[
                row,
                col,
            ] = (
                baseline_conf
                - masked_prob
            )

    return (
        importance.cpu().numpy(),
        pred_class,
        baseline_conf,
    )


def save_occlusion_visualization(
    model,
    tensor_dataset,
    raw_dataset,
    classes,
    index,
    output_path,
    device,
):
    image_tensor, target = (
        tensor_dataset[index]
    )

    raw_image, _ = (
        raw_dataset[index]
    )

    (
        importance,
        pred_class,
        confidence,
    ) = patch_occlusion_map(
        model,
        image_tensor,
        device,
        patch_size=16,
    )

    # Negative values mean masking increased confidence.
    # For "supportive evidence" visualization,
    # keep positive drops only.
    positive_importance = np.maximum(
        importance,
        0,
    )

    if positive_importance.max() > 0:
        normalized = (
            positive_importance
            / positive_importance.max()
        )
    else:
        normalized = positive_importance

    heatmap = torch.tensor(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    heatmap_up = torch.nn.functional.interpolate(
        heatmap,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    raw_array = np.asarray(
        raw_image
    ).astype(np.float32) / 255.0

    fig = plt.figure(
        figsize=(15, 5)
    )

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(raw_image)
    ax1.axis("off")
    ax1.set_title(
        f"Input\nTrue: {classes[target]}"
    )

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    im = ax2.imshow(
        importance
    )

    ax2.set_title(
        "16x16 patch occlusion importance"
    )

    ax2.axis("off")

    fig.colorbar(
        im,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
    )

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(raw_array)

    ax3.imshow(
        heatmap_up,
        alpha=0.48,
    )

    ax3.axis("off")

    ax3.set_title(
        f"Importance overlay\n"
        f"Pred: {classes[pred_class]} "
        f"({confidence * 100:.1f}%)"
    )

    fig.suptitle(
        "ViT-B/16 patch occlusion analysis",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        importance,
        pred_class,
        confidence,
    )


def main():
    args = parse_args()

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: "
            f"{checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "ViT-B/16 - VISUAL MODEL INSPECTION"
    )
    print("=" * 78)

    print(
        f"Device      : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint  : "
        f"{checkpoint_path}"
    )

    model, classes = load_model(
        checkpoint_path,
        device,
    )

    data_root = detect_data_root()

    tensor_dataset = (
        datasets.OxfordIIITPet(
            root=data_root,
            split="test",
            target_types="category",
            transform=eval_transform(),
            download=True,
        )
    )

    raw_dataset = (
        datasets.OxfordIIITPet(
            root=data_root,
            split="test",
            target_types="category",
            transform=display_transform(),
            download=True,
        )
    )

    print(
        f"Test images : "
        f"{len(tensor_dataset)}"
    )

    print(
        f"Classes     : "
        f"{len(classes)}"
    )

    print()
    print(
        "Test set taranıyor..."
    )

    records = evaluate_all(
        model,
        tensor_dataset,
        device,
    )

    correct = [
        r
        for r in records
        if r["correct"]
    ]

    wrong = [
        r
        for r in records
        if not r["correct"]
    ]

    print(
        f"Correct     : "
        f"{len(correct)}"
    )

    print(
        f"Wrong       : "
        f"{len(wrong)}"
    )

    rng = np.random.default_rng(
        args.seed
    )

    mixed_indices = rng.choice(
        len(tensor_dataset),
        size=min(
            args.num_samples,
            len(tensor_dataset),
        ),
        replace=False,
    ).tolist()

    save_sample_grid(
        model,
        tensor_dataset,
        raw_dataset,
        classes,
        mixed_indices,
        output_root
        / "01_random_test_predictions.png",
        device,
        "ViT-B/16 - Random test predictions",
    )

    # Highest-confidence correct predictions.
    correct_sorted = sorted(
        correct,
        key=lambda r: r["confidence"],
        reverse=True,
    )

    correct_indices = [
        r["index"]
        for r in correct_sorted[
            :args.num_samples
        ]
    ]

    save_sample_grid(
        model,
        tensor_dataset,
        raw_dataset,
        classes,
        correct_indices,
        output_root
        / "02_high_confidence_correct.png",
        device,
        "ViT-B/16 - High-confidence correct predictions",
    )

    # Highest-confidence mistakes are especially interesting.
    wrong_sorted = sorted(
        wrong,
        key=lambda r: r["confidence"],
        reverse=True,
    )

    wrong_indices = [
        r["index"]
        for r in wrong_sorted[
            :args.num_samples
        ]
    ]

    if wrong_indices:
        save_sample_grid(
            model,
            tensor_dataset,
            raw_dataset,
            classes,
            wrong_indices,
            output_root
            / "03_high_confidence_errors.png",
            device,
            "ViT-B/16 - High-confidence errors",
        )

    sample_index = int(
        mixed_indices[0]
    )

    save_top5_for_sample(
        model,
        tensor_dataset,
        raw_dataset,
        classes,
        sample_index,
        output_root
        / "04_sample_top5_predictions.png",
        device,
    )

    occlusion_index = args.occlusion_index

    if (
        occlusion_index < 0
        or occlusion_index
        >= len(tensor_dataset)
    ):
        occlusion_index = sample_index

    print()
    print(
        f"Patch occlusion analysis: "
        f"test index {occlusion_index}"
    )

    (
        importance,
        pred_class,
        confidence,
    ) = save_occlusion_visualization(
        model,
        tensor_dataset,
        raw_dataset,
        classes,
        occlusion_index,
        output_root
        / "05_patch_occlusion_importance.png",
        device,
    )

    np.save(
        output_root
        / "patch_importance.npy",
        importance,
    )

    report = [
        "ViT-B/16 VISUAL INSPECTION",
        "=" * 60,
        f"Test images: {len(tensor_dataset)}",
        f"Correct predictions: {len(correct)}",
        f"Wrong predictions: {len(wrong)}",
        "",
        (
            f"Occlusion sample index: "
            f"{occlusion_index}"
        ),
        (
            f"Occlusion predicted class: "
            f"{classes[pred_class]}"
        ),
        (
            f"Baseline confidence: "
            f"{confidence * 100:.2f}%"
        ),
        "",
        "Interpretation:",
        (
            "Positive occlusion values indicate patches whose masking "
            "decreased confidence in the original predicted class."
        ),
        (
            "Higher positive values therefore indicate stronger "
            "support for that prediction under this perturbation test."
        ),
    ]

    (
        output_root
        / "06_visual_inspection_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("SAVED OUTPUTS")
    print("-" * 78)

    for name in [
        "01_random_test_predictions.png",
        "02_high_confidence_correct.png",
        "03_high_confidence_errors.png",
        "04_sample_top5_predictions.png",
        "05_patch_occlusion_importance.png",
        "patch_importance.npy",
        "06_visual_inspection_report.txt",
    ]:
        print(
            output_root / name
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
