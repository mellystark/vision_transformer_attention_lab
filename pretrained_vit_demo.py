
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights, vit_b_16


# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = Path("pretrained_vit_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

project_dir = Path(__file__).resolve().parent
shared_data_root = (
    project_dir.parent
    / "cnn_architecture_benchmark"
    / "data"
)

if shared_data_root.exists():
    DATA_ROOT = shared_data_root
else:
    DATA_ROOT = project_dir / "data"


# ============================================================
# 1) LOAD SAME OXFORD-IIIT PET IMAGE
# ============================================================
dataset = datasets.OxfordIIITPet(
    root=DATA_ROOT,
    split="trainval",
    target_types="category",
    download=True,
)

image_pil, pet_label = dataset[0]


# ============================================================
# 2) LOAD PRETRAINED ViT-B/16
# ============================================================
weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(
    weights=weights
).to(device)

model.eval()

categories = weights.meta["categories"]
preprocess = weights.transforms()


# ============================================================
# 3) PREPROCESS IMAGE
# ============================================================
input_tensor = preprocess(
    image_pil
).unsqueeze(0).to(device)


# ============================================================
# 4) MODEL INFO
# ============================================================
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


# ============================================================
# 5) INFERENCE
# ============================================================
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

if device.type == "cuda":
    torch.cuda.synchronize()

timings = []

with torch.no_grad():
    for _ in range(50):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings.append(
            (time.perf_counter() - start) * 1000.0
        )

probabilities = torch.softmax(
    logits,
    dim=1,
)

top5_prob, top5_idx = torch.topk(
    probabilities,
    k=5,
    dim=1,
)

median_latency_ms = float(
    torch.tensor(timings).median().item()
)

fps = 1000.0 / median_latency_ms

peak_memory_gb = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 3)
    if device.type == "cuda"
    else 0.0
)


# ============================================================
# 6) TERMINAL REPORT
# ============================================================
print("=" * 78)
print("PRETRAINED VISION TRANSFORMER DEMO")
print("=" * 78)
print(f"Device              : {device}")

if device.type == "cuda":
    print(
        f"GPU                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Model               : ViT-B/16")
print(f"Weights             : ImageNet pretrained")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Input tensor shape  : {tuple(input_tensor.shape)}")
print(f"Oxford pet label    : {pet_label}")
print()

print("TOP-5 IMAGENET PREDICTIONS")
print("-" * 78)

for rank in range(5):
    idx = int(
        top5_idx[0, rank].item()
    )
    prob = float(
        top5_prob[0, rank].item()
    )

    print(
        f"{rank + 1}. "
        f"{categories[idx]:<30} "
        f"{prob * 100:6.2f}%"
    )

print()
print("ENGINEERING METRICS")
print("-" * 78)
print(
    f"Median latency      : "
    f"{median_latency_ms:.3f} ms / image"
)
print(
    f"Estimated FPS       : "
    f"{fps:.2f}"
)

if device.type == "cuda":
    print(
        f"Peak GPU memory     : "
        f"{peak_memory_gb:.2f} GB"
    )

print("=" * 78)


# ============================================================
# 7) SAVE ORIGINAL IMAGE
# ============================================================
plt.figure(
    figsize=(6, 6)
)
plt.imshow(image_pil)
plt.axis("off")
plt.title(
    "Input image - Oxford-IIIT Pet"
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR
    / "01_input_image.png",
    dpi=180,
)
plt.close()


# ============================================================
# 8) SAVE TOP-5 PREDICTION BAR CHART
# ============================================================
labels = [
    categories[int(i)]
    for i in top5_idx[0].detach().cpu()
]

values = (
    top5_prob[0]
    .detach()
    .cpu()
    .numpy()
    * 100.0
)

plt.figure(
    figsize=(9, 5)
)

bars = plt.barh(
    labels[::-1],
    values[::-1],
)

plt.xlabel("Probability (%)")
plt.title(
    "ViT-B/16 - Top-5 ImageNet predictions"
)

for bar, value in zip(
    bars,
    values[::-1],
):
    plt.text(
        value,
        bar.get_y()
        + bar.get_height() / 2,
        f" {value:.2f}%",
        va="center",
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR
    / "02_top5_predictions.png",
    dpi=180,
)
plt.close()


# ============================================================
# 9) SAVE TEXT REPORT
# ============================================================
report_lines = [
    "PRETRAINED VISION TRANSFORMER DEMO",
    "=" * 60,
    f"Model: ViT-B/16",
    f"Device: {device}",
    f"Parameters: {total_params:,}",
    f"Input shape: {tuple(input_tensor.shape)}",
    "",
    "Top-5 ImageNet predictions:",
]

for rank in range(5):
    idx = int(
        top5_idx[0, rank].item()
    )
    prob = float(
        top5_prob[0, rank].item()
    )

    report_lines.append(
        f"{rank + 1}. "
        f"{categories[idx]}: "
        f"{prob * 100:.2f}%"
    )

report_lines += [
    "",
    f"Median latency: {median_latency_ms:.3f} ms/image",
    f"Estimated FPS: {fps:.2f}",
    f"Peak GPU memory: {peak_memory_gb:.2f} GB",
]

(
    OUTPUT_DIR
    / "03_report.txt"
).write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print()
print("SAVED OUTPUTS")
print("-" * 78)
print(
    OUTPUT_DIR
    / "01_input_image.png"
)
print(
    OUTPUT_DIR
    / "02_top5_predictions.png"
)
print(
    OUTPUT_DIR
    / "03_report.txt"
)
print("=" * 78)
