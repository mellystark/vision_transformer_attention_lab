
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms


# ============================================================
# CONFIG
# ============================================================
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 128
NUM_HEADS = 8
MLP_RATIO = 4
OUTPUT_DIR = Path("multihead_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Önce önceki CNN projesindeki Oxford-IIIT Pet verisini arar.
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
# 1) DATASET
# ============================================================
dataset = datasets.OxfordIIITPet(
    root=DATA_ROOT,
    split="trainval",
    target_types="category",
    download=True,
)

image_pil, label = dataset[0]

image_pil = transforms.Resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)(image_pil)

image = transforms.ToTensor()(image_pil)


# ============================================================
# 2) PATCHIFY
# ============================================================
patches_per_side = IMAGE_SIZE // PATCH_SIZE
num_patches = patches_per_side ** 2

patch_grid = (
    image
    .unfold(1, PATCH_SIZE, PATCH_SIZE)
    .unfold(2, PATCH_SIZE, PATCH_SIZE)
    .permute(1, 2, 0, 3, 4)
)

patches = patch_grid.reshape(
    num_patches,
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

patch_vectors = patches.reshape(
    num_patches,
    -1,
).to(device)


# ============================================================
# 3) PATCH EMBEDDING + CLS + POSITION
# ============================================================
patch_projection = nn.Linear(
    patch_vectors.shape[1],
    EMBED_DIM,
).to(device)

patch_tokens = patch_projection(
    patch_vectors
).unsqueeze(0)  # [1, 196, 128]

cls_token = nn.Parameter(
    torch.zeros(
        1,
        1,
        EMBED_DIM,
        device=device,
    )
)

tokens = torch.cat(
    [cls_token, patch_tokens],
    dim=1,
)  # [1, 197, 128]

pos_embedding = nn.Parameter(
    torch.zeros(
        1,
        num_patches + 1,
        EMBED_DIM,
        device=device,
    )
)

x = tokens + pos_embedding


# ============================================================
# 4) MULTI-HEAD SELF-ATTENTION
# ============================================================
mha = nn.MultiheadAttention(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    batch_first=True,
).to(device)

# Self-attention:
# Query = Key = Value = x
attention_output, attention_weights = mha(
    x,
    x,
    x,
    need_weights=True,
    average_attn_weights=False,
)

# attention_weights:
# [batch, heads, query_tokens, key_tokens]


# ============================================================
# 5) RESIDUAL + LAYERNORM + MLP
# ============================================================
norm1 = nn.LayerNorm(
    EMBED_DIM
).to(device)

norm2 = nn.LayerNorm(
    EMBED_DIM
).to(device)

mlp = nn.Sequential(
    nn.Linear(
        EMBED_DIM,
        EMBED_DIM * MLP_RATIO,
    ),
    nn.GELU(),
    nn.Linear(
        EMBED_DIM * MLP_RATIO,
        EMBED_DIM,
    ),
).to(device)

# Attention residual connection
x_after_attention = norm1(
    x + attention_output
)

# Feed-forward / MLP
mlp_output = mlp(
    x_after_attention
)

# Second residual connection
encoder_output = norm2(
    x_after_attention + mlp_output
)

cls_output = encoder_output[:, 0, :]


# ============================================================
# 6) TERMINAL REPORT
# ============================================================
head_dim = EMBED_DIM // NUM_HEADS

print("=" * 76)
print("VISION TRANSFORMER - MULTI-HEAD SELF-ATTENTION DEMO")
print("=" * 76)
print(f"Device               : {device}")

if device.type == "cuda":
    print(
        f"GPU                  : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Image                : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Patch size           : {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Patch count          : {num_patches}")
print(f"Embedding dimension  : {EMBED_DIM}")
print(f"Attention heads      : {NUM_HEADS}")
print(f"Dimension per head   : {head_dim}")
print()

print("TOKEN INPUT")
print("-" * 76)
print(
    f"Patch tokens         : "
    f"{tuple(patch_tokens.shape)}"
)
print(
    f"+ CLS + position     : "
    f"{tuple(x.shape)}"
)
print()

print("MULTI-HEAD SELF-ATTENTION")
print("-" * 76)
print(
    f"Attention output     : "
    f"{tuple(attention_output.shape)}"
)
print(
    f"Attention weights    : "
    f"{tuple(attention_weights.shape)}"
)
print(
    "Meaning             : "
    "[batch, head, query token, key token]"
)
print()

print("ONE HEAD")
print("-" * 76)
print(
    f"Head 0 matrix        : "
    f"{tuple(attention_weights[0, 0].shape)}"
)

cls_row_sum = attention_weights[
    0, 0, 0
].sum()

print(
    f"Head 0 / CLS row sum : "
    f"{cls_row_sum.item():.6f}"
)
print()

print("TRANSFORMER ENCODER BLOCK")
print("-" * 76)
print(
    f"After attention      : "
    f"{tuple(x_after_attention.shape)}"
)
print(
    f"MLP hidden dimension : "
    f"{EMBED_DIM * MLP_RATIO}"
)
print(
    f"After MLP            : "
    f"{tuple(encoder_output.shape)}"
)
print(
    f"Final CLS vector     : "
    f"{tuple(cls_output.shape)}"
)
print()

print("FLOW")
print("-" * 76)
print(
    "(1,197,128)"
    " -> Multi-Head Attention"
    " -> Residual + LayerNorm"
)
print(
    " -> MLP"
    " -> Residual + LayerNorm"
    " -> (1,197,128)"
)
print(
    " -> CLS token"
    " -> (1,128)"
)
print()


# ============================================================
# 7) CLS -> PATCH ATTENTION, EACH HEAD
# ============================================================
# CLS query = index 0
# Patch keys = indices 1..196
cls_to_patches = attention_weights[
    0,
    :,
    0,
    1:,
].detach().cpu()

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6),
)

for head_idx, ax in enumerate(
    axes.flat
):
    grid = cls_to_patches[
        head_idx
    ].reshape(
        patches_per_side,
        patches_per_side,
    )

    im = ax.imshow(grid)
    ax.set_title(
        f"Head {head_idx + 1}"
    )
    ax.axis("off")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

fig.suptitle(
    "CLS token -> image patches attention\n"
    "(untrained random weights)"
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR
    / "01_cls_attention_heads.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# 8) AVERAGE CLS ATTENTION
# ============================================================
avg_cls_attention = (
    cls_to_patches
    .mean(dim=0)
    .reshape(
        patches_per_side,
        patches_per_side,
    )
)

plt.figure(
    figsize=(6, 6)
)
plt.imshow(
    avg_cls_attention
)
plt.title(
    "Average CLS -> patch attention\n"
    "(8 heads, untrained model)"
)
plt.axis("off")
plt.colorbar(
    label="Attention weight"
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR
    / "02_average_cls_attention.png",
    dpi=180,
)
plt.close()


# ============================================================
# 9) ORIGINAL IMAGE FOR REFERENCE
# ============================================================
plt.figure(
    figsize=(6, 6)
)
plt.imshow(image_pil)
plt.title(
    "Reference image"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR
    / "03_reference_image.png",
    dpi=180,
)
plt.close()


print("SAVED OUTPUTS")
print("-" * 76)
print(
    OUTPUT_DIR
    / "01_cls_attention_heads.png"
)
print(
    OUTPUT_DIR
    / "02_average_cls_attention.png"
)
print(
    OUTPUT_DIR
    / "03_reference_image.png"
)
print("=" * 76)
