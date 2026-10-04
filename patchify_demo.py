
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 128
DATA_ROOT = r"C:\Users\Melike\Desktop\myFolds\YL\DERSLER\LLM\cnn_architecture_benchmark\data"
OUTPUT_DIR = Path("patchify_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 1) DATASET
# ------------------------------------------------------------
dataset = datasets.OxfordIIITPet(
    root=DATA_ROOT,
    split="trainval",
    target_types="category",
    download=True,
)

image_pil, label = dataset[0]

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
to_tensor = transforms.ToTensor()

image_pil = resize(image_pil)
image = to_tensor(image_pil)  # [C, H, W]

print("=" * 72)
print("VISION TRANSFORMER - PATCHIFY DEMO")
print("=" * 72)
print(f"Device          : {device}")
if device.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

print(f"Class label     : {label}")
print(f"Image shape     : {tuple(image.shape)}")
print(f"Image size      : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Patch size      : {PATCH_SIZE} x {PATCH_SIZE}")
print()


# ------------------------------------------------------------
# 2) HOW MANY PATCHES?
# ------------------------------------------------------------
patches_per_side = IMAGE_SIZE // PATCH_SIZE
num_patches = patches_per_side * patches_per_side

print("PATCH COUNT")
print("-" * 72)
print(f"{IMAGE_SIZE} / {PATCH_SIZE} = {patches_per_side}")
print(f"{patches_per_side} x {patches_per_side} = {num_patches} patches")
print()


# ------------------------------------------------------------
# 3) MANUAL PATCHIFY WITH UNFOLD
# ------------------------------------------------------------
# image: [C, H, W]
#
# unfold(1, patch, patch) -> split height
# unfold(2, patch, patch) -> split width
#
# Result:
# [C, patches_h, patches_w, patch_h, patch_w]

patch_grid = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(
    2, PATCH_SIZE, PATCH_SIZE
)

# Rearrange to:
# [patches_h, patches_w, C, patch_h, patch_w]
patch_grid = patch_grid.permute(1, 2, 0, 3, 4)

# Flatten grid:
# [196, 3, 16, 16]
patches_5d = patch_grid.reshape(
    num_patches,
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

# Flatten each patch:
# [196, 3*16*16] = [196, 768]
patch_vectors = patches_5d.reshape(
    num_patches,
    -1,
)

print("PATCH TENSORS")
print("-" * 72)
print(f"Patch image tensor shape : {tuple(patches_5d.shape)}")
print(f"Patch vector shape       : {tuple(patch_vectors.shape)}")
print(
    f"Each patch vector size   : "
    f"3 x {PATCH_SIZE} x {PATCH_SIZE} = {patch_vectors.shape[1]}"
)
print()


# ------------------------------------------------------------
# 4) PATCH EMBEDDING
# ------------------------------------------------------------
# Real ViT models project each flattened patch into a learned embedding.
patch_embedding = nn.Linear(
    patch_vectors.shape[1],
    EMBED_DIM,
).to(device)

patch_vectors_device = patch_vectors.to(device)

embedded_patches = patch_embedding(
    patch_vectors_device
)

print("PATCH EMBEDDING")
print("-" * 72)
print(f"Before embedding : {tuple(patch_vectors.shape)}")
print(f"After embedding  : {tuple(embedded_patches.shape)}")
print(
    f"Meaning          : {num_patches} image patches -> "
    f"{num_patches} tokens, each {EMBED_DIM}-D"
)
print()


# ------------------------------------------------------------
# 5) CLS TOKEN + POSITIONAL EMBEDDING
# ------------------------------------------------------------
cls_token = nn.Parameter(
    torch.zeros(1, 1, EMBED_DIM, device=device)
)

# Add a batch dimension first:
# [196, 128] -> [1, 196, 128]
tokens = embedded_patches.unsqueeze(0)

# [CLS] + 196 patch tokens -> 197 tokens
tokens_with_cls = torch.cat(
    [cls_token, tokens],
    dim=1,
)

positional_embedding = nn.Parameter(
    torch.zeros(
        1,
        num_patches + 1,
        EMBED_DIM,
        device=device,
    )
)

vit_input = tokens_with_cls + positional_embedding

print("TRANSFORMER INPUT")
print("-" * 72)
print(f"Patch tokens        : {tuple(tokens.shape)}")
print(f"+ CLS token         : {tuple(tokens_with_cls.shape)}")
print(f"+ Position embedding: {tuple(vit_input.shape)}")
print()
print(
    f"Final sequence: 1 CLS token + {num_patches} patch tokens "
    f"= {num_patches + 1} tokens"
)
print()


# ------------------------------------------------------------
# 6) SAVE ORIGINAL IMAGE
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(image_pil)
plt.axis("off")
plt.title("Original image - 224 x 224")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_original_image.png",
    dpi=180,
)
plt.close()


# ------------------------------------------------------------
# 7) SAVE PATCH GRID
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.imshow(image_pil)

for x in range(0, IMAGE_SIZE + 1, PATCH_SIZE):
    plt.axvline(
        x=x,
        linewidth=0.7,
    )

for y in range(0, IMAGE_SIZE + 1, PATCH_SIZE):
    plt.axhline(
        y=y,
        linewidth=0.7,
    )

plt.xlim(0, IMAGE_SIZE)
plt.ylim(IMAGE_SIZE, 0)
plt.xticks([])
plt.yticks([])
plt.title(
    f"ViT patch grid: {patches_per_side} x {patches_per_side} "
    f"= {num_patches} patches"
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_patch_grid.png",
    dpi=180,
)
plt.close()


# ------------------------------------------------------------
# 8) SAVE SAMPLE PATCHES
# ------------------------------------------------------------
sample_indices = [
    0,
    1,
    14,
    15,
    60,
    90,
    120,
    150,
    180,
]

fig = plt.figure(figsize=(8, 8))

for plot_idx, patch_idx in enumerate(
    sample_indices,
    start=1,
):
    patch = (
        patches_5d[patch_idx]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    ax = fig.add_subplot(3, 3, plot_idx)
    ax.imshow(patch)
    ax.set_title(f"Patch {patch_idx}")
    ax.axis("off")

fig.suptitle("Example 16 x 16 image patches")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "03_sample_patches.png",
    dpi=180,
)
plt.close(fig)


print("SAVED OUTPUTS")
print("-" * 72)
print(OUTPUT_DIR / "01_original_image.png")
print(OUTPUT_DIR / "02_patch_grid.png")
print(OUTPUT_DIR / "03_sample_patches.png")
print("=" * 72)
