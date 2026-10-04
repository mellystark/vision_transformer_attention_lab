# Vision Transformer & Attention Lab

A hands-on PyTorch study of **Self-Attention**, **Multi-Head Attention**, and **Vision Transformers (ViT)**, followed by a real image-classification experiment on the **Oxford-IIIT Pet** dataset.

The project was designed as a learning-oriented lab rather than a single benchmark. The workflow starts from the mathematics of attention, moves through image patchification and Transformer encoder mechanics, and finishes with fine-tuning **ViT-B/16**, comparing it with **ResNet18**, and inspecting real model decisions with patch occlusion.

---

## Goals

This project answers five main questions:

1. How does scaled dot-product self-attention work mathematically?
2. How does a Vision Transformer convert an image into tokens?
3. Why does ViT use multi-head attention and Transformer encoder blocks?
4. How well does a pretrained ViT-B/16 perform on Oxford-IIIT Pet?
5. What trade-off appears between ViT-B/16 and a lighter CNN such as ResNet18?

---

## Core Attention Equation

The main attention operation is:

\[
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

where:

- **Q (Query):** what information is being searched for?
- **K (Key):** what information does each token represent?
- **V (Value):** what information is transferred when a token receives attention?

The first experiment in this repository computes this operation directly on a small toy sequence and visualizes the attention matrix.

---

## From Image to Vision Transformer Tokens

For a `224 x 224` RGB image and a `16 x 16` patch size:

\[
224 / 16 = 14
\]

\[
14 \times 14 = 196 \text{ patches}
\]

Each patch contains:

\[
16 \times 16 \times 3 = 768
\]

pixel values after flattening.

In the teaching demo, each patch is projected as:

\[
768 \rightarrow 128
\]

and becomes a token embedding.

A `[CLS]` token is then added:

\[
196 + 1 = 197 \text{ tokens}
\]

The resulting sequence is passed through Transformer encoder blocks.

---

## Multi-Head Attention Demo

The educational demo uses:

```text
Embedding dimension : 128
Attention heads     : 8
Dimension per head  : 16
```

because:

\[
128 / 8 = 16
\]

Each attention head processes token relationships using its own learned projections.

The demo also illustrates:

```text
Tokens
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
MLP
  ↓
Residual Connection
  ↓
LayerNorm
```

---

## Dataset

### Oxford-IIIT Pet

- **37 classes**
- **7,349 total images**
- Train: **2,944**
- Validation: **736**
- Test: **3,669**
- Input resolution: **224 x 224**
- Task: fine-grained cat and dog breed classification

The dataset is useful for this experiment because many classes are visually similar, making it a meaningful fine-grained classification problem.

---

## Hardware & Environment

The experiments were run locally on:

```text
GPU      : NVIDIA GeForce RTX 3060 Laptop GPU
VRAM     : 6 GB
CUDA     : 12.6
PyTorch  : 2.14.1+cu126
AMP      : Enabled
```

Inference latency and FPS are hardware-dependent and should not be interpreted as universal model speeds.

---

# Experiment Pipeline

## 1. Scaled Dot-Product Self-Attention

Script:

```text
attention_demo.py
```

This experiment:

- creates a small token embedding matrix,
- constructs Q, K, and V,
- calculates scaled dot-product attention,
- applies softmax,
- verifies that attention rows sum to 1,
- saves an attention heatmap.

Output:

```text
attention_matrix.png
```

---

## 2. Image Patchification

Script:

```text
patchify_demo.py
```

This experiment takes a real Oxford-IIIT Pet image and demonstrates:

```text
224 x 224 image
↓
16 x 16 patches
↓
196 patches
↓
flattened patch vectors
↓
patch embeddings
↓
+ CLS token
↓
+ positional embedding
↓
197-token Transformer input
```

Generated outputs include:

```text
patchify_outputs/
├── 01_original_image.png
├── 02_patch_grid.png
└── 03_sample_patches.png
```

---

## 3. Multi-Head Self-Attention

Script:

```text
multihead_attention_demo.py
```

This experiment passes the image tokens through an educational Transformer encoder block.

It visualizes:

- individual attention heads,
- average CLS-to-patch attention,
- the original reference image.

Outputs:

```text
multihead_outputs/
├── 01_cls_attention_heads.png
├── 02_average_cls_attention.png
└── 03_reference_image.png
```

The model in this stage is intentionally untrained, so its attention maps are expected to be close to uniform.

---

## 4. Pretrained ViT-B/16 Inference

Script:

```text
pretrained_vit_demo.py
```

An ImageNet-pretrained **ViT-B/16** is evaluated on a sample Oxford-IIIT Pet image before task-specific fine-tuning.

The experiment reports:

- Top-5 ImageNet predictions
- parameter count
- inference latency
- estimated FPS
- GPU memory usage

This stage demonstrates the difference between a pretrained ImageNet classifier and a model fine-tuned specifically for the Oxford-IIIT Pet task.

---

# ViT-B/16 Fine-Tuning

Script:

```text
train_vit_oxford.py
```

The final training strategy uses transfer learning.

## Phase 1: Classifier Warm-up

```text
Backbone       : Frozen
Classifier head: Trainable
Epochs         : 3
Learning rate  : 1e-3
```

## Phase 2: Transformer Fine-Tuning

```text
Last 2 encoder blocks : Trainable
Classifier head       : Trainable
Epochs                 : 8
Learning rate          : 2e-5
```

Additional settings:

```text
Batch size             : 8
Gradient accumulation  : 4
Effective batch size   : 32
Optimizer              : AdamW
Loss                   : CrossEntropyLoss
Mixed precision        : AMP
Model selection metric : Validation Macro-F1
```

---

# Final ViT-B/16 Results

| Metric | Result |
|---|---:|
| Test Accuracy | **93.30%** |
| Test Macro Precision | **0.9340** |
| Test Macro Recall | **0.9326** |
| Test Macro-F1 | **0.9323** |
| Top-5 Accuracy | **99.73%** |
| Best Validation Macro-F1 | **0.9570** |
| Parameters | **85.83M** |
| Checkpoint size | **327.46 MB** |
| Inference latency | **10.69 ms / image** |
| Estimated FPS | **93.54** |
| Training time | **~6.58 min** |

The final fine-tuned model correctly classified:

```text
3423 / 3669 test images
```

with:

```text
246 incorrect predictions
```

---

# ResNet18 vs ViT-B/16

A previous ResNet18 benchmark on the same Oxford-IIIT Pet task is used as a reference.

| Metric | ResNet18 | ViT-B/16 |
|---|---:|---:|
| Test Accuracy | 89.26% | **93.30%** |
| Macro-F1 | 0.8905 | **0.9323** |
| Parameters | **11.2M** | 85.83M |
| Latency | **3.50 ms** | 10.69 ms |
| FPS | **285.7** | 93.54 |

Accuracy improvement:

\[
93.30 - 89.26 = 4.04
\]

percentage points.

The result demonstrates a clear engineering trade-off:

> **ViT-B/16 achieved higher classification quality, while ResNet18 remained much lighter and faster.**

---

# Training-Curve Observations

The final training curves show:

- training loss dropping close to zero,
- training accuracy approaching 100%,
- validation accuracy stabilizing around 95–96%,
- validation Macro-F1 reaching approximately 0.957,
- validation loss reaching a plateau.

This suggests strong learning with a mild overfitting tendency in the later epochs.

---

# Confusion Matrix

The test confusion matrix shows a strong diagonal, indicating that most predictions are assigned to the correct class.

Errors are more common between visually similar breeds.

Examples include:

```text
Boxer ↔ American Bulldog
Basset Hound ↔ Beagle
Egyptian Mau ↔ Bengal
Birman ↔ Ragdoll
```

---

# Visual Model Inspection

Script:

```text
inspect_vit_predictions.py
```

This stage examines the trained model on actual test images rather than relying only on aggregate metrics.

Generated analyses include:

```text
vit_visual_inspection/
├── 01_random_test_predictions.png
├── 02_high_confidence_correct.png
├── 03_high_confidence_errors.png
├── 04_sample_top5_predictions.png
├── 05_patch_occlusion_importance.png
├── patch_importance.npy
└── 06_visual_inspection_report.txt
```

---

## High-Confidence Errors

One of the most important observations is:

\[
\boxed{\text{High confidence} \neq \text{correct prediction}}
\]

The model can be confidently wrong.

Examples observed during inspection include:

```text
Boxer                → American Bulldog
Basset Hound         → Beagle
Egyptian Mau         → Bengal
Birman               → Ragdoll
```

Some of these incorrect predictions were made with more than 99% softmax confidence.

Therefore, softmax confidence should not automatically be interpreted as calibrated real-world correctness probability.

---

# Patch Occlusion Analysis

To inspect which image regions affect the model's prediction, the image is divided into `16 x 16` patches.

Each patch is masked one at a time and the predicted-class probability is measured again.

The patch importance score is:

\[
Importance_i =
P(y|x) -
P(y|x_{\setminus i})
\]

where:

- \(P(y|x)\) is the original predicted-class probability,
- \(P(y|x_{\setminus i})\) is the probability after masking patch \(i\).

A larger positive value indicates that masking that patch reduced confidence more strongly.

This is an **occlusion sensitivity analysis**, not a direct Transformer attention map.

The inspected example showed sensitivity around meaningful parts of the animal, while also demonstrating that background regions can influence the model.

---

# Main Lessons

1. **Vision Transformers treat image patches as a token sequence.**
2. **Self-attention enables global interactions between image regions.**
3. **Multi-head attention allows multiple learned relationships to be modeled in parallel.**
4. **Transfer learning is extremely effective for relatively small image datasets.**
5. **ViT-B/16 outperformed ResNet18 in accuracy and Macro-F1 on this experiment.**
6. **The accuracy gain came with substantially higher parameter count and inference latency.**
7. **High softmax confidence does not guarantee a correct prediction.**
8. **Confusion matrices and visual failure analysis reveal information that aggregate accuracy cannot.**
9. **Patch occlusion provides a concrete way to study which regions influence a model decision.**
10. **Model selection should consider accuracy, latency, memory, hardware, and deployment constraints together.**

---

# Project Structure

```text
vision_transformer_attention_lab/
│
├── README.md
├── attention_demo.py
├── patchify_demo.py
├── multihead_attention_demo.py
├── pretrained_vit_demo.py
├── train_vit_oxford.py
├── inspect_vit_predictions.py
│
├── attention_matrix.png
│
├── patchify_outputs/
│   ├── 01_original_image.png
│   ├── 02_patch_grid.png
│   └── 03_sample_patches.png
│
├── multihead_outputs/
│   ├── 01_cls_attention_heads.png
│   ├── 02_average_cls_attention.png
│   └── 03_reference_image.png
│
├── pretrained_vit_outputs/
│
├── vit_finetune_outputs/
│
├── vit_finetune_final_outputs/
│   ├── history.csv
│   ├── final_metrics.json
│   ├── 01_loss_curve.png
│   ├── 02_accuracy_curve.png
│   ├── 03_macro_f1_curve.png
│   ├── 04_confusion_matrix.png
│   ├── resnet18_vs_vit_b16.csv
│   ├── 05_resnet18_vs_vit_accuracy.png
│   └── 06_resnet18_vs_vit_latency.png
│
└── vit_visual_inspection/
    ├── 01_random_test_predictions.png
    ├── 02_high_confidence_correct.png
    ├── 03_high_confidence_errors.png
    ├── 04_sample_top5_predictions.png
    ├── 05_patch_occlusion_importance.png
    └── 06_visual_inspection_report.txt
```

Large datasets, checkpoints, caches, and generated binary artifacts should not be committed to Git.

---

# Installation

Activate the environment:

```powershell
conda activate llm_ders
```

Install the main dependencies:

```powershell
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

For GPU execution, use a CUDA-compatible PyTorch installation appropriate for the target machine.

---

# Run the Attention Demo

```powershell
python .\attention_demo.py
```

---

# Run the Patchification Demo

```powershell
python .\patchify_demo.py
```

---

# Run the Multi-Head Attention Demo

```powershell
python .\multihead_attention_demo.py
```

---

# Run Pretrained ViT Inference

```powershell
python .\pretrained_vit_demo.py
```

---

# Run the Final ViT Fine-Tuning

```powershell
python .\train_vit_oxford.py --warmup-epochs 3 --finetune-epochs 8 --require-cuda --output-root vit_finetune_final_outputs
```

---

# Inspect Real Test Predictions

After training:

```powershell
python .\inspect_vit_predictions.py --checkpoint .\vit_finetune_final_outputs\best_vit_b16.pth --output-root .\vit_visual_inspection
```

---

## Notes

- The repository does not include the Oxford-IIIT Pet dataset.
- The trained ViT checkpoint is intentionally excluded because of its large file size.
- Results depend on the software stack, random seed, dataset split, augmentation, and hardware.
- Inference latency must be re-measured on the final deployment device.
- Patch occlusion analysis measures sensitivity to perturbation and should not be described as a direct attention map.

---

## Technologies

- Python
- PyTorch
- Torchvision
- CUDA
- AMP
- NumPy
- Pandas
- Matplotlib
- scikit-learn
