
import math

import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("SELF-ATTENTION DEMO")
print("=" * 72)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
print()

tokens = ["Kedi", "mamada", "uyuyor", "bugün"]

X = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
    device=device,
)

print("Token embedding matrisi X")
print(f"Shape: {tuple(X.shape)}")
print(X)
print()

W_Q = torch.eye(4, device=device)
W_K = torch.eye(4, device=device)
W_V = torch.diag(torch.tensor([0.5, 0.5, 1.0, 1.0], device=device))

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q shape:", tuple(Q.shape))
print("K shape:", tuple(K.shape))
print("V shape:", tuple(V.shape))
print()

d_k = Q.shape[-1]
scores = (Q @ K.T) / math.sqrt(d_k)
attention_weights = torch.softmax(scores, dim=-1)
output = attention_weights @ V

print("Ham attention skorları:")
print(scores)
print()

print("Softmax sonrası attention ağırlıkları:")
print(attention_weights)
print()

print("Her satırın toplamı:")
print(attention_weights.sum(dim=-1))
print()

print("Attention çıktısı:")
print(output)
print()

print("Her token en çok hangi token'a dikkat ediyor?")
print("-" * 72)

for i, token in enumerate(tokens):
    strongest_idx = int(torch.argmax(attention_weights[i]).item())
    strongest_value = float(attention_weights[i, strongest_idx].item())
    print(f"{token:>8} -> {tokens[strongest_idx]:<8} (weight={strongest_value:.3f})")

weights_cpu = attention_weights.detach().cpu().numpy()

plt.figure(figsize=(7, 6))
plt.imshow(weights_cpu)
plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Key token")
plt.ylabel("Query token")
plt.title("Scaled Dot-Product Self-Attention")

for i in range(len(tokens)):
    for j in range(len(tokens)):
        plt.text(j, i, f"{weights_cpu[i, j]:.2f}", ha="center", va="center")

plt.colorbar(label="Attention weight")
plt.tight_layout()
plt.savefig("attention_matrix.png", dpi=180)
plt.close()

print()
print("Kaydedildi: attention_matrix.png")
print("=" * 72)
