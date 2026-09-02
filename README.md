# Assignment 1: Transformers from Scratch, Architectural Ablations, and Byte Latent Transformers (BLT)

This repository contains a from-scratch PyTorch implementation of an Encoder-Decoder Sequence-to-Sequence Transformer designed to translate encrypted binary sequences into plaintext English. It implements all core transformer components without using high-level PyTorch abstractions (like `nn.Transformer` or `nn.MultiheadAttention`) and conducts a controlled ablation study across five architectural configurations (C1–C5).

---

## 🔗 Project Links

- **Hugging Face Model Checkpoints:** [https://huggingface.co/phani4104/Transformer](https://huggingface.co/phani4104/Transformer)
- **Weights & Biases (WandB) Project:** [https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1)
  - **C1 (Baseline):** [WandB Run j2scymv0](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/j2scymv0)
  - **C2 (RoPE):** [WandB Run 0z87mvj2](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/0z87mvj2)
  - **C3 (GQA):** [WandB Run m4caixvc](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/m4caixvc)
  - **C4 (RMSNorm):** [WandB Run swevqufm](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/swevqufm)
  - **C5 (BLT Token-Free):** [WandB Run o8nwzt18](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/o8nwzt18)

---

## 📁 Repository Structure

```
├── src/
│   ├── models/
│   │   ├── attention.py       # Scaled Dot-Product, Multi-Head Attention (MHA), Grouped-Query Attention (GQA)
│   │   ├── positional.py      # Sinusoidal Absolute Positional Encoding & Rotary Position Embedding (RoPE)
│   │   ├── norm.py            # Pre-LayerNorm & RMSNorm
│   │   ├── blt.py             # Local Byte Encoder & Decoder patch modules for Byte Latent Transformer
│   │   └── transformer.py     # Seq2Seq Transformer model assembly
│   ├── dataset.py             # Tokenized BPE vs Token-Free Byte dataset loaders & collators
│   ├── train.py               # Main training and evaluation engine with WandB logging
│   └── utils.py               # Evaluation metrics (Bit Acc, Seq Acc, Levenshtein, BLEU, ROUGE) & plotting
├── outputs/
│   ├── checkpoints/           # Saved model checkpoints (*_best.pt, *_final.pt)
│   ├── logs/                  # Training history & test evaluation JSON logs
│   ├── plots/                 # Training and validation loss curves
│   └── tokenizers/            # BPE tokenizer vocabulary and merge tables
├── dataset/                   # Dataset directory containing line-aligned cipher and plain text
├── plan.md                    # Detailed execution plan
├── README.md                  # Setup, reproduction instructions, and results
└── Report.pdf                 # Final 6-page ablation study report
```

---

## ⚙️ Architectural Configurations (Ablation Study)

Each variation (C2–C5) alters **exactly one** component from the baseline model (C1) to isolate its impact under identical hyperparameters (4 layers, embedding dimension $d_{\text{model}}=256$, 8 attention heads, FFN hidden dimension 1024, 20 epochs, AdamW optimizer with learning rate $3 \times 10^{-4}$).

| Configuration | Positional Encoding | Attention Mechanism | Normalization | Tokenization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **C1 (Base)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | Standard Subword (BPE) |
| **C2 (Positional)** | **RoPE (Rotary)** | Multi-Head Attention (MHA) | LayerNorm | Standard Subword (BPE) |
| **C3 (Attention)** | Sinusoidal Absolute | **Grouped-Query Attention (GQA, 4 groups)** | LayerNorm | Standard Subword (BPE) |
| **C4 (Normalization)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | **RMSNorm** | Standard Subword (BPE) |
| **C5 (Tokenization)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | **BLT (Token-Free Bytes, patch=16)** |

---

## 📊 Experimental Results & Benchmarks

All models were evaluated on the held-out test split (500 sequences) using **greedy decoding**:

| Config | Positional | Attention | Norm | Tokenizer | Bit Acc (%) | Seq Acc (%) | Levenshtein Dist. | BLEU | ROUGE-L | Peak GPU (MB) | Total Train Time |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Sinusoidal | MHA | LayerNorm | Subword | 54.96% | 0.00% | 494.61 | 0.0127 | 0.1711 | 503.9 MB | 2223 s |
| **C2** | **RoPE** | MHA | LayerNorm | Subword | **86.37%** | **24.40%** | **86.45** | **0.8324** | **0.9420** | 503.9 MB | 2240 s |
| **C3** | Sinusoidal | **GQA** | LayerNorm | Subword | 56.73% | 0.00% | 484.63 | 0.0111 | 0.1650 | 503.9 MB | 2073 s |
| **C4** | Sinusoidal | MHA | **RMSNorm** | Subword | 55.48% | 0.00% | 494.01 | 0.0121 | 0.1656 | 503.9 MB | 2062 s |
| **C5** | Sinusoidal | MHA | LayerNorm | **BLT** | 44.81% | 0.00% | 472.06 | N/A | N/A | 511.9 MB | **1756 s** |

### Key Findings & Tradeoffs:
1. **RoPE (C2) Breakthrough:** Replacing Sinusoidal embeddings with Rotary Position Embeddings (RoPE) yielded a massive performance leap (Bit Accuracy from $54.96\%$ to $86.37\%$, Sequence Accuracy from $0\%$ to $24.40\%$, BLEU from $0.013$ to $0.832$). RoPE preserves relative token offsets directly within query-key dot products, crucial for deciphering positional dependencies in encrypted bitstreams.
2. **GQA (C3) Efficiency:** Grouped-Query Attention ($8$ query heads grouped into $4$ key-value heads) reduced parameter size (~$83.8$ MB checkpoint vs ~$93.3$ MB) and trimmed epoch training times by $\sim 7\%$ without loss in representation capability.
3. **RMSNorm (C4) Speedup:** RMSNorm avoids mean-centering computations, resulting in consistent $\sim 7\%$ faster training than standard LayerNorm with matching loss and accuracy.
4. **BLT (C5) Token-Free Tradeoffs:** Operating directly on raw byte patches (patch size 16) achieved the fastest overall training time ($1756$ s vs $2223$ s for C1) by bypassing subword overhead, though fixed-patch compression presents distinct representation tradeoffs on encrypted sequences.

---

## 🚀 Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12 / 3.14)
- PyTorch 2.0+ (CUDA recommended)
- `wandb`, `huggingface_hub`

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv anlp
source anlp/bin/activate
pip install torch torchvision torchaudio wandb huggingface_hub matplotlib
```

---

## 🏃 Reproduction & Usage

### 1. Train Any Configuration
To train a configuration, pass the config ID (`C1` through `C5`):

```bash
# Run Baseline (C1)
python -m src.train --config C1 --epochs 20 --batch_size 4

# Run RoPE (C2)
python -m src.train --config C2 --epochs 20 --batch_size 4

# Run GQA (C3)
python -m src.train --config C3 --epochs 20 --batch_size 4

# Run RMSNorm (C4)
python -m src.train --config C4 --epochs 20 --batch_size 4

# Run Byte Latent Transformer (C5)
python -m src.train --config C5 --epochs 20 --batch_size 4
```

### 2. Quick Smoke Test (Tiny Subset)
Verify the complete pipeline end-to-end on a mini subset:
```bash
python -m src.train --config C1 --smoke
```

### 3. Evaluate Checkpoint on Test Set (Greedy Decoding)
```bash
python -m src.train --config C2 --evaluate_only --checkpoint outputs/checkpoints/A1_C2_rope_best.pt
```

---

## 📦 Hugging Face Checkpoint Syncing
To download or upload checkpoints to Hugging Face:
```bash
python upload_to_hf.py
```
Checkpoints can be loaded in PyTorch using `torch.load("checkpoints/A1_C2_rope_best.pt", map_location="cpu")`.
