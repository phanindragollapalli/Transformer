# Assignment 1: Transformers from Scratch, Architectural Ablations, and Byte Latent Transformers (BLT)

This repository contains a from-scratch PyTorch implementation of an Encoder-Decoder Sequence-to-Sequence Transformer designed to translate encrypted binary sequences into plaintext English. It implements all core transformer components without using high-level PyTorch abstractions (like `nn.Transformer` or `nn.MultiheadAttention`) and conducts a controlled ablation study across five architectural configurations (C1–C5).

---

## 🔗 Project Links

- **Hugging Face Model Checkpoints:** [https://huggingface.co/phani4104/Transformer](https://huggingface.co/phani4104/Transformer)
- **Weights & Biases (WandB) Project:** [https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1)
  - **C1 (Baseline):** [WandB Run rmflfhfi](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/rmflfhfi)
  - **C2 (RoPE):** [WandB Run p66vs720](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/p66vs720)
  - **C3 (GQA):** [WandB Run zr218vg4](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/zr218vg4)
  - **C4 (RMSNorm):** [WandB Run ux2iiyfy](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/ux2iiyfy)
  - **C5 (BLT Token-Free):** [WandB Run t2fsc4do](https://wandb.ai/phani1729-iiit-hyderabad/ANLP_A1/runs/t2fsc4do)

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
├── README.md                  # Setup, reproduction instructions, and results
└── Report.pdf                 # Final ablation study report
```

---

## ⚙️ Architectural Configurations (Ablation Study)

Each variation (C2–C5) alters **exactly one** component from the baseline model (C1) to isolate its impact under identical hyperparameters (4 layers, embedding dimension $d_{\text{model}}=256$, 8 attention heads, FFN hidden dimension 1024, 30 epochs, AdamW optimizer with learning rate $3 \times 10^{-4}$).

| Configuration | Positional Encoding | Attention Mechanism | Normalization | Tokenization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **C1 (Base)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | Standard Subword (BPE) |
| **C2 (Positional)** | **RoPE (Rotary)** | Multi-Head Attention (MHA) | LayerNorm | Standard Subword (BPE) |
| **C3 (Attention)** | Sinusoidal Absolute | **Grouped-Query Attention (GQA, 4 groups)** | LayerNorm | Standard Subword (BPE) |
| **C4 (Normalization)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | **RMSNorm** | Standard Subword (BPE) |
| **C5 (Tokenization)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | **BLT (Token-Free Bytes, patch=16)** |

---

## 📊 Experimental Results & Benchmarks

All models were evaluated on the held-out test split ($4,988$ sequences) using **greedy decoding**:

| Config | Positional | Attention | Norm | Tokenizer | Bit Acc (%) | Seq Acc (%) | Levenshtein Dist. | BLEU | ROUGE-L | Peak GPU (MB) | Total Train Time |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Sinusoidal | MHA | LayerNorm | Subword | 99.01% | 89.78% | 0.185 | 0.9742 | 0.9878 | 1212.15 MB | 1693.10 s |
| **C2** | **RoPE** | MHA | LayerNorm | Subword | **99.60%** | **95.06%** | **0.083** | **0.9886** | **0.9930** | 1212.06 MB | 1867.87 s |
| **C3** | Sinusoidal | **GQA** | LayerNorm | Subword | 99.00% | 89.34% | 0.206 | 0.9733 | 0.9880 | 1202.25 MB | 1630.75 s |
| **C4** | Sinusoidal | MHA | **RMSNorm** | Subword | 99.01% | 90.15% | 0.185 | 0.9753 | 0.9880 | 1091.14 MB | 1604.79 s |
| **C5** | Sinusoidal | MHA | LayerNorm | **BLT** | 84.15% | 3.48% | 18.338 | N/A | N/A | **608.31 MB** | **1306.09 s** |

### Key Findings & Tradeoffs:
1. **RoPE (C2) Breakthrough:** Replacing Sinusoidal embeddings with Rotary Position Embeddings (RoPE) yielded the highest accuracy across the board ($95.06\%$ sequence accuracy, $99.60\%$ bit accuracy, $0.083$ edit distance). RoPE preserves relative token offsets directly within query-key dot products, crucial for exact cipher sequence alignment.
2. **GQA (C3) Efficiency:** Grouped-Query Attention ($8$ query heads grouped into $4$ key-value heads) reduced parameter size by $10\%$ ($7.16$ M vs $7.95$ M) and trimmed training time with virtually identical accuracy ($89.34\%$ vs $89.78\%$).
3. **RMSNorm (C4) Speedup:** RMSNorm avoids mean-centering computations, resulting in $10\%$ peak GPU memory savings and $5.2\%$ faster training with matching stability and loss.
4. **BLT (C5) Token-Free Tradeoffs:** Operating directly on raw byte patches (patch size 16) cut peak GPU memory consumption in half (**608 MB vs 1212 MB**) and sped up training by **23%** due to compressing the sequence by $16\times$ in the global transformer. Autoregressive exposure bias across 64 byte steps and fixed-patch boundary bisecting account for the exact sequence match difference.

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
python -m src.train --config C1 --epochs 30 --batch_size 4

# Run RoPE (C2)
python -m src.train --config C2 --epochs 30 --batch_size 4

# Run GQA (C3)
python -m src.train --config C3 --epochs 30 --batch_size 4

# Run RMSNorm (C4)
python -m src.train --config C4 --epochs 30 --batch_size 4

# Run Byte Latent Transformer (C5)
python -m src.train --config C5 --epochs 30 --batch_size 4
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
