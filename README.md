# Assignment 1: Transformers from Scratch, Architectural Ablations, and Byte Latent Transformers (BLT)

This repository contains a **from-scratch PyTorch implementation** of an Encoder-Decoder Sequence-to-Sequence Transformer designed to translate encrypted binary cipher sequences into plaintext English. It implements all core transformer components without using high-level PyTorch abstractions (such as `nn.Transformer` or `nn.MultiheadAttention`) and conducts a strictly controlled ablation study across five architectural configurations (**C1 through C5**).

---

## 🔗 Project Links

- **Hugging Face Model Repository:** [https://huggingface.co/phani4104/Transformer](https://huggingface.co/phani4104/Transformer)
- **Hugging Face Direct Files & Checkpoints Download:** [https://huggingface.co/phani4104/Transformer/tree/main](https://huggingface.co/phani4104/Transformer/tree/main)
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
│   │   └── transformer.py     # FeedForward, Encoder/Decoder blocks, full Seq2Seq assembly & greedy decoding
│   ├── dataset.py             # Tokenized BPE vs Token-Free Byte dataset loaders & collators
│   ├── train.py               # Main training and evaluation engine with WandB logging & CLI
│   └── utils.py               # Evaluation metrics (Bit Acc, Seq Acc, Levenshtein, BLEU, ROUGE) & plotting
├── outputs/
│   ├── checkpoints/           # Saved model checkpoints (*_best.pt, *_final.pt)
│   ├── logs/                  # Training history & test evaluation JSON logs for C1–C5
│   ├── plots/                 # Training and validation loss curves for all configurations
│   └── tokenizers/            # BPE tokenizer vocabulary and merge tables
├── dataset/                   # Dataset directory containing line-aligned cipher and plain text
├── upload_to_hf.py            # Automated Hugging Face Hub checkpoint downloader and uploader
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete reproduction and documentation guide
└── Report.pdf                 # Final 5-page ablation study report (12pt Times, 1-inch margins)
```

---

## ⚙️ Architectural Configurations (Ablation Study)

Each variation (**C2–C5**) alters **exactly one** component from the baseline model (**C1**) to isolate its impact under identical hyperparameters:
- **Depth & Width:** 4 Encoder Layers, 4 Decoder Layers, $d_{\text{model}} = 256$, $d_{\text{ff}} = 1024$, 8 Attention Heads.
- **Optimization:** 30 Epochs, AdamW ($\beta_1=0.9, \beta_2=0.98, \text{weight decay}=0.01$), learning rate $3 \times 10^{-4}$, batch size 64, gradient clipping 1.0.
- **Data Representation:** Aligned 64-byte plaintext chunks mapped to 512-bit cipher chunks, with 80/10/10 train/val/test splits (seed 42).

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
1. **RoPE (C2) Breakthrough:** Replacing Sinusoidal embeddings with Rotary Position Embeddings (RoPE) yielded the highest accuracy across the board (**95.06%** sequence accuracy, **99.60%** bit accuracy, **0.083** edit distance). RoPE preserves relative token offsets directly within query-key dot products, crucial for exact cipher sequence alignment.
2. **GQA (C3) Efficiency:** Grouped-Query Attention ($8$ query heads grouped into $4$ key-value heads) reduced parameter size by $10\%$ ($7.16$ M vs $7.95$ M) and trimmed training time with virtually identical accuracy ($89.34\%$ vs $89.78\%$).
3. **RMSNorm (C4) Speedup:** RMSNorm avoids mean-centering computations, resulting in $10\%$ peak GPU memory savings and $5.2\%$ faster training with matching stability and loss.
4. **BLT (C5) Token-Free Tradeoffs:** Operating directly on raw byte patches (patch size 16) cut peak GPU memory consumption in half (**608 MB vs 1212 MB**) and sped up training by **23%** due to compressing the sequence by $16\times$ in the global transformer. Autoregressive exposure bias across 64 byte steps and fixed-patch boundary bisecting account for the exact sequence match difference.

---

## 🚀 Environment Setup & Installation

### 1. Prerequisites
- Linux / macOS / Windows with WSL
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12)
- NVIDIA GPU with CUDA recommended (CPU execution is fully supported)

### 2. Create Virtual Environment & Install Dependencies
```bash
# Clone repository if needed and enter directory
git clone https://github.com/phani4104/Transformer.git
cd Transformer

# Create a virtual environment
python3 -m venv anlp_env
source anlp_env/bin/activate

# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: If installing manually without `requirements.txt`:)*
```bash
pip install torch torchvision torchaudio numpy matplotlib wandb huggingface_hub Levenshtein nltk rouge-score
```

---

## 📦 Checkpoint Syncing & Downloads (Hugging Face)

All trained model weights for configurations **C1 through C5** are hosted and publicly downloadable from Hugging Face:
- 🌐 **Repository:** [https://huggingface.co/phani4104/Transformer](https://huggingface.co/phani4104/Transformer)
- 📂 **Direct Files & Versions Tree:** [https://huggingface.co/phani4104/Transformer/tree/main](https://huggingface.co/phani4104/Transformer/tree/main)

### Direct Checkpoint Download Links

You can download individual model weights directly using your browser or via the links below:

| Configuration | Model Variant | Checkpoint File | Direct Download Link |
| :--- | :--- | :--- | :--- |
| **C1** | Baseline (Sinusoidal + MHA + LayerNorm) | `A1_C1_baseline_best.pt` | [Download C1 Best Checkpoint](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C1_baseline_best.pt) |
| **C2** | Rotary Positional Embedding (RoPE) | `A1_C2_rope_best.pt` | [Download C2 Best Checkpoint](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C2_rope_best.pt) |
| **C3** | Grouped-Query Attention (GQA) | `A1_C3_gqa_best.pt` | [Download C3 Best Checkpoint](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C3_gqa_best.pt) |
| **C4** | Root Mean Square Norm (RMSNorm) | `A1_C4_rmsnorm_best.pt` | [Download C4 Best Checkpoint](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C4_rmsnorm_best.pt) |
| **C5** | Byte Latent Transformer (BLT) | `A1_C5_blt_best.pt` | [Download C5 Best Checkpoint](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C5_blt_best.pt) |

*(Final epoch checkpoints are also available in the repository tree: [`A1_C1_baseline_final.pt`](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C1_baseline_final.pt), [`A1_C2_rope_final.pt`](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C2_rope_final.pt), [`A1_C3_gqa_final.pt`](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C3_gqa_final.pt), [`A1_C4_rmsnorm_final.pt`](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C4_rmsnorm_final.pt), [`A1_C5_blt_final.pt`](https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C5_blt_final.pt))*

---

### Command-Line Download Methods

#### Method 1: Automated Download Script (Recommended)
Automatically download all checkpoints into `outputs/checkpoints/`:
```bash
python upload_to_hf.py --action download
```

#### Method 2: Direct `wget` / `curl` Download
To download a specific checkpoint directly into `outputs/checkpoints/`:
```bash
mkdir -p outputs/checkpoints

# Example: Download C2 (RoPE) checkpoint
wget -P outputs/checkpoints/ https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C2_rope_best.pt

# Or using curl:
curl -L -o outputs/checkpoints/A1_C2_rope_best.pt https://huggingface.co/phani4104/Transformer/resolve/main/checkpoints/A1_C2_rope_best.pt
```

#### Method 3: Upload Checkpoints (for maintainers)
```bash
python upload_to_hf.py --action upload --token <YOUR_HUGGINGFACE_TOKEN>
```
*(Or export `HF_TOKEN=<token>` and run `python upload_to_hf.py --action upload`)*

---

## 🏃 Running Experiments

All training, validation, evaluation, and plotting are handled through `src/train.py`.

### 1. Quick Smoke Test (Sanity Check)
Run a fast 2-epoch sanity check on 64 samples to verify your environment, PyTorch tensors, and modules without waiting for a full training cycle:
```bash
python -m src.train --config C1 --smoke
```

### 2. Train Individual Configurations
Train any of the 5 configurations on the full dataset (30 epochs):

```bash
# Configuration 1: Baseline (Sinusoidal, MHA, LayerNorm, BPE)
python -m src.train --config C1

# Configuration 2: Rotary Position Embeddings (RoPE)
python -m src.train --config C2

# Configuration 3: Grouped-Query Attention (GQA, 4 KV groups)
python -m src.train --config C3

# Configuration 4: Root Mean Square Normalization (RMSNorm)
python -m src.train --config C4

# Configuration 5: Byte Latent Transformer (BLT, Token-Free byte patches)
python -m src.train --config C5
```

### 3. Train All Configurations Sequentially
To train all configurations (`C1` through `C5`) end-to-end:
```bash
python -m src.train --config all
```

### 4. Running Without Weights & Biases (Offline Mode)
If you do not have a WandB account or wish to train completely offline, use the `--no-wandb` flag:
```bash
python -m src.train --config C1 --no-wandb
```

### 5. Overriding Epochs, Batch Size, or Device
You can override default hyperparameters via CLI arguments:
```bash
# Train C2 for 10 epochs with batch size 32 on CPU
python -m src.train --config C2 --epochs 10 --batch-size 32 --device cpu

# Force CUDA device 0
python -m src.train --config C4 --device cuda:0
```

---

## 🧪 Evaluating Checkpoints (Greedy Decoding)

You can evaluate any saved or downloaded checkpoint directly on the held-out test split using **greedy decoding**:

```bash
# Step 1: Ensure checkpoints are downloaded
python upload_to_hf.py --action download

# Step 2: Evaluate a specific configuration checkpoint
python -m src.train --config C1 --evaluate-only --checkpoint outputs/checkpoints/A1_C1_baseline_best.pt
python -m src.train --config C2 --evaluate-only --checkpoint outputs/checkpoints/A1_C2_rope_best.pt
python -m src.train --config C3 --evaluate-only --checkpoint outputs/checkpoints/A1_C3_gqa_best.pt
python -m src.train --config C4 --evaluate-only --checkpoint outputs/checkpoints/A1_C4_rmsnorm_best.pt
python -m src.train --config C5 --evaluate-only --checkpoint outputs/checkpoints/A1_C5_blt_best.pt
```

The script will report:
- **Bit-Level Accuracy**
- **Sequence Accuracy** (exact match)
- **Levenshtein Distance**
- **BLEU & ROUGE-L** (for tokenized models C1–C4)
- Sample qualitative predictions (Cipher input, Ground Truth, Predicted Text)

---

## 📈 Generated Outputs & Artifacts

After training runs, artifacts are saved inside the `outputs/` directory:
- **`outputs/checkpoints/`**: Contains `<run_name>_best.pt` (lowest validation loss) and `<run_name>_final.pt`.
- **`outputs/logs/`**: JSON files with per-epoch loss history, elapsed training time, peak VRAM, and full test set metrics.
- **`outputs/plots/`**: PNG figures displaying training loss vs. validation loss curves across epochs (e.g., `A1_C1_baseline_curves.png`).
- **`outputs/tokenizers/`**: Trained Byte-Pair Encoding (BPE) merge tables and vocabularies for both source and target text.

---

## 📜 Report
The final analytical report is compiled as **[Report.pdf](Report.pdf)** in the root directory. It follows strict academic submission criteria (maximum 5 pages, 12pt Times New Roman, 1-inch margins), presenting:
- Detailed mathematical formulations of all from-scratch modules.
- Controlled ablation comparison table and loss convergence plots.
- In-depth qualitative and computational analysis of BLT tradeoffs.
