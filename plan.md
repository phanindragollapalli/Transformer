# Assignment 1 Execution Plan

## 1. Assignment Understanding

This assignment asks for a **from-scratch PyTorch implementation** of a sequence-to-sequence Encoder-Decoder Transformer for mapping encrypted binary sequences to plaintext data.

The implementation must:

- Avoid built-in high-level transformer layers such as `nn.Transformer` and `nn.MultiheadAttention`.
- Implement core components manually.
- Support a **controlled ablation study** across **five configurations (C1-C5)**.
- Use **WandB** for experiment tracking.
- Host trained checkpoints on **Hugging Face**.
- Produce a final submission with the exact requested directory structure.

The five required configurations are:

- `C1`: Baseline with sinusoidal positional encoding, multi-head attention, LayerNorm, standard subword tokenization.
- `C2`: Same as C1 except positional encoding changes to RoPE.
- `C3`: Same as C1 except attention changes to GQA.
- `C4`: Same as C1 except normalization changes to RMSNorm.
- `C5`: Same as C1 except tokenization changes to a simplified BLT-style token-free byte pipeline.

The key requirement is that **C2-C5 each change exactly one component relative to C1**.

---

## 2. High-Level Execution Strategy

The safest way to complete the assignment is:

1. Ground the design in the extracted dataset and define preprocessing around its real constraints.
2. Build the reusable baseline infrastructure first.
3. Implement each required architectural variant in isolated modules.
4. Train and validate the baseline before running the ablations.
5. Run all five configurations under tightly controlled hyperparameters.
6. Collect metrics, plots, speed, and memory statistics.
7. Prepare final artifacts: code, outputs, README, report, WandB links, and Hugging Face checkpoints.

This order reduces risk because the baseline becomes the reference point for every later comparison.

---

## 3. Detailed Step-by-Step Plan

## Phase 1: Dataset-Grounded Setup

### Dataset facts confirmed from the extracted files

- The dataset is already extracted under `Dataset_A1/`.
- It contains exactly two parallel files:
  - `brown_cipher.txt`
  - `brown_plain.txt`
- Both files contain `5,000` lines and are strictly line-aligned.
- The source side uses only `0` and `1`.
- The target side is plain ASCII English text with `53` unique characters in the corpus.
- Source length is always exactly `8x` the target length, which strongly suggests each plaintext character was converted to one byte and then encrypted as an 8-bit binary string.
- Sequence lengths are highly variable:
  - cipher length: min `168`, max `21,360`, avg `4,780.85`
  - plain length: min `21`, max `2,670`, avg `597.61`
- There are `4,997` unique pairs, with only `3` duplicated examples and no conflicting cipher-to-plaintext mappings.

These facts change the implementation strategy: the main challenge is not dataset extraction, but handling long variable-length aligned sequences without exploding memory.

### Step 1.1: Lock the dataset contract

- Treat line `k` in `brown_cipher.txt` and line `k` in `brown_plain.txt` as one training pair.
- Build a reproducible train/validation/test split from the `5,000` aligned pairs because the dataset does not ship with predefined splits.
- Record the core assumption explicitly in `README.md`:
  - each source example is a binary sequence representing encrypted bytes
  - each target example is the corresponding plaintext character sequence
- Preserve the raw files unchanged and make every downstream artifact derive from them deterministically.

### Step 1.2: Decide the preprocessing interfaces

- For `C1-C4`, define the standard tokenized pipeline:
  - source representation: reshape binary strings into byte units before tokenization so the encoder does not attend over eight times more positions than necessary
  - target representation: standard tokenized English text
  - special tokens if needed: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- For `C5`, define the token-free/byte-level pipeline:
  - byte extraction from the binary source by grouping every 8 bits into one byte
  - patch grouping rule
  - local encoder input shape
  - local decoder output shape
- Because the sequences are long, decide up front whether training will use:
  - full sequences with bucketing
  - fixed-size chunks/windows
  - or a capped max length with documented truncation
- Document all assumptions in `README.md` so the design is defensible during viva.

### Step 1.3: Create the required project structure

- Create:
  - `src/models/attention.py`
  - `src/models/positional.py`
  - `src/models/norm.py`
  - `src/models/blt.py`
  - `src/dataset.py`
  - `src/train.py`
  - `src/utils.py`
  - `outputs/`
  - `README.md`
- Keep experiments reproducible by planning for:
  - random seeds
  - config-driven runs
  - deterministic logging names

### Phase 1 Incremental Steps

- [x] Read `brown_cipher.txt` and `brown_plain.txt` together and assert equal line counts.
- [x] Save dataset statistics and length summaries for reference.
- [x] Fix a reproducible split strategy for train, validation, and test.
- [x] Document the binary-to-byte conversion assumption.
- [x] Decide the sequence length handling strategy.
- [x] Create the required source and output directories.
- [x] Add starter module files so the implementation can proceed incrementally.

---

## Phase 2: Define Common Training and Evaluation Infrastructure

### Step 2.1: Define a single experiment configuration system

- Create a clean configuration object or dictionary containing:
  - model name / run name
  - configuration ID (`C1` to `C5`)
  - vocabulary settings
  - embedding dimension
  - number of heads
  - number of query groups for GQA
  - feed-forward hidden size
  - number of encoder/decoder layers
  - dropout
  - learning rate
  - batch size
  - number of epochs
  - max source length
  - max target length
  - whether binary-to-byte compaction is enabled
  - length bucketing / dynamic batching settings
  - positional encoding type
  - normalization type
  - attention type
  - tokenization type
  - BLT patch size if applicable

### Step 2.2: Fix shared hyperparameters for controlled comparison

- Choose a common training setup for all runs:
  - same depth
  - same width
  - same optimizer
  - same scheduler if used
  - same effective token budget per batch where memory permits
  - same training/validation split
  - same decoding strategy
- Only let the required component vary across configurations.
- If `C5` forces a practical change, record the reason very clearly and mention it in the report.
- Since the longest sequences are large, prefer dynamic batching by token count or tight length buckets rather than a naive fixed batch size.

### Step 2.3: Implement core utility functions

- Build utilities for:
  - seed setting
  - mask creation
  - causal masks for decoder self-attention
  - padding masks
  - checkpoint saving/loading
  - metric accumulation
  - training history plotting
  - memory/time measurement
- Add a utility to compute:
  - training epoch time
  - validation time
  - peak GPU memory if CUDA is available

### Phase 2 Incremental Steps

- [x] Create the shared config structure for `C1-C5`.
- [x] Define baseline hyperparameters for `C1`.
- [x] Encode single-change overrides for `C2-C5`.
- [x] Add reproducibility helpers such as seed setting.
- [x] Add masking helpers for padding and causality.
- [x] Add checkpoint save and load helpers.
- [x] Add metric history and runtime tracking helpers.
- [x] Add optional WandB hooks that do not crash if WandB is unavailable.

---

## Phase 3: Implement the Data Pipeline

### Step 3.1: Build the standard dataset loader for `C1-C4`

- In `src/dataset.py`, implement a dataset class that:
  - reads source-target pairs
  - converts each binary source string into byte-sized units
  - tokenizes them
  - converts them to IDs
  - adds `<bos>` and `<eos>` to target sequences if needed
  - pads batches properly
- Implement a vocabulary builder if the dataset does not already provide one.
- Prefer one consistent tokenization approach for `C1-C4` because the assignment expects “standard subword”.

### Step 3.2: Choose and implement the standard tokenization method

- Decide whether to use:
  - character-level tokenization
  - byte-level tokenization
  - BPE/subword tokenization
- Since the assignment explicitly says “standard subword”, the better plan is:
  - train a lightweight subword tokenizer on the plaintext side
  - tokenize the source at the byte-symbol level after converting 8-bit groups into byte tokens
- Save tokenizer artifacts so experiments are reproducible.
- Keep the tokenizer small and practical; with only `5,000` samples, an overly large subword vocabulary is unlikely to help.

### Step 3.3: Build the token-free dataset loader for `C5`

- Implement a separate path for BLT-style loading:
  - read raw bytes directly from the plaintext and byte-compacted cipher stream
  - avoid a learned vocabulary lookup in the usual subword sense
  - chunk bytes into local patches
  - return byte tensors and patch metadata
- Ensure batching works for variable numbers of patches.

### Step 3.4: Verify data loaders thoroughly

- Run sanity checks on:
  - tensor shapes
  - padding correctness
  - mask correctness
  - vocabulary size
  - byte range validity for BLT
- Print one batch example for both pipelines before training.

### Phase 3 Incremental Steps

- [ ] Implement binary-string to byte conversion.
- [ ] Implement the standard source tokenizer for byte-compacted cipher inputs.
- [ ] Implement the standard target tokenizer for plaintext.
- [ ] Fit tokenizer artifacts on the training split only.
- [ ] Build the dataset and collate path for `C1-C4`.
- [ ] Build the raw-byte dataset and collate path for `C5`.
- [ ] Run a one-batch sanity check for `C1-C4`.
- [ ] Run a one-batch sanity check for `C5`.
- [ ] Save tokenizer and preprocessing artifacts for reproducibility.

---

## Phase 4: Implement the Required Transformer Modules

### Step 4.1: Implement scaled dot-product attention

- In `src/models/attention.py`, implement:
  - query-key dot product
  - scale by `sqrt(d_k)`
  - optional masking
  - softmax
  - weighted sum with values
- Make the implementation reusable for:
  - encoder self-attention
  - decoder self-attention
  - encoder-decoder cross-attention

### Step 4.2: Implement Multi-Head Attention (MHA)

- Build projection layers for:
  - queries
  - keys
  - values
  - output projection
- Split into heads manually.
- Apply attention independently per head.
- Concatenate head outputs and project back.
- Ensure shape handling is correct for batched inputs.

### Step 4.3: Implement Grouped-Query Attention (GQA)

- Reuse as much of the MHA logic as possible.
- Implement a setup where:
  - there are more query heads
  - keys and values are shared across groups of query heads
- Make the number of query groups configurable.
- Validate that GQA preserves expected output dimensions.

### Step 4.4: Implement positional encodings

- In `src/models/positional.py`, implement:
  - sinusoidal absolute positional encoding
  - rotary positional embedding (RoPE)
- Make positional encoding selection configurable.
- For RoPE:
  - apply it directly to query/key representations
  - ensure dimension pairing is valid
  - test that sequence length broadcasting works correctly

### Step 4.5: Implement normalization layers

- In `src/models/norm.py`, implement:
  - custom LayerNorm
  - custom RMSNorm
- Avoid depending on PyTorch’s high-level transformer internals.
- Verify numerical stability with epsilon handling.

### Step 4.6: Implement feed-forward networks

- Build the position-wise FFN used in each transformer block:
  - linear projection up
  - nonlinearity such as ReLU or GELU
  - linear projection down
  - dropout if desired
- Keep FFN identical across all configs unless the assignment asks otherwise.

### Step 4.7: Implement encoder and decoder blocks

- Build encoder blocks with:
  - normalization
  - self-attention
  - residual connection
  - normalization
  - FFN
  - residual connection
- Build decoder blocks with:
  - masked self-attention
  - cross-attention
  - FFN
  - residual pathways
- Since the assignment mentions **Pre-Layer Normalization**, use the pre-norm structure consistently unless a justified alternative is needed.

### Step 4.8: Implement the full encoder-decoder transformer

- Assemble:
  - source embeddings
  - target embeddings
  - positional encoding logic
  - stacked encoder
  - stacked decoder
  - output projection to vocabulary logits for tokenized models
- Add helper methods for:
  - forward pass during training
  - greedy decoding during evaluation

### Phase 4 Incremental Steps

- [ ] Implement scaled dot-product attention.
- [ ] Implement MHA.
- [ ] Implement GQA.
- [ ] Implement sinusoidal positional encoding.
- [ ] Implement RoPE.
- [ ] Implement custom LayerNorm.
- [ ] Implement RMSNorm.
- [ ] Implement the FFN.
- [ ] Implement encoder blocks.
- [ ] Implement decoder blocks.
- [ ] Assemble the baseline encoder-decoder transformer.
- [ ] Run tensor-shape checks for every module.

---

## Phase 5: Implement the BLT-Specific Components for `C5`

### Step 5.1: Define a simplified BLT architecture

- The assignment asks for a **simplified Byte Latent Transformer**, not a full research-scale BLT.
- Design it with three stages:
  - local byte encoder
  - global transformer over patch representations
  - local byte decoder

### Step 5.2: Implement local encoder patch modules

- In `src/models/blt.py`, create a local encoder that:
  - takes raw byte embeddings
  - groups bytes into patches
  - compresses each patch into a latent representation
- Candidate designs:
  - small self-attention inside each patch
  - simple MLP pooling
  - convolution-like projection
- Prefer the simplest design that is easy to explain during viva.

### Step 5.3: Implement the global latent transformer

- Feed patch representations into the same main transformer backbone.
- Ensure this stage reuses as much baseline infrastructure as possible.
- Keep the latent transformer hyperparameters aligned with C1 for fair comparison.

### Step 5.4: Implement the local byte-level decoder

- Decode latent patch outputs back into byte predictions.
- Ensure output can be compared against target bytes directly.
- Support greedy decoding at the byte level for evaluation.

### Step 5.5: Validate the BLT path carefully

- Check:
  - patch formation correctness
  - latent shape consistency
  - decoder target alignment
  - reconstruction output format
- Because `C5` is the most complex configuration, isolate unit tests or sanity scripts for it early.

### Phase 5 Incremental Steps

- [ ] Define the simplified BLT patching rule.
- [ ] Implement the local byte encoder.
- [ ] Implement the global latent transformer path.
- [ ] Implement the local byte decoder.
- [ ] Wire the BLT-specific forward pass.
- [ ] Validate patch shapes and byte alignment.
- [ ] Verify that `C5` differs from `C1` only in the tokenization/BLT path as intended.

---

## Phase 6: Build Training, Validation, and Greedy Decoding

### Step 6.1: Implement the training loop

- In `src/train.py`, build:
  - model initialization from config
  - optimizer creation
  - loss function setup
  - data loading
  - epoch loop
  - validation loop
  - checkpoint saving
- For tokenized models, use cross-entropy with padding ignored.
- For byte-level BLT, use an appropriate token/byte prediction loss.

### Step 6.2: Add teacher forcing during training

- Prepare decoder inputs by shifting targets right.
- Use teacher forcing during training.
- Ensure target labels are aligned with decoder outputs.

### Step 6.3: Implement greedy decoding for evaluation

- The assignment explicitly requires **greedy decoding** for all reported metrics.
- Implement a decoding function that:
  - starts from `<bos>` for tokenized models
  - generates one token at a time
  - stops at `<eos>` or max length
- For `C5`, adapt this to byte-level autoregressive generation if needed.

### Step 6.4: Add logging to WandB

- Log:
  - training loss
  - validation loss
  - bit-level accuracy
  - sequence accuracy
  - Levenshtein distance
  - BLEU
  - ROUGE
  - epoch time
  - GPU memory
- Use informative run names such as:
  - `A1_C1_baseline`
  - `A1_C2_rope`
  - `A1_C3_gqa`
  - `A1_C4_rmsnorm`
  - `A1_C5_blt`

### Phase 6 Incremental Steps

- [ ] Implement one training step.
- [ ] Implement one validation step.
- [ ] Add teacher forcing with shifted decoder inputs.
- [ ] Implement greedy decoding.
- [ ] Add checkpointing.
- [ ] Add per-epoch metric aggregation.
- [ ] Add local logging to files under `outputs/`.
- [ ] Add optional WandB logging.
- [ ] Run a smoke test on a tiny subset of data.

---

## Phase 7: Implement Evaluation Metrics

### Step 7.1: Bit-level accuracy

- Compare predicted and target outputs at the bit level.
- If outputs are tokenized text, convert both prediction and target into a common bit-string or byte/bit representation before computing this metric.

### Step 7.2: Sequence accuracy

- Count a sequence as correct only if the entire prediction exactly matches the target.

### Step 7.3: Levenshtein distance

- Compute edit distance between prediction and target.
- Report average Levenshtein distance over the test set.

### Step 7.4: BLEU and ROUGE

- Compute these only for tokenized configurations (`C1-C4`) as requested.
- Exclude `C5` if the evaluation setup is genuinely token-free and not naturally compatible.
- State this clearly in the report.

### Step 7.5: Runtime and memory benchmarking for `C5`

- Measure and compare against `C1`:
  - per-epoch training time
  - per-batch time if useful
  - peak GPU memory usage
  - final test performance
- This comparison is explicitly required in the assignment.

### Phase 7 Incremental Steps

- [ ] Implement bit-level accuracy.
- [ ] Implement sequence accuracy.
- [ ] Implement Levenshtein distance.
- [ ] Implement BLEU for tokenized runs.
- [ ] Implement ROUGE for tokenized runs.
- [ ] Verify metric behavior on small hand-written examples.
- [ ] Add C1 vs C5 speed and memory comparison hooks.

---

## Phase 8: Run Experiments in a Controlled Order

### Step 8.1: Run a very small smoke test

- Use a tiny subset of data first.
- Confirm:
  - forward pass works
  - backward pass works
  - loss decreases
  - decoding runs without crashing

### Step 8.2: Train and debug the baseline `C1`

- Train `C1` first until it produces meaningful outputs.
- Fix all stability issues before starting any other configuration.
- Save:
  - best checkpoint
  - final checkpoint
  - training curves
  - test metrics

### Step 8.3: Run single-change ablations `C2-C4`

- Train `C2`, `C3`, and `C4` using the same pipeline and hyperparameters as `C1`.
- Ensure each run differs from baseline in exactly one component.
- Record all results in a comparison table.

### Step 8.4: Run the BLT experiment `C5`

- Train `C5` after the tokenized pipeline is stable.
- Log speed and memory carefully because they are central to the analysis.
- Save representative reconstructions for qualitative comparison.

### Step 8.5: Repeat runs if needed for stability

- If results appear noisy, rerun the most important experiments with the same seed or a second seed.
- If multiple seeds are impossible due to time, clearly acknowledge this limitation in the report.

### Phase 8 Incremental Steps

- [ ] Run a tiny smoke test.
- [ ] Train and debug `C1` first.
- [ ] Run `C2` with only the positional change.
- [ ] Run `C3` with only the attention change.
- [ ] Run `C4` with only the normalization change.
- [ ] Run `C5` after the baseline path is stable.
- [ ] Save best checkpoints and final metrics for every run.

---

## Phase 9: Analyze Results

### Step 9.1: Create a consolidated results table

- Build a final table containing:
  - configuration ID
  - positional encoding
  - attention type
  - normalization type
  - tokenization type
  - bit-level accuracy
  - sequence accuracy
  - Levenshtein distance
  - BLEU
  - ROUGE
  - training time
  - peak memory

### Step 9.2: Compare each configuration directly against `C1`

- For `C2`, analyze the effect of RoPE alone.
- For `C3`, analyze the effect of GQA alone.
- For `C4`, analyze the effect of RMSNorm alone.
- For `C5`, analyze the effect of token-free BLT alone.

### Step 9.3: Focus especially on tradeoffs

- Accuracy vs speed
- Accuracy vs memory
- Simplicity vs architectural sophistication
- Standard tokenization vs token-free bytes

These comparisons should directly answer the assignment’s ablation objective.

### Phase 9 Incremental Steps

- [ ] Build a single consolidated results table.
- [ ] Compare each configuration directly against `C1`.
- [ ] Highlight the single changed component in each ablation.
- [ ] Summarize the tradeoffs in accuracy, speed, and memory.
- [ ] Select a few qualitative predictions for discussion.

---

## Phase 10: Prepare Submission Artifacts

### Step 10.1: Save final outputs

- Store in `outputs/`:
  - plots
  - metric tables
  - selected predictions
  - memory/time comparison figures
  - possibly exported CSV summaries

### Step 10.2: Upload checkpoints to Hugging Face

- Upload best or final checkpoints for the main configurations.
- Keep the repository organized and mention:
  - model names
  - config mapping
  - checkpoint purpose

### Step 10.3: Write `README.md`

- Include:
  - environment setup
  - dependency installation
  - dataset preparation
  - training commands
  - evaluation commands
  - explanation of configs `C1-C5`
  - WandB project link
  - Hugging Face model links
  - submission notes

### Step 10.4: Write the report

- Structure the report as:
  - Introduction
  - Dataset and preprocessing
  - Model architecture
  - Description of `C1-C5`
  - Experimental setup
  - Results
  - Ablation analysis
  - BLT comparison with C1
  - Conclusion
- Keep it within the 6-page limit excluding references and links.

### Step 10.5: Verify directory structure exactly

- Ensure the final folder matches the required layout precisely.
- Verify filenames before zipping.

### Step 10.6: Create final ZIP

- Package everything as:
  - `<rollnumber>_assignment1.zip`
- Perform a final dry run by inspecting the ZIP contents before submission.

### Phase 10 Incremental Steps

- [ ] Save final plots and metric summaries.
- [ ] Upload checkpoints to Hugging Face.
- [ ] Finalize `README.md`.
- [ ] Finalize `Report.pdf`.
- [ ] Verify the directory structure exactly matches the assignment.
- [ ] Create the final ZIP with the correct roll number.

---

## 4. Recommended Implementation Order

To minimize rework, follow this coding order:

1. `src/dataset.py`
2. `src/models/attention.py`
3. `src/models/positional.py`
4. `src/models/norm.py`
5. baseline encoder-decoder assembly
6. greedy decoding
7. `src/train.py`
8. `src/utils.py`
9. `src/models/blt.py`
10. configuration switching and experiment scripts
11. plots, README, report

---

## 5. Suggested Validation Checklist

Before full training, verify each of these:

- Data loader returns expected shapes.
- Padding masks are correct.
- Decoder causal mask is correct.
- MHA output shape matches input embedding dimension.
- GQA output shape matches MHA interface.
- RoPE changes only positional handling, not tensor rank expectations.
- RMSNorm can replace LayerNorm without breaking block structure.
- Greedy decoding stops correctly.
- BLEU/ROUGE are computed only where valid.
- `C2-C5` each differ from `C1` in one and only one component.

---

## 6. Common Risks and Mitigation

### Risk 1: Dataset ambiguity

- Mitigation:
  - inspect raw files early
  - document assumptions
  - keep preprocessing modular

### Risk 2: Shape bugs in attention modules

- Mitigation:
  - add assertions
  - test with toy tensors
  - verify masks independently

### Risk 3: Unfair ablation comparisons

- Mitigation:
  - centralize configuration
  - freeze shared hyperparameters
  - log exact settings per run

### Risk 4: BLT becoming too complex

- Mitigation:
  - implement the simplest defensible local encoder/decoder
  - prioritize correctness and explainability over novelty

### Risk 5: Evaluation mismatch

- Mitigation:
  - standardize greedy decoding across all runs
  - compute metrics through one shared evaluation path

### Risk 6: Report becoming descriptive instead of analytical

- Mitigation:
  - always compare each configuration directly to baseline
  - emphasize single-factor impact and tradeoffs

---

## 7. Final Success Criteria

The assignment can be considered complete when all of the following are true:

- All required modules are implemented from scratch.
- `C1-C5` can be launched reproducibly from one training pipeline.
- WandB logs exist for all runs.
- Hugging Face checkpoint links are ready.
- Test metrics are collected using greedy decoding.
- BLT vs baseline speed and memory comparison is documented.
- Report and README are complete.
- Final folder structure matches the assignment exactly.
- Final ZIP is ready for submission.

---

## 8. Immediate Next Actions

The most sensible next steps are:

1. Scaffold the required directory structure and starter files.
2. Implement the dataset pipeline with binary-to-byte conversion.
3. Build the baseline `C1` transformer modules and forward pass.
4. Add training, decoding, and metric utilities.
