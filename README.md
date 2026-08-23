# ANLP Assignment 1

## Dataset Contract

The extracted dataset lives in `Dataset_A1/` and contains two line-aligned files:

- `brown_cipher.txt`
- `brown_plain.txt`

Each line index `k` defines one source-target pair.

- Source: a binary string containing only `0` and `1`
- Target: the corresponding plaintext English string

The source length is consistently `8x` the target length, so the working assumption for this implementation is:

- each plaintext character corresponds to one byte
- each source example should be compacted from bits into byte-sized units before standard tokenization

## Phase 1 Decisions

- Split strategy: deterministic `80/10/10` train/validation/test split using a fixed seed.
- Sequence handling strategy: keep full sequences, but use length-aware bucketing or dynamic batching to limit padding waste and memory pressure.
- Raw dataset policy: the original files in `Dataset_A1/` remain unchanged, and all derived artifacts are generated reproducibly from them.

## Phase 1 Artifacts

- Dataset summary: `outputs/dataset_summary.json`
- Starter modules:
  - `src/dataset.py`
  - `src/train.py`
  - `src/utils.py`
  - `src/models/attention.py`
  - `src/models/positional.py`
  - `src/models/norm.py`
  - `src/models/blt.py`

## Next Implementation Focus

Phase 2 and Phase 3 will build:

- shared experiment configuration for `C1-C5`
- preprocessing/tokenization pipeline
- batching and masking helpers
- the first runnable baseline data path
