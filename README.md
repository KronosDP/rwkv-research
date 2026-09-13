# RWKV-7 Research: Formal Language Recognition & Custom CUDA Kernel

A from-scratch PyTorch implementation of the **RWKV-7** recurrent architecture, built to study how RWKV's linear-attention-style recurrence compares to RNNs, LSTMs, GRUs, and Transformers on formal language recognition — plus a **hand-written, fused CUDA kernel** for the WKV recurrence itself.

## Why formal languages?

Instead of benchmarking on natural-language corpora (expensive, and confounds architecture with data scale), this project tests sequence models on **synthetic regular languages** with precisely known structure:

| Language | Definition |
|---|---|
| `L1` | `(ab)^n`, n ≥ 0 |
| `L2` | strings over `{a,b}` with an even count of `a` |
| `L3` | strings over `{a,b,c}` containing `abbccc` as a substring |
| `L4` | `L1 ∪ L3` |

`dataset_generator.py` generates large labeled datasets per language, including "near-miss" negatives (edit distance 1-3 from a positive example) so the classifier has to learn the actual rule rather than superficial statistics. Models are trained on short strings and evaluated on longer held-out lengths to test length generalization — a key open question for recurrent/linear-attention architectures.

## What's implemented

- **`rwkv_model.py`** — `RWKV7_Model_Classifier`: a from-scratch RWKV-7 stack (time-mix + channel-mix blocks), implementing the paper's LoRA-parameterized data-dependent decay, in-context learning rate, value-residual gating, and output gating (eqs. 4, 8, 11 & 14 of the RWKV-7 formulation), plus RMSNorm and a ReLU²/SwiGLU-style FFN (see commit history for the architecture iteration from SwiGLU → linear projection + ReLU²).
- **`wkv_kernel.cu` + `setup.py`** — a custom fused CUDA kernel for the WKV recurrence, exposed to PyTorch via `torch.utils.cpp_extension`, with a hand-written forward **and** backward pass (custom `autograd.Function`) so gradients flow through the recurrence without unrolling it in Python. Falls back to a pure-PyTorch path automatically if the compiled extension isn't available.
- **`other-models/`** — matched-capacity baselines (`rnn.py`, `lstm.py`, `gru.py`, `transformer.py`) trained on the same tasks for apples-to-apples comparison.
- **`train_rwkv.py`** — the experiment runner: mixed-precision training, automatic batch-size backoff on OOM, early stopping (with a minimum-epoch floor), and full [Weights & Biases](https://wandb.ai/) logging of metrics, configs, and model artifacts.

## Status

Active research/experimentation — architecture iterations (SwiGLU → linear+ReLU², d_model/learning-rate sweeps) are tracked in the commit history rather than a fixed release. Results are logged to W&B rather than committed here.

## Setup

```bash
# 1. Create the environment (Windows/CUDA, see nvidia-regular-language-experiment.yml for exact pins)
conda env create -f nvidia-regular-language-experiment.yml
conda activate nvidia-regular-language-experiment

# 2. Build the custom CUDA kernel (requires a CUDA toolkit matching your torch build)
python setup.py install

# 3. Generate a dataset
python dataset_generator.py

# 4. Run an experiment (edit main() in train_rwkv.py to pick languages / hyperparameters)
python train_rwkv.py
```

If `custom_wkv_kernel` isn't compiled, `rwkv_model.py` automatically falls back to a pure-PyTorch implementation of the recurrence, so the model still runs (slower) without the CUDA build.

## Tech stack

Python · PyTorch · CUDA / C++ extensions · Weights & Biases · scikit-learn (metrics) · pandas/numpy
