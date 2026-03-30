# When Adaptation Fails

Official implementation for:

**When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in Vision-Language Prompt Learning**  
ICME 2026

## Overview

This repository contains the code for our diagnostic study of adaptive gating in CLIP-based prompt learning.

We investigate why prompt-level adaptive gating often fails under frozen few-shot prompt learning, and identify two recurring failure modes:

- **Gradient magnitude imbalance**: gate parameters receive gradients that are 2–3 orders of magnitude smaller than prompt parameters
- **Gate degradation**: gate activations converge to near-constant values and become functionally inert

Our experiments are conducted mainly on top of **MaPLe**, with additional adaptations to **CoOp** and **CoCoOp** for cross-model validation.

## Main Findings

- Adaptive gating often fails to provide consistent gains over fixed prompts in this regime
- Performance improvements on small datasets can often be explained by parameter-count buffering or regularization effects rather than genuine adaptive behavior
- Failure patterns recur across multiple CLIP-based prompt learning frameworks

## Repository Structure

```text
.
├── clip/                  # CLIP-related source code
├── configs/               # Dataset and trainer configs
├── datasets/              # Dataset definitions / dataset utilities
├── dassl/                 # Dassl framework code
├── diagnostics/           # Diagnostic analysis scripts
├── model_statistics/      # Model statistics / analysis utilities
├── scripts/               # Train / test shell scripts
├── trainers/              # Trainer implementations, including our variants
├── train.py               # Main training entry point
├── requirements.txt
└── README.md
````

This repository is built on top of prior codebases including MaPLe and Dassl, with our modifications for adaptive gating and diagnostic analysis.

## Environment

This code was developed in a research codebase built around MaPLe / Dassl-style training.

Install dependencies from:

```bash
pip install -r requirements.txt
```

Depending on your local environment, minor package/version adjustments may be needed.

## Datasets

Experiments in the paper mainly use:

* ImageNet
* Caltech101
* EuroSAT

Please prepare datasets under your local data root and update the `DATA` path in the scripts accordingly.

## Main Models in the Paper

The paper mainly discusses the following models:

* **MaPLe**
* **BiMaPLe**
* **AdaptiveBiDirMaPLe** (our adaptive diagnostic testbed)
* **CoOp**
* **CoCoOp**
* Gated variants of **CoOp** and **CoCoOp**

The main diagnostic results in the paper are centered on **AdaptiveBiDirMaPLe**, with cross-model validation on **CoOp** and **CoCoOp**.

## Quick Start

### 1. AdaptiveBiDirMaPLe

The main adaptive model used in our paper can be trained with scripts under:

```text
scripts/adaptive_bidir_maple/
```

A representative command is:

```bash
python train.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer AdaptiveBiDirMaPLe \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/AdaptiveBiDirMaPLe/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple.yaml \
  --output-dir ${DIR} \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES base
```

### 2. MaPLe / BiMaPLe Baselines

Baseline models can be run using the corresponding scripts/configs in `scripts/` and `configs/trainers/`.

These models serve as the primary non-adaptive references in the paper.

### 3. CoOp / CoCoOp and Their Gated Variants

Cross-model validation is implemented through the corresponding trainer/config entries for:

* CoOp
* CoCoOp
* gated CoOp
* gated CoCoOp

Please check the relevant scripts under `scripts/` and trainer implementations under `trainers/`.

## Representative Hyperparameters

For the main AdaptiveBiDirMaPLe experiments, representative settings include:

* backbone: **ViT-B/16**
* optimizer: **SGD**
* base learning rate: **0.0025**
* batch size: **4**
* max epoch: **5**
* scheduler: **cosine**
* number of shots: **16**

Adaptive gating related settings are defined in the corresponding trainer config files.

## Diagnostics

This repository focuses not only on training, but also on **diagnosing failure modes**.

The main diagnostics include:

* gradient norm comparison between prompt parameters and gate parameters
* gate activation / effective prompt length tracking
* cross-model validation across MaPLe, CoOp, and CoCoOp
* additional analyses for small-data behavior such as EuroSAT

Relevant code is mainly located in:

```text
diagnostics/
model_statistics/
```

## Reproducing the Paper

To reproduce the main conclusions of the paper, we recommend the following order:

1. Train or evaluate the **AdaptiveBiDirMaPLe** model
2. Compare against **MaPLe** / **BiMaPLe** baselines
3. Run cross-model validation with **CoOp** and **CoCoOp**
4. Inspect diagnostics related to:

   * gradient magnitude imbalance
   * gate collapse / near-constant activation
   * limited performance gains of adaptive gating

This repository is intended primarily to reproduce the **diagnostic findings and qualitative conclusions** in the paper. Exact numerical results may vary slightly across environments.

## Notes

* This codebase contains source components inherited from upstream frameworks and project dependencies used during research development.
* The repository is released in a practical research form rather than as a heavily refactored software package.
* The focus is on transparency of the experimental pipeline and diagnostic analysis.
* *This repository was reconstructed from an earlier research codebase. Some implementation details reflect the exploratory nature of the original experiments. We have done our best to make it usable and consistent with the paper. If it runs on the first try, please enjoy the moment.*

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{fang2026adaptation,
  title={When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in Vision-Language Prompt Learning},
  author={Fang, Yunxuan and Zhang, Ziwei and Wang, Xinhe},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo},
  year={2026}
  note={Accepted, to appear}
}
```

## Acknowledgements

This repository builds upon prior open-source efforts in CLIP prompt learning, especially MaPLe and Dassl.
