# Active Learning Pipeline for MLIP Training — Example (v1)

## Overview

This repository provides a **minimal working example** of a coupled sampling–training
framework for machine-learning interatomic potential (MLIP) development.

Using a small dataset, the example demonstrates how different active learning
sampling strategies affect model quality, making it straightforward to compare
methods side-by-side. Specifically, the pipeline shows how:

- Structures are selected from an AIMD candidate pool using three different
  sampling strategies (Random, DIRECT, LCMD)
- A SevenNet model is trained from scratch on a foundational dataset
- The scratch model is fine-tuned separately for each sampling method
- The resulting models can be evaluated and compared quantitatively

Because the sampling and training steps are tightly coupled — the choice of
sampling strategy directly determines what the fine-tuned model sees — this
framework illustrates the effect of data selection on model accuracy and
generalisation.

> **Note:** This is an example repository using a reduced dataset.
> All data sizes and epoch counts are intentionally small for reproducibility
> and fast execution. For production-scale runs, increase dataset sizes,
> training epochs, and adjust hyperparameters accordingly.

---

## Acknowledgements

This pipeline builds on the following open-source packages:

- **[maml](https://github.com/materialsvirtuallab/maml)** — M3GNet structural
  feature extraction (`maml.describers.M3GNetStructure`)
- **[SevenNet](https://github.com/MDIL-SNU/SevenNet)** — E3-equivariant graph
  neural network potential training and inference

Code in this repository was written with the assistance of
**[Claude](https://claude.ai)** (Anthropic).

---

## Pipeline

```
01_sampling_and_dft_prep.ipynb   ← feature extraction → sampling → VASP inputs
        │
        │  (run VASP DFT calculations)
        ▼
02_training_and_evaluation.ipynb ← scratch training → fine-tuning → evaluation
```

| # | Step | Module |
|---|------|--------|
| 1 | M3GNet feature extraction | `src/features.py` |
| 2 | PCA dimension reduction (Kaiser's rule) | `src/features.py` |
| 3 | Sampling — Random / DIRECT / LCMD | `src/sampling.py` |
| 4 | VASP input generation (MatPESStaticSet) | `src/vasp_inputs.py` |
| 5 | SevenNet scratch training | `src/training.py` |
| 6 | SevenNet fine-tuning (per sampling method) | `src/training.py` |
| 7 | Visualisation | `src/plotting.py` |

---

## Sampling methods

| Method | Algorithm |
|--------|-----------|
| **Random** | Uniform random selection (baseline) |
| **DIRECT** | Birch clustering; one centroid-closest representative per cluster. Threshold auto-tuned via bisection to match target count. |
| **LCMD** | Greedy farthest-point (sequential maximin). Seeded from the foundational dataset via KDTree; at each step the candidate with the largest minimum-distance to the current selected set is chosen. Distance updates use BLAS gemv. |

---

## Directory layout

```
data/
└── 1_example_data/
    ├── v1_dataset/
    │   ├── initial_200fs_500.xyz           ← center (foundational) structures
    │   ├── augmented_10fs_10000.xyz         ← AIMD candidate pool
    │   └── pca_model/
    │       ├── m3gnet_pca_pca_model.json   ← pre-trained PCA model
    │       ├── initial_200fs_500_reduced.h5
    │       └── augmented_10fs_10000_reduced.h5
    └── v1_models/
        ├── sampling/                        ← selected XYZ + JSON per method
        ├── training/
        │   ├── scratch/                     ← foundational scratch model
        │   ├── ft_random/
        │   ├── ft_direct/
        │   └── ft_lcmd/
        └── results/                         ← evaluation figures
```

Pre-computed feature and PCA files are included so Steps 1–2 can be skipped.

---

## Installation

```bash
pip install -r requirements.txt
```

> A licensed VASP POTCAR library is required for Step 4 (DFT input generation).
> Set `POTCAR_ROOT` in notebook 01 to point to your `potpaw_PBE.54` directory.

---

## Quick start

```bash
# 1. Sampling and DFT input preparation
jupyter notebook 01_sampling_and_dft_prep.ipynb

# 2. Submit VASP calculations and collect outputs into ExtXYZ format

# 3. Training and evaluation
jupyter notebook 02_training_and_evaluation.ipynb
```

---

## src module reference

| File | Key functions |
|------|---------------|
| `src/features.py` | `extract_features`, `train_pca`, `apply_pca` |
| `src/sampling.py` | `load_reduced_h5`, `random_sampling`, `direct_sampling`, `lcmd_sampling`, `save_selection` |
| `src/vasp_inputs.py` | `generate_vasp_inputs`, `read_lammps_atoms` |
| `src/training.py` | `train_scratch`, `fine_tune`, `evaluate_model` |
| `src/plotting.py` | `plot_pca_space`, `plot_parity`, `plot_cluster_analysis`, `plot_force_rmse_by_bin`, `plot_violin`, `plot_pca_variance` |
