"""
src/training.py
---------------
SevenNet model training utilities.

Two modes
---------
train_scratch   : train a SevenNet model from random initialisation
                  on a given dataset (xyz file)
fine_tune       : fine-tune an existing SevenNet checkpoint on an
                  augmented dataset (center xyz + sampled xyz)

Typical usage (notebook)
------------------------
from src.training import train_scratch, fine_tune, evaluate_model

# 1. Scratch training on foundational (center) dataset
ckpt_scratch = train_scratch(
    train_xyz   = "v1_dataset/initial_200fs_500.xyz",
    output_dir  = "results/scratch",
    total_epochs = 50,
)

# 2. Fine-tune for each sampling method
for method, xyz in [("random", "random_selected.xyz"),
                    ("direct", "direct_selected.xyz"),
                    ("lcmd",   "lcmd_selected.xyz")]:
    ckpt_ft = fine_tune(
        base_checkpoint  = ckpt_scratch,
        augment_xyz      = xyz,
        center_xyz       = "v1_dataset/initial_200fs_500.xyz",
        output_dir       = f"results/ft_{method}",
        total_epochs     = 10,
    )

# 3. Evaluate
results = evaluate_model(
    checkpoint = ckpt_ft,
    test_xyz   = "test_selected.xyz",
)
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_loaders(
    data_paths: List[str],
    cutoff: float,
    train_ratio: float = 0.90,
    n_valid: int = 10,
    batch_size: int = 4,
    seed: int = 42,
):
    """
    Build SevenNet graph dataset and DataLoaders.

    Parameters
    ----------
    data_paths  : list of xyz file paths
    cutoff      : graph cutoff radius (Å)
    train_ratio : fraction of data used for training
    n_valid     : number of validation structures (overrides ratio if < remainder)
    batch_size  : batch size for both loaders
    seed        : shuffle seed

    Returns
    -------
    train_loader, valid_loader, dataset
    """
    import torch
    from torch_geometric.loader import DataLoader
    from sevenn.train.graph_dataset import SevenNetGraphDataset

    dataset = SevenNetGraphDataset(cutoff=cutoff, files=data_paths, drop_info=False)
    dataset = dataset.shuffle(seed)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_valid_actual = min(n_valid, n_total - n_train)

    trainset = dataset[:n_train]
    validset = dataset[n_train : n_train + n_valid_actual]

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size)

    print(f"  Dataset: {n_total} total  |  train={n_train}  valid={n_valid_actual}")
    return train_loader, valid_loader, dataset


def _default_model_config(cutoff: float = 5.0) -> Dict:
    """Return a minimal SevenNet model config suitable for fine-tuning demos."""
    import sevenn
    from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
    import sevenn.util as util

    cfg = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
    cfg.update({
        "version": sevenn.__version__,
        "channel": 16,
        "lmax": 2,
        "cutoff": cutoff,
        "num_convolution_layer": 3,
        "is_parity": False,
    })
    cfg.update(util.chemical_species_preprocess([], universal=True))
    return cfg


def _default_train_config(
    lr: float = 0.01,
    total_iters: int = 100,
    force_weight: float = 0.1,
    stress_weight: float = 0.01,
    loss: str = "huber",
    loss_delta: float = 0.01,
    device: str = "cuda",
) -> Dict:
    from sevenn._const import DEFAULT_TRAINING_CONFIG

    cfg = deepcopy(DEFAULT_TRAINING_CONFIG)
    cfg.update({
        "device": device,
        "optimizer": "adam",
        "optim_param": {"lr": lr},
        "scheduler": "linearlr",
        "scheduler_param": {
            "start_factor": 1.0,
            "total_iters": total_iters,
            "end_factor": 0.0001,
        },
        "loss": loss,
        "loss_param": {"delta": loss_delta},
        "force_loss_weight": force_weight,
        "stress_loss_weight": stress_weight,
        "is_ddp": False,
        "error_record": [
            ("Energy", "RMSE"),
            ("Force", "RMSE"),
            ("TotalLoss", "None"),
        ],
    })
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _run_training(
    model,
    train_loader,
    valid_loader,
    train_cfg: Dict,
    total_epochs: int,
    output_dir: str,
    combined_cfg: Dict,
    checkpoint_name: str = "checkpoint_best.pth",
) -> str:
    """
    Inner training loop shared by train_scratch and fine_tune.

    Returns path to best checkpoint.
    """
    import torch
    from tqdm import tqdm
    from sevenn.error_recorder import ErrorRecorder
    from sevenn.train.trainer import Trainer
    from sevenn.logger import Logger

    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, checkpoint_name)

    trainer = Trainer.from_config(model, train_cfg)
    train_rec = ErrorRecorder.from_config(train_cfg)
    valid_rec = deepcopy(train_rec)

    valid_best = float("inf")

    with Logger(
        filename=os.path.join(output_dir, "train.log"), screen=True
    ) as logger:
        logger.greeting()
        for epoch in range(total_epochs):
            logger.timer_start("epoch")
            trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_rec)
            trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_rec)
            trainer.scheduler_step()

            t_err = train_rec.epoch_forward()
            v_err = valid_rec.epoch_forward()

            logger.bar()
            logger.writeline(
                f"Epoch {epoch+1}/{total_epochs}  LR: {trainer.get_lr():.6f}"
            )
            logger.write_full_table([t_err, v_err], ["Train", "Valid"])
            logger.timer_end("epoch", message=f"Epoch {epoch+1} elapsed")

            if v_err["TotalLoss"] < valid_best:
                valid_best = v_err["TotalLoss"]
                trainer.write_checkpoint(best_path, config=combined_cfg, epoch=epoch)

    # Always save final checkpoint too
    final_path = os.path.join(output_dir, "checkpoint_final.pth")
    trainer.write_checkpoint(final_path, config=combined_cfg, epoch=total_epochs)
    torch.cuda.empty_cache()

    print(f"\n  Best checkpoint → {best_path}  (TotalLoss={valid_best:.6f})")
    return best_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_scratch(
    train_xyz: str,
    output_dir: str,
    cutoff: float = 5.0,
    total_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.01,
    train_ratio: float = 0.95,
    n_valid: int = 10,
    force_weight: float = 0.1,
    stress_weight: float = 0.01,
    device: str = "cuda",
) -> str:
    """
    Train a SevenNet model from scratch on a foundational dataset.

    Parameters
    ----------
    train_xyz     : path to training ExtXYZ file
    output_dir    : directory to save checkpoints and logs
    cutoff        : graph cutoff radius in Å
    total_epochs  : number of training epochs
    batch_size    : batch size
    lr            : initial learning rate
    train_ratio   : fraction of data for training (rest → validation)
    n_valid       : max validation set size
    force_weight  : force loss weight
    stress_weight : stress loss weight
    device        : 'cuda' or 'cpu'

    Returns
    -------
    path to best checkpoint (str)
    """
    import sevenn.util as util
    from sevenn.model_build import build_E3_equivariant_model

    print("=" * 60)
    print("SevenNet — Scratch Training")
    print(f"  data    : {train_xyz}")
    print(f"  output  : {output_dir}")
    print(f"  epochs  : {total_epochs}")
    print("=" * 60)

    # Build dataset
    train_loader, valid_loader, dataset = _make_loaders(
        [train_xyz], cutoff, train_ratio, n_valid, batch_size
    )

    # Model config
    model_cfg = _default_model_config(cutoff)
    model_cfg.update({
        "shift": dataset.per_atom_energy_mean,
        "scale": dataset.force_rms,
        "conv_denominator": dataset.avg_num_neigh,
    })
    model = build_E3_equivariant_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Train config
    train_cfg = _default_train_config(
        lr=lr,
        total_iters=total_epochs,
        force_weight=force_weight,
        stress_weight=stress_weight,
        device=device,
    )
    combined_cfg = {**model_cfg, **train_cfg}

    return _run_training(
        model, train_loader, valid_loader,
        train_cfg, total_epochs, output_dir,
        combined_cfg, checkpoint_name="checkpoint_best.pth",
    )


def fine_tune(
    base_checkpoint: str,
    augment_xyz: str,
    output_dir: str,
    center_xyz: Optional[str] = None,
    total_epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.004,
    train_ratio: float = 0.90,
    n_valid: int = 10,
    force_weight: float = 0.1,
    stress_weight: float = 0.01,
    device: str = "cuda",
) -> str:
    """
    Fine-tune a pre-trained SevenNet checkpoint on an augmented dataset.

    The training data is the union of augment_xyz and (optionally) center_xyz.

    Parameters
    ----------
    base_checkpoint : path to pre-trained SevenNet checkpoint (.pth)
    augment_xyz     : path to sampled / augmented ExtXYZ file
    output_dir      : directory to save fine-tuned checkpoints and logs
    center_xyz      : optional path to foundational dataset xyz to include
    total_epochs    : number of fine-tuning epochs
    batch_size      : batch size
    lr              : initial learning rate for fine-tuning
    train_ratio     : fraction of data for training
    n_valid         : max validation set size
    force_weight    : force loss weight
    stress_weight   : stress loss weight
    device          : 'cuda' or 'cpu'

    Returns
    -------
    path to best fine-tuned checkpoint (str)
    """
    import sevenn.util as util

    print("=" * 60)
    print("SevenNet — Fine-Tuning")
    print(f"  base    : {base_checkpoint}")
    print(f"  augment : {augment_xyz}")
    if center_xyz:
        print(f"  center  : {center_xyz}")
    print(f"  output  : {output_dir}")
    print(f"  epochs  : {total_epochs}")
    print("=" * 60)

    model, config = util.model_from_checkpoint(base_checkpoint)
    config.update({"is_ddp": False})

    data_paths = [augment_xyz]
    if center_xyz:
        data_paths.append(center_xyz)

    train_loader, valid_loader, _ = _make_loaders(
        data_paths, config["cutoff"], train_ratio, n_valid, batch_size
    )

    train_cfg = _default_train_config(
        lr=lr,
        total_iters=total_epochs,
        force_weight=force_weight,
        stress_weight=stress_weight,
        device=device,
    )
    combined_cfg = {**config, **train_cfg}

    return _run_training(
        model, train_loader, valid_loader,
        train_cfg, total_epochs, output_dir,
        combined_cfg, checkpoint_name="checkpoint_best.pth",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    checkpoint: str,
    test_xyz: str,
    device: str = "cuda",
    max_structures: Optional[int] = None,
) -> Dict:
    """
    Run inference on a test set and compute energy / force RMSE and MAE.

    Parameters
    ----------
    checkpoint      : path to SevenNet checkpoint
    test_xyz        : path to test ExtXYZ file
    device          : 'cuda' or 'cpu'
    max_structures  : if set, only evaluate first N structures

    Returns
    -------
    dict with keys:
        energy_rmse, energy_mae  (eV/atom)
        force_rmse, force_mae    (eV/Å)
        dft_energy, mlip_energy  (lists, eV/atom)
        dft_forces, mlip_forces  (flat lists, eV/Å)
    """
    import gc
    import torch
    from ase.io import read as ase_read
    from tqdm import tqdm as _tqdm
    from sevenn.calculator import SevenNetCalculator

    print(f"Evaluating: {checkpoint}")
    traj = ase_read(test_xyz, index=":")
    if max_structures is not None:
        traj = traj[:max_structures]
    print(f"  {len(traj)} test structures")

    dft_e, dft_f = [], []
    mlip_e, mlip_f = [], []

    for atoms in _tqdm(traj, desc="DFT labels"):
        dft_e.append(atoms.get_potential_energy() / len(atoms))
        dft_f.extend(atoms.get_forces().flatten().tolist())

    calc = SevenNetCalculator(checkpoint)
    for atoms in _tqdm(traj, desc="MLIP inference"):
        atoms.calc = calc
        mlip_e.append(atoms.get_potential_energy() / len(atoms))
        mlip_f.extend(atoms.get_forces().flatten().tolist())
        atoms.calc = None

    del calc
    gc.collect()
    torch.cuda.empty_cache()

    dft_e = np.array(dft_e)
    mlip_e_arr = np.array(mlip_e)
    dft_f = np.array(dft_f)
    mlip_f_arr = np.array(mlip_f)

    e_rmse = float(np.sqrt(np.mean((dft_e - mlip_e_arr) ** 2)))
    e_mae = float(np.mean(np.abs(dft_e - mlip_e_arr)))
    f_rmse = float(np.sqrt(np.mean((dft_f - mlip_f_arr) ** 2)))
    f_mae = float(np.mean(np.abs(dft_f - mlip_f_arr)))

    print(f"  Energy RMSE: {e_rmse:.4f} eV/atom  MAE: {e_mae:.4f} eV/atom")
    print(f"  Force  RMSE: {f_rmse:.4f} eV/Å     MAE: {f_mae:.4f} eV/Å")

    return {
        "energy_rmse": e_rmse,
        "energy_mae": e_mae,
        "force_rmse": f_rmse,
        "force_mae": f_mae,
        "dft_energy": dft_e.tolist(),
        "mlip_energy": mlip_e_arr.tolist(),
        "dft_forces": dft_f.tolist(),
        "mlip_forces": mlip_f_arr.tolist(),
    }
