"""
src/features.py
---------------
M3GNet structural feature extraction and PCA dimension reduction.

Key functions
-------------
extract_features(xyz_file, output_h5, batch_size, n_jobs)
    ExtXYZ → M3GNet features → HDF5

train_pca(h5_files, output_dir, model_name)
    Multiple H5 files → PCA model (Kaiser's rule) → JSON + reduced H5s

apply_pca(h5_file, pca_model_json, output_h5)
    Apply saved PCA model to new H5 feature file
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _process_batch(batch_data: Tuple) -> List[Dict]:
    """
    Worker function for multiprocessing.
    Extract M3GNet features for a batch of ASE Atoms objects.

    Parameters
    ----------
    batch_data : (structures, indices, batch_id)

    Returns
    -------
    list of dicts with keys: structure_index, features, num_atoms
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    from maml.describers import M3GNetStructure

    batch_structures, batch_indices, batch_id = batch_data
    try:
        encoder = M3GNetStructure()
        feats = encoder.transform(batch_structures)
        results = [
            {
                "structure_index": int(idx),
                "features": feat,
                "num_atoms": len(atoms),
            }
            for idx, atoms, feat in zip(batch_indices, batch_structures, feats)
        ]
        print(f"  Batch {batch_id}: {len(results)} structures done")
        return results
    except Exception as exc:
        print(f"  Batch {batch_id} failed: {exc}")
        return []


def extract_features(
    xyz_file: str,
    output_h5: str,
    batch_size: int = 32,
    n_jobs: int = 4,
) -> str:
    """
    Extract M3GNet structural features from an ExtXYZ file and save to HDF5.

    Parameters
    ----------
    xyz_file   : path to input ExtXYZ file
    output_h5  : path to output HDF5 file
    batch_size : number of structures per batch
    n_jobs     : number of parallel processes (1 = sequential)

    Returns
    -------
    output_h5 path (str)
    """
    from ase.io import read

    os.makedirs(Path(output_h5).parent, exist_ok=True)

    print(f"Loading structures from: {xyz_file}")
    structures = read(xyz_file, index=":")
    n = len(structures)
    print(f"  {n} structures loaded")

    # Build batches
    batches = [
        (structures[i : i + batch_size], list(range(i, min(i + batch_size, n))), b + 1)
        for b, i in enumerate(range(0, n, batch_size))
    ]
    print(f"  {len(batches)} batches of up to {batch_size}")

    # Extract
    all_results: List[Dict] = []
    if n_jobs == 1:
        for batch in tqdm(batches, desc="Extracting features"):
            all_results.extend(_process_batch(batch))
    else:
        with mp.Pool(processes=n_jobs) as pool:
            for batch_res in tqdm(
                pool.imap(_process_batch, batches),
                total=len(batches),
                desc="Extracting features",
            ):
                all_results.extend(batch_res)

    all_results.sort(key=lambda x: x["structure_index"])

    if not all_results:
        raise RuntimeError("No features were extracted.")

    feat_dim = all_results[0]["features"].shape[0]
    feature_matrix = np.vstack([r["features"] for r in all_results])
    indices = np.array([r["structure_index"] for r in all_results])
    num_atoms = np.array([r["num_atoms"] for r in all_results])

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("features", data=feature_matrix, compression="gzip", compression_opts=9)
        meta = f.create_group("metadata")
        meta.attrs["num_structures"] = len(all_results)
        meta.attrs["feature_dim"] = feat_dim
        meta.attrs["source_file"] = os.path.basename(xyz_file)
        meta.attrs["timestamp"] = datetime.now().isoformat()
        meta.create_dataset("structure_indices", data=indices)
        meta.create_dataset("num_atoms", data=num_atoms)

    print(f"  Saved {len(all_results)} features → {output_h5}  shape={feature_matrix.shape}")
    return output_h5


# ──────────────────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────────────────

def _load_h5_features(h5_file: str) -> Tuple[np.ndarray, Dict]:
    """Load feature matrix and metadata from HDF5."""
    with h5py.File(h5_file, "r") as f:
        feats = f["features"][:]
        meta: Dict = {}
        if "metadata" in f:
            for k, v in f["metadata"].attrs.items():
                meta[k] = v
            if "structure_indices" in f["metadata"]:
                meta["structure_indices"] = f["metadata"]["structure_indices"][:]
            if "num_atoms" in f["metadata"]:
                meta["num_atoms"] = f["metadata"]["num_atoms"][:]
    return feats, meta


def train_pca(
    h5_files: List[str],
    output_dir: str,
    model_name: str = "m3gnet_pca",
    save_reduced: bool = True,
    reduced_output_dir: Optional[str] = None,
) -> str:
    """
    Train a PCA model on M3GNet features using Kaiser's rule (eigenvalue > 1).

    Parameters
    ----------
    h5_files          : list of HDF5 feature files to combine for fitting
    output_dir        : directory to save PCA model JSON and summary
    model_name        : base name for output files
    save_reduced      : if True, also save reduced feature H5 for each input file
    reduced_output_dir: where to save reduced H5s (default: same as output_dir)

    Returns
    -------
    path to saved PCA model JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    red_dir = reduced_output_dir or output_dir
    os.makedirs(red_dir, exist_ok=True)

    # ── Load and concatenate ────────────────────────────────────────────────
    all_feats, all_meta = [], []
    for path in h5_files:
        feats, meta = _load_h5_features(path)
        all_feats.append(feats)
        all_meta.append(meta)
        print(f"  Loaded {path}  shape={feats.shape}")

    feat_dims = [f.shape[1] for f in all_feats]
    if len(set(feat_dims)) > 1:
        raise ValueError(f"Feature dimension mismatch: {dict(zip(h5_files, feat_dims))}")

    combined = np.concatenate(all_feats, axis=0)
    print(f"\nCombined: {combined.shape[0]} samples × {combined.shape[1]} features")

    # ── Normalise ───────────────────────────────────────────────────────────
    print("Normalising features …")
    scaler = StandardScaler()
    X = scaler.fit_transform(combined)

    # ── Full PCA → Kaiser's rule ────────────────────────────────────────────
    print("Running full PCA to determine Kaiser components …")
    pca_full = PCA()
    pca_full.fit(X)
    n_kaiser = int(np.sum(pca_full.explained_variance_ > 1))
    print(f"Kaiser's rule → {n_kaiser} components  "
          f"(eigenvalue range: {pca_full.explained_variance_[0]:.3f} – "
          f"{pca_full.explained_variance_[-1]:.6f})")

    del pca_full; gc.collect()

    # ── Refit with selected components ────────────────────────────────────
    pca = PCA(n_components=n_kaiser)
    reduced_combined = pca.fit_transform(X)
    del X, combined; gc.collect()

    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(f"Cumulative explained variance: {cumvar:.4f}")

    # ── Save model JSON ─────────────────────────────────────────────────────
    model_info = {
        "n_components": n_kaiser,
        "original_feature_dim": int(scaler.mean_.shape[0]),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "kaiser_threshold": 1.0,
        "source_files": [os.path.basename(p) for p in h5_files],
        "timestamp": datetime.now().isoformat(),
    }
    model_file = os.path.join(output_dir, f"{model_name}_pca_model.json")
    with open(model_file, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"PCA model saved → {model_file}")

    # ── Save reduced H5 for each input file ────────────────────────────────
    if save_reduced:
        offset = 0
        for path, feats, meta in zip(h5_files, all_feats, all_meta):
            n = len(feats)
            reduced = reduced_combined[offset : offset + n]
            offset += n
            base = Path(path).stem
            out = os.path.join(red_dir, f"{base}_reduced.h5")
            _save_reduced_h5(reduced, meta, out, os.path.basename(model_file))
            print(f"  Reduced features → {out}")

    return model_file


def apply_pca(
    h5_file: str,
    pca_model_json: str,
    output_h5: str,
) -> str:
    """
    Apply a pre-trained PCA model to a new feature HDF5 file.

    Parameters
    ----------
    h5_file        : input HDF5 with 'features' dataset
    pca_model_json : path to PCA model JSON (from train_pca)
    output_h5      : output path for reduced features

    Returns
    -------
    output_h5 path (str)
    """
    with open(pca_model_json) as f:
        info = json.load(f)

    scaler_mean = np.array(info["scaler_mean"])
    scaler_scale = np.array(info["scaler_scale"])
    components = np.array(info["components"])
    pca_mean = np.array(info["mean"])

    feats, meta = _load_h5_features(h5_file)
    X = (feats - scaler_mean) / scaler_scale
    reduced = (X - pca_mean) @ components.T

    os.makedirs(Path(output_h5).parent, exist_ok=True)
    _save_reduced_h5(reduced, meta, output_h5, os.path.basename(pca_model_json))
    print(f"Applied PCA → {output_h5}  shape={reduced.shape}")
    return output_h5


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_reduced_h5(
    reduced: np.ndarray,
    meta: Dict,
    output_path: str,
    pca_model_name: str,
) -> None:
    """Save reduced features + metadata to HDF5."""
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("reduced_features", data=reduced, compression="gzip", compression_opts=9)
        f.attrs["pca_model_file"] = pca_model_name
        f.attrs["timestamp"] = datetime.now().isoformat()
        if meta:
            mg = f.create_group("metadata")
            for k, v in meta.items():
                if k in ("structure_indices", "num_atoms"):
                    mg.create_dataset(k, data=v)
                else:
                    try:
                        mg.attrs[k] = v
                    except Exception:
                        pass
