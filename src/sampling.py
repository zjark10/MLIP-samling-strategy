"""
src/sampling.py
---------------
Structure sampling from PCA-reduced feature space.

Three methods
-------------
random_sampling   : uniform random selection
direct_sampling   : Birch clustering, one representative per cluster
lcmd_sampling     : greedy farthest-point sampling seeded from center features
                    (a.k.a. "LCMD — Largest min-dist" or sequential maximin)

Algorithm notes
---------------
DIRECT  — Birch-clusters the candidate set, picks the centroid-closest
          member of each cluster. Threshold is auto-tuned via bisection
          to match the target count.

LCMD    — Pure greedy farthest-point (no clustering step):
          1. Compute min-distance of every candidate to the center set (KDTree).
          2. Greedily pick the candidate with the largest min-distance.
          3. Update min-distance array using BLAS gemv:
             dist²(x, y_new) = ‖x‖² + ‖y_new‖² − 2 x·y_new
          Equivalent to the lcmd_greedy_fast algorithm in the reference
          implementation (4_lcmd_fast_version_save.py).

All methods accept arbitrary center / candidate H5 files.

Typical usage
-------------
from src.sampling import load_reduced_h5, random_sampling, direct_sampling, lcmd_sampling, save_selection

center_feats, center_ids = load_reduced_h5("initial_reduced.h5")
cand_feats,   cand_ids   = load_reduced_h5("augmented_reduced.h5")

idx = lcmd_sampling(cand_feats, center_feats, n_samples=100)
save_selection(idx, cand_ids, "lcmd_selected.json",
               xyz_src="augmented.xyz", xyz_out="lcmd_selected.xyz")
"""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import Birch
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_reduced_h5(h5_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load PCA-reduced features and structure IDs from an HDF5 file.

    Parameters
    ----------
    h5_path : path to HDF5 file with 'reduced_features' dataset

    Returns
    -------
    feats : (N, D) float64 array
    ids   : list of N structure ID strings
    """
    with h5py.File(h5_path, "r") as f:
        if "reduced_features" not in f:
            raise KeyError(f"'reduced_features' not found in {h5_path}")
        feats = f["reduced_features"][:].astype(np.float64)
        if "structure_ids" in f:
            ids = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f["structure_ids"][:]
            ]
        else:
            base = os.path.splitext(os.path.basename(h5_path))[0]
            ids = [f"{base}_{i}" for i in range(len(feats))]
    return feats, ids


def read_xyz_frames(xyz_path: str) -> List[str]:
    """
    Read a multi-frame ExtXYZ file and return each frame as a raw string.

    Parameters
    ----------
    xyz_path : path to ExtXYZ file

    Returns
    -------
    frames : list of raw frame strings (one per structure)
    """
    frames: List[str] = []
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        try:
            n_atoms = int(stripped)
        except ValueError:
            raise ValueError(
                f"XYZ parse error at line {i+1} of {xyz_path}: "
                f"expected atom count, got {stripped!r}"
            )
        block = lines[i : i + n_atoms + 2]
        if len(block) < n_atoms + 2:
            raise ValueError(
                f"Incomplete frame {len(frames)+1} in {xyz_path}"
            )
        frames.append("".join(block))
        i += n_atoms + 2
    print(f"  Read {len(frames)} frames from {xyz_path}")
    return frames


def save_selection(
    selected_indices: List[int],
    cand_ids: List[str],
    json_out: str,
    method: str = "unknown",
    xyz_src: Optional[str] = None,
    xyz_out: Optional[str] = None,
    extra_info: Optional[Dict] = None,
) -> str:
    """
    Save selected structure indices to JSON (and optionally to XYZ).

    Parameters
    ----------
    selected_indices : list of integer indices into cand_ids / xyz_src
    cand_ids         : list of all candidate structure ID strings
    json_out         : output JSON path
    method           : sampling method name (for metadata)
    xyz_src          : if provided, source ExtXYZ file to extract frames from
    xyz_out          : if provided, output ExtXYZ path for selected frames
    extra_info       : additional metadata to include in JSON

    Returns
    -------
    json_out path
    """
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)

    result = {
        "method": method,
        "n_selected": len(selected_indices),
        "timestamp": datetime.now().isoformat(),
        "structures": [
            {
                "original_index": int(i),
                "structure_id": cand_ids[i],
                "selection_order": order,
            }
            for order, i in enumerate(selected_indices, 1)
        ],
    }
    if extra_info:
        result.update(extra_info)

    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{method}] JSON → {json_out}  ({len(selected_indices)} structures)")

    if xyz_src is not None and xyz_out is not None:
        frames = read_xyz_frames(xyz_src)
        os.makedirs(os.path.dirname(xyz_out) or ".", exist_ok=True)
        with open(xyz_out, "w") as f:
            for idx in selected_indices:
                frame = frames[idx]
                if not frame.endswith("\n"):
                    frame += "\n"
                f.write(frame)
        print(f"  [{method}] XYZ  → {xyz_out}  ({len(selected_indices)} frames)")

    return json_out


# ──────────────────────────────────────────────────────────────────────────────
# Sampling methods
# ──────────────────────────────────────────────────────────────────────────────

def random_sampling(
    cand_feats: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> List[int]:
    """
    Uniform random sampling (baseline).

    Parameters
    ----------
    cand_feats : (N, D) candidate feature matrix
    n_samples  : number of structures to select
    seed       : random seed

    Returns
    -------
    selected_indices : list of int (length = min(n_samples, N))
    """
    rng = np.random.default_rng(seed)
    N = len(cand_feats)
    n = min(n_samples, N)
    selected = rng.choice(N, size=n, replace=False).tolist()
    print(f"  [Random] {n} selected")
    return selected


def direct_sampling(
    cand_feats: np.ndarray,
    n_samples: int,
    threshold_init: float = 1.0,
    tol: float = 0.05,
    max_iter: int = 60,
) -> Tuple[List[int], float]:
    """
    DIRECT sampling: Birch clustering with automatic threshold search
    to yield approximately n_samples clusters, one representative each.

    Parameters
    ----------
    cand_feats     : (N, D) candidate feature matrix
    n_samples      : target number of structures
    threshold_init : initial Birch threshold for bisection search
    tol            : acceptable relative deviation from n_samples
    max_iter       : maximum bisection iterations

    Returns
    -------
    selected_indices : list of int
    best_threshold   : float (Birch threshold used)
    """
    lo, hi = 1e-4, 20.0
    best_indices: List[int] = list(range(min(n_samples, len(cand_feats))))
    best_thresh = threshold_init
    best_diff = float("inf")

    print(f"  [DIRECT] bisection search (target={n_samples}) …")
    for it in range(max_iter):
        mid = (lo + hi) / 2.0
        brc = Birch(n_clusters=None, threshold=mid)
        labels = brc.fit_predict(cand_feats)
        unique_labels = np.unique(labels)

        indices: List[int] = []
        for lab in unique_labels:
            members = np.where(labels == lab)[0]
            centroid = cand_feats[members].mean(axis=0)
            dists = np.linalg.norm(cand_feats[members] - centroid, axis=1)
            indices.append(int(members[np.argmin(dists)]))

        n_got = len(indices)
        diff = abs(n_got - n_samples)

        if diff < best_diff:
            best_diff = diff
            best_indices = indices
            best_thresh = mid

        if n_got > n_samples:
            lo = mid
        else:
            hi = mid

        if diff <= max(1, int(n_samples * tol)):
            print(f"    iter={it+1:3d}  threshold={mid:.5f}  got={n_got}  ✓ converged")
            break
        if it % 10 == 0:
            print(f"    iter={it+1:3d}  threshold={mid:.5f}  got={n_got}  diff={diff}")

    print(f"  [DIRECT] threshold={best_thresh:.5f}  selected={len(best_indices)}")
    return best_indices, best_thresh


def lcmd_sampling(
    cand_feats: np.ndarray,
    center_feats: np.ndarray,
    n_samples: int,
    batch_size: int = 65536,
) -> List[int]:
    """
    LCMD — greedy farthest-point (sequential maximin) sampling.

    No clustering step. At every iteration the candidate with the largest
    minimum-distance to the current selected set is chosen.

    Algorithm
    ---------
    1. Compute min-distance of every candidate to center_feats via KDTree.
    2. Loop n_samples times:
       a. chosen = argmax(min_dist_sq)          # farthest unselected point
       b. selected.append(chosen)
       c. Update min_dist_sq for all candidates using BLAS gemv:
          dist²(x, y_new) = ‖x‖² + ‖y_new‖² − 2 x·y_new

    This is equivalent to the lcmd_greedy_fast function in the reference
    implementation (4_lcmd_fast_version_save.py).

    Parameters
    ----------
    cand_feats   : (N, D) candidate feature matrix (float64)
    center_feats : (Nc, D) initial center (reference) feature matrix
    n_samples    : number of structures to select
    batch_size   : chunk size for BLAS distance updates

    Returns
    -------
    selected_indices : list of int (length = n_samples or N if exhausted)
    """
    N = len(cand_feats)
    n_select = min(n_samples, N)
    print(f"\n  [LCMD] candidates={N:,}  centers={len(center_feats):,}  target={n_select}")

    feats = np.ascontiguousarray(cand_feats, dtype=np.float64)

    # ── Step 1: initial min-dist² from center set (KDTree) ───────────────
    print("  [LCMD] Computing initial distances from centers (KDTree) …")
    kd = KDTree(np.ascontiguousarray(center_feats, dtype=np.float64))
    min_dist_sq = np.empty(N, dtype=np.float64)
    for i in tqdm(range(0, N, batch_size), desc="  init dist", leave=False):
        j = min(i + batch_size, N)
        d, _ = kd.query(feats[i:j], k=1)
        min_dist_sq[i:j] = d * d

    # Precompute ‖x‖² for all candidates (used in BLAS update)
    cand_sq = np.einsum("ij,ij->i", feats, feats)

    # ── Step 2: greedy farthest-point loop ───────────────────────────────
    selected: List[int] = []
    mask = np.zeros(N, dtype=bool)   # True = already selected

    print("  [LCMD] Greedy farthest-point selection …")
    with tqdm(total=n_select, desc="  LCMD") as pbar:
        while len(selected) < n_select:
            # Pick candidate farthest from current center set
            # (mask out already-selected points)
            tmp = np.where(mask, -np.inf, min_dist_sq)
            chosen = int(np.argmax(tmp))

            selected.append(chosen)
            mask[chosen] = True

            # Update min-dist²: compare every candidate to the new point
            # dist²(x, y) = ‖x‖² + ‖y‖² − 2 x·y   (BLAS gemv per batch)
            y    = feats[chosen]
            y_sq = float(np.dot(y, y))
            for i in range(0, N, batch_size):
                j    = min(i + batch_size, N)
                dot  = feats[i:j] @ y                    # BLAS gemv
                d2   = cand_sq[i:j] + y_sq - 2.0 * dot
                np.minimum(min_dist_sq[i:j], d2, out=min_dist_sq[i:j])

            pbar.update(1)
            if len(selected) % 500 == 0:
                gc.collect()

    print(f"  [LCMD] Selected {len(selected)} structures")
    return selected
