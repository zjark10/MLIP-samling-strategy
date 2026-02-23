"""
src/plotting.py
---------------
Visualisation utilities for the active learning pipeline.

Functions
---------
plot_pca_space          : 2-D PCA scatter with sampling highlights
plot_parity             : density-coloured parity plot (energy / force)
plot_cluster_analysis   : mean / max cluster size & CV vs. dataset size
plot_force_rmse_by_bin  : force RMSE binned by interatomic distance
plot_violin             : split violin for force MAE / force softening
plot_pca_variance       : cumulative & individual explained variance
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.5,
})

# Shared colour / marker scheme
COLORS = {
    "center":  "#F28E2B",   # orange  – foundational / center dataset
    "random":  "#4E79A7",   # blue
    "direct":  "#B07AA1",   # purple
    "lcmd":    "#E15759",   # red
    "all":     "#AAAAAA",   # grey    – all candidates
}
MARKERS = {
    "center": "*",
    "random": "o",
    "direct": "^",
    "lcmd":   "D",
}
MARKER_SIZE = {"center": 80, "random": 40, "direct": 40, "lcmd": 40}
EDGE_WIDTH  = 1.5


def _save(fig: plt.Figure, path: str, dpi: int = 300) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# PCA space
# ──────────────────────────────────────────────────────────────────────────────

def plot_pca_space(
    all_feats: np.ndarray,
    selected: Optional[Dict[str, np.ndarray]] = None,
    center_feats: Optional[np.ndarray] = None,
    output_path: str = "pca_space.png",
    pc_x: int = 0,
    pc_y: int = 1,
    title: str = "PCA Feature Space",
) -> str:
    """
    Scatter plot of the PCA feature space with optional highlighted subsets.

    Parameters
    ----------
    all_feats    : (N, D) array of all candidate features (background)
    selected     : dict mapping method name → (n, D) selected feature arrays.
                   Keys should be 'random', 'direct', or 'lcmd'.
    center_feats : (Nc, D) array of center (foundational) dataset features
    output_path  : file path for saved figure
    pc_x, pc_y   : which principal components to plot (0-indexed)
    title        : figure title
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Background
    ax.scatter(
        all_feats[:, pc_x], all_feats[:, pc_y],
        c=COLORS["all"], alpha=0.3, s=4, edgecolors="none", label="Candidates",
    )

    # Center dataset
    if center_feats is not None:
        ax.scatter(
            center_feats[:, pc_x], center_feats[:, pc_y],
            c="none",
            edgecolors=COLORS["center"],
            marker=MARKERS["center"],
            s=MARKER_SIZE["center"],
            linewidths=EDGE_WIDTH,
            alpha=0.9,
            label="Center (foundational)",
            zorder=4,
        )

    # Sampling results
    if selected:
        for name, feats in selected.items():
            c = COLORS.get(name, "#333333")
            m = MARKERS.get(name, "o")
            s = MARKER_SIZE.get(name, 40)
            ax.scatter(
                feats[:, pc_x], feats[:, pc_y],
                c="none",
                edgecolors=c,
                marker=m,
                s=s,
                linewidths=EDGE_WIDTH,
                alpha=0.85,
                label=name.capitalize(),
                zorder=5,
            )

    ax.set_xlabel(f"PC {pc_x + 1}")
    ax.set_ylabel(f"PC {pc_y + 1}")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, framealpha=0.7)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Parity plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_parity(
    results: Dict[str, Dict],
    output_path: str = "parity.png",
    title: str = "",
    max_points: int = 2000,
) -> str:
    """
    Density-coloured parity plot (energy and force) for one or more models.

    Parameters
    ----------
    results     : dict mapping label → evaluate_model() output dict.
                  Each value must have 'dft_energy', 'mlip_energy',
                  'dft_forces', 'mlip_forces'.
    output_path : file path for saved figure
    title       : overall figure title
    max_points  : max scatter points per panel (random subsample if exceeded)
    """
    from scipy.stats import gaussian_kde

    labels = list(results.keys())
    n_models = len(labels)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models), squeeze=False)

    units = {"energy": "eV/atom", "force": "eV/Å"}

    for row, label in enumerate(labels):
        r = results[label]
        for col, (key_dft, key_mlip, mode) in enumerate([
            ("dft_energy",  "mlip_energy",  "energy"),
            ("dft_forces",  "mlip_forces",  "force"),
        ]):
            ax = axes[row][col]
            x = np.array(r[key_dft])
            y = np.array(r[key_mlip])

            # Subsample for density estimation
            if len(x) > max_points:
                idx = np.random.choice(len(x), max_points, replace=False)
            else:
                idx = np.arange(len(x))
            xs, ys = x[idx], y[idx]

            try:
                z = gaussian_kde(np.vstack([xs, ys])).pdf(np.vstack([xs, ys]))
                order = z.argsort()
                ax.scatter(xs[order], ys[order], c=z[order], s=6,
                           cmap="plasma", alpha=0.8)
            except Exception:
                ax.scatter(xs, ys, s=6, alpha=0.5)

            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    color="grey", linestyle="--", linewidth=1.5)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlabel(f"DFT {mode} ({units[mode]})")
            ax.set_ylabel(f"MLIP {mode} ({units[mode]})")
            ax.set_aspect("equal")
            ax.set_title(f"{label} — {mode}")
            ax.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=18, y=1.01)
    plt.tight_layout()
    return _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_cluster_analysis(
    data: Dict[str, Dict],
    output_path: str = "cluster_analysis.png",
    reference: Optional[Dict[str, float]] = None,
) -> str:
    """
    Plot mean cluster size, max cluster size, and CV vs. dataset size
    for different sampling methods.

    Parameters
    ----------
    data        : dict mapping method name → dict with keys:
                    'sizes'     : list of dataset sizes (x-axis)
                    'mean'      : list of mean active cluster sizes
                    'std'       : list of std active cluster sizes
                    'max'       : list of max cluster sizes
    output_path : file path for saved figure
    reference   : optional dict {'mean': float, 'std': float, 'max': float}
                  representing a reference dataset (e.g. foundational set),
                  drawn as a horizontal dashed line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Mean Active Cluster Size", "Max Cluster Size", "Coefficient of Variation"]

    for method, d in data.items():
        c = COLORS.get(method, "#333333")
        m = MARKERS.get(method, "o")
        sizes = d["sizes"]
        mean  = np.array(d["mean"])
        std   = np.array(d["std"])
        mx    = np.array(d["max"])
        cv    = std / mean

        kw = dict(color=c, marker=m, markersize=10,
                  markeredgecolor=c, markerfacecolor="none",
                  markeredgewidth=EDGE_WIDTH, label=method.capitalize())
        axes[0].plot(sizes, mean, **kw)
        axes[1].plot(sizes, mx,   **kw)
        axes[2].plot(sizes, cv,   **kw)

    if reference is not None:
        ref_kw = dict(color=COLORS["center"], linestyle="--",
                      linewidth=2, label="Foundational")
        if "mean" in reference:
            axes[0].axhline(reference["mean"], **ref_kw)
        if "max" in reference:
            axes[1].axhline(reference["max"],  **ref_kw)
        if "mean" in reference and "std" in reference:
            axes[2].axhline(reference["std"] / reference["mean"], **ref_kw)

    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("Dataset size")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.7)

    plt.tight_layout()
    return _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Force RMSE by interatomic distance bin
# ──────────────────────────────────────────────────────────────────────────────

def plot_force_rmse_by_bin(
    bin_data: Dict[str, Dict],
    output_path: str = "force_rmse_by_bin.png",
    metric: str = "rmse",
    title: str = "Force Error by Interatomic Distance",
) -> str:
    """
    Plot force RMSE (or MAE) as a function of interatomic distance bin.

    Parameters
    ----------
    bin_data    : dict mapping method name → dict with keys:
                    'bin_centers' : list of float (Å)
                    'rmse'        : list of float (eV/Å)  [or 'mae']
    output_path : file path for saved figure
    metric      : 'rmse' or 'mae'
    title       : figure title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, d in bin_data.items():
        c = COLORS.get(method, "#333333")
        m = MARKERS.get(method, "o")
        ms = 12 if method == "center" else 10
        ax.plot(
            d["bin_centers"], d[metric],
            color=c, marker=m,
            markersize=ms,
            markeredgecolor=c, markerfacecolor="none",
            markeredgewidth=EDGE_WIDTH,
            label=method.capitalize(),
        )

    ax.set_xlabel("Interatomic distance (Å)")
    ax.set_ylabel(f"Force {metric.upper()} (eV/Å)")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Violin plots (force MAE distribution & force softening)
# ──────────────────────────────────────────────────────────────────────────────

def plot_violin(
    data: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "violin.png",
    metric: str = "Force MAE",
    ylim: Optional[Tuple[float, float]] = None,
    reference_line: Optional[float] = None,
) -> str:
    """
    Split-violin plot comparing scratch training vs. fine-tuning.

    Parameters
    ----------
    data        : dict mapping method name → dict with keys:
                    'scratch' : 1-D array of per-structure values
                    'ft'      : 1-D array of per-structure values
                  Method names should be in COLORS dict.
    output_path : file path for saved figure
    metric      : y-axis label
    ylim        : optional (ymin, ymax)
    reference_line : optional horizontal reference line (e.g. y=1 for softening)
    """
    methods = list(data.keys())
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(methods)), 5))

    ft_color_map = {
        "random": "#A0C8E8",
        "direct": "#D4B3D9",
        "lcmd":   "#F5A5A8",
        "center": "#FFD580",
    }

    for pos, method in enumerate(methods):
        d_scratch = np.array(data[method].get("scratch", []))
        d_ft      = np.array(data[method].get("ft",      []))
        d_scratch = d_scratch[~np.isnan(d_scratch)]
        d_ft      = d_ft[~np.isnan(d_ft)]

        c_scratch = COLORS.get(method, "#888888")
        c_ft      = ft_color_map.get(method, "#CCCCCC")

        for side, vals, color, offset, hatch in [
            ("left",  d_scratch, c_scratch, -0.15, None),
            ("right", d_ft,      c_ft,      +0.15, "///"),
        ]:
            if len(vals) == 0:
                continue
            parts = ax.violinplot([vals], positions=[pos],
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.9)
                if hatch:
                    pc.set_hatch(hatch)
                    pc.set_edgecolor("black")
                    pc.set_linewidth(1.2)
                verts = pc.get_paths()[0].vertices
                centre = float(pos)
                if side == "left":
                    verts[:, 0] = np.minimum(verts[:, 0], centre)
                else:
                    verts[:, 0] = np.maximum(verts[:, 0], centre)
                pc.get_paths()[0].vertices = verts

            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.scatter(pos + offset, med, color="white", s=60,
                       zorder=10, edgecolor="black", linewidth=1.5)
            ax.plot([pos + offset, pos + offset], [q25, q75],
                    color="black", linewidth=2.5, zorder=9)

    if reference_line is not None:
        ax.axhline(reference_line, color="black", linestyle="--", alpha=0.6, linewidth=1.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel(metric)
    ax.set_title(metric)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PCA explained variance
# ──────────────────────────────────────────────────────────────────────────────

def plot_pca_variance(
    pca_model_json: str,
    output_path: str = "pca_variance.png",
    extra_components: int = 5,
) -> str:
    """
    Plot cumulative and individual explained variance from a saved PCA model.

    Parameters
    ----------
    pca_model_json   : path to PCA model JSON (from src.features.train_pca)
    output_path      : file path for saved figure
    extra_components : number of extra components beyond Kaiser cutoff to show
    """
    import json

    with open(pca_model_json) as f:
        info = json.load(f)

    n_kaiser = info["n_components"]
    full_ratio = np.array(info.get(
        "full_explained_variance_ratio",
        info["explained_variance_ratio"],
    ))
    eigenvals = np.array(info["explained_variance"])

    plot_n = min(n_kaiser + extra_components, len(full_ratio))
    cumvar = np.cumsum(full_ratio[:plot_n])
    comps  = np.arange(1, plot_n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Cumulative
    ax1.plot(comps[:n_kaiser], cumvar[:n_kaiser],
             color="#4E79A7", linewidth=2.5, marker="o", markersize=6,
             label="Kaiser selected")
    if plot_n > n_kaiser:
        ax1.plot(comps[n_kaiser - 1:], cumvar[n_kaiser - 1:],
                 color="#E15759", linewidth=2.5, marker="o", markersize=6,
                 linestyle="--", label=f"Beyond Kaiser (+{extra_components})")
    ax1.axvline(n_kaiser, color="grey", linestyle=":", linewidth=1.8,
                label=f"Cutoff (n={n_kaiser})")
    ax1.axhline(cumvar[n_kaiser - 1], color="grey", linestyle=":", linewidth=1.2)
    ax1.annotate(f"{cumvar[n_kaiser-1]:.3f}",
                 xy=(n_kaiser, cumvar[n_kaiser - 1]),
                 xytext=(n_kaiser + 0.4, cumvar[n_kaiser - 1] - 0.025),
                 fontsize=14)
    ax1.set_xlabel("Number of principal components")
    ax1.set_ylabel("Cumulative explained variance")
    ax1.set_title("PCA — Cumulative Variance (Kaiser's rule)")
    ax1.set_xlim(0.5, plot_n + 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Individual
    colors_bar = ["#4E79A7"] * n_kaiser + ["#E15759"] * (plot_n - n_kaiser)
    ax2.bar(comps, full_ratio[:plot_n], color=colors_bar, edgecolor="white", linewidth=0.5)
    ax2.axvline(n_kaiser + 0.5, color="grey", linestyle=":", linewidth=1.8)
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_title("PCA — Individual Variance")
    ax2.set_xlim(0.5, plot_n + 0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)
