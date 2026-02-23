"""
src/vasp_inputs.py
------------------
Generate VASP static calculation inputs (POSCAR / INCAR / KPOINTS / POTCAR)
from an ExtXYZ file using pymatgen's MatPESStaticSet.

Typical usage
-------------
from src.vasp_inputs import generate_vasp_inputs

generate_vasp_inputs(
    xyz_file      = "lcmd_selected.xyz",
    output_root   = "static_vasp_inputs",
    potcar_root   = "/opt/vasp/potpaw_PBE.54",
    kpoint_density = 100,
    n_groups       = 1,
)

Directory layout
----------------
static_vasp_inputs/
    group_000/
        0001/          <- structure index (zero-padded)
            POSCAR
            INCAR
            KPOINTS
            POTCAR
        0002/
            ...
    group_001/
        ...

If n_groups == 1, the group_000/ level is omitted:
static_vasp_inputs/
    0001/
        POSCAR ...
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

warnings.filterwarnings("ignore")

# POTCAR aliases for special elements (lanthanides etc.)
_POTCAR_ALIAS: Dict[str, str] = {
    "Er": "Er_3",
    "Nd": "Nd_3",
    "Tm": "Tm_3",
    "Pr": "Pr_3",
    "Ho": "Ho_3",
    "Ce": "Ce",
    "Eu": "Eu",
    "Yb": "Yb_2",
}


# ──────────────────────────────────────────────────────────────────────────────
# POTCAR helpers
# ──────────────────────────────────────────────────────────────────────────────

def _elem_from_folder(folder_name: str) -> str:
    """Extract base element symbol from a POTCAR folder name (e.g. 'Na_pv' → 'Na')."""
    m = re.match(r"[A-Za-z]+", folder_name)
    if not m:
        raise ValueError(f"Cannot parse element from folder name: {folder_name!r}")
    return m.group(0)


def _find_potcar(elem: str, potcar_root: str) -> str:
    """
    Locate POTCAR file for an element under potcar_root.
    Search order: alias → _sv → _3 → _h → plain.

    Parameters
    ----------
    elem       : element symbol (e.g. 'Na')
    potcar_root: root directory of POTCAR library

    Returns
    -------
    absolute path to POTCAR file
    """
    folder_name = _POTCAR_ALIAS.get(elem, elem)
    candidates = [folder_name, f"{elem}_sv", f"{elem}_3", f"{elem}_h", elem]
    for cand in candidates:
        p = Path(potcar_root, cand, "POTCAR")
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"POTCAR for '{elem}' not found under {potcar_root}. "
        f"Tried: {candidates}"
    )


def _build_potcar(spec_file: Path, potcar_root: str, out_path: Path) -> None:
    """
    Read a POTCAR.spec file and concatenate individual POTCARs into one file.

    Parameters
    ----------
    spec_file  : path to POTCAR.spec (one folder name per line)
    potcar_root: root POTCAR library directory
    out_path   : destination POTCAR file
    """
    with open(spec_file) as f:
        spec_lines = [l.strip() for l in f if l.strip()]

    remapped = [_POTCAR_ALIAS.get(_elem_from_folder(l), l) for l in spec_lines]
    spec_file.write_text("\n".join(remapped) + "\n")

    pot_paths = [
        _find_potcar(_elem_from_folder(fn), potcar_root) for fn in remapped
    ]
    with open(out_path, "wb") as w:
        subprocess.run(["cat", *pot_paths], stdout=w, check=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core: one structure → VASP input set
# ──────────────────────────────────────────────────────────────────────────────

def _write_vasp_set(
    atoms,  # ASE Atoms
    dst: Path,
    potcar_root: str,
    kpoint_density: int,
    user_incar_settings: Optional[Dict] = None,
) -> None:
    """
    Write POSCAR / INCAR / KPOINTS / POTCAR for one structure.

    Parameters
    ----------
    atoms              : ASE Atoms object
    dst                : destination directory (created if absent)
    potcar_root        : root POTCAR library directory
    kpoint_density     : k-point density (Å⁻³) passed to automatic_density_by_vol
    user_incar_settings: additional INCAR overrides (merged with defaults)
    """
    from ase.io import write as ase_write
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Kpoints
    from pymatgen.io.vasp.sets import MatPESStaticSet

    dst.mkdir(parents=True, exist_ok=True)

    # POSCAR
    ase_write(str(dst / "POSCAR"), atoms, format="vasp", direct=True)

    # INCAR + POTCAR.spec via MatPESStaticSet
    struct = Structure.from_file(str(dst / "POSCAR"))
    incar_overrides = {"NPAR": 4, "NCORE": 4}
    if user_incar_settings:
        incar_overrides.update(user_incar_settings)

    vset = MatPESStaticSet(
        struct,
        xc_functional="PBE",
        user_potcar_functional="PBE_54",
        user_incar_settings=incar_overrides,
    )
    vset.write_input(str(dst), potcar_spec=True)

    # KPOINTS
    Kpoints.automatic_density_by_vol(struct, kpoint_density).write_file(
        str(dst / "KPOINTS")
    )

    # POTCAR
    spec_file = dst / "POTCAR.spec"
    if spec_file.exists():
        _build_potcar(spec_file, potcar_root, dst / "POTCAR")
    else:
        warnings.warn(f"POTCAR.spec not found in {dst} — POTCAR not generated.")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_vasp_inputs(
    xyz_file: str,
    output_root: str,
    potcar_root: str,
    kpoint_density: int = 100,
    n_groups: int = 1,
    user_incar_settings: Optional[Dict] = None,
    indices: Optional[List[int]] = None,
) -> None:
    """
    Generate VASP static calculation inputs for all (or selected) structures
    in an ExtXYZ file.

    Parameters
    ----------
    xyz_file           : input ExtXYZ file
    output_root        : root output directory
    potcar_root        : root directory of POTCAR library
    kpoint_density     : k-point density for automatic_density_by_vol (default 100)
    n_groups           : number of sub-groups to distribute structures across
                         (useful for splitting HPC job arrays).
                         If 1, no group subdirectory is created.
    user_incar_settings: optional dict of extra INCAR tags
    indices            : if given, only process these 0-based structure indices

    Directory layout (n_groups > 1)
    --------------------------------
    output_root/
        group_000/
            0001/ POSCAR INCAR KPOINTS POTCAR
            0002/ ...
        group_001/
            ...

    Directory layout (n_groups == 1)
    ---------------------------------
    output_root/
        0001/ POSCAR INCAR KPOINTS POTCAR
        0002/ ...
    """
    from ase.io import read as ase_read

    print(f"Loading structures from: {xyz_file}")
    all_atoms = ase_read(xyz_file, index=":")
    print(f"  {len(all_atoms)} structures loaded")

    if indices is not None:
        selected = [(i, all_atoms[i]) for i in indices]
    else:
        selected = list(enumerate(all_atoms))

    n = len(selected)
    width = len(str(n))

    ok = fail = 0
    fail_log: List[str] = []

    print(f"\nGenerating VASP inputs for {n} structures …")
    for pos, (orig_idx, atoms) in enumerate(tqdm(selected, unit="struct"), start=1):
        struct_name = f"{orig_idx + 1:0{width}}"

        if n_groups > 1:
            group_id = (pos - 1) % n_groups
            dst = Path(output_root) / f"group_{group_id:03d}" / struct_name
        else:
            dst = Path(output_root) / struct_name

        try:
            _write_vasp_set(
                atoms, dst, potcar_root, kpoint_density, user_incar_settings
            )
            ok += 1
        except FileNotFoundError as exc:
            fail += 1
            msg = f"  POTCAR missing for {struct_name}: {exc}"
            fail_log.append(msg)
            print(msg)
        except Exception as exc:
            fail += 1
            msg = f"  Error for {struct_name}: {exc}"
            fail_log.append(msg)
            print(msg)

    print(f"\n── VASP input generation complete ──")
    print(f"  Success : {ok}")
    print(f"  Failed  : {fail}")
    if fail_log:
        log_path = Path(output_root) / "failed_structures.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(fail_log) + "\n")
        print(f"  Failure log → {log_path}")


# ──────────────────────────────────────────────────────────────────────────────
# LAMMPS trajectory reader (for NVT/NPT trajectories)
# ──────────────────────────────────────────────────────────────────────────────

def read_lammps_atoms(
    xyz_path: str,
    data_path: str,
    timestep: int,
):
    """
    Read a single snapshot from a LAMMPS dump (custom xyz format) at a given
    timestep and return an ASE Atoms object.

    Expected dump format
    --------------------
    ITEM: TIMESTEP
    <step>
    ITEM: NUMBER OF ATOMS
    <n>
    ITEM: BOX BOUNDS ...
    ...
    ITEM: ATOMS id type element x y z ...

    Parameters
    ----------
    xyz_path  : LAMMPS dump file (out.xyz)
    data_path : LAMMPS data file (input.data) — used for cell information
    timestep  : target timestep to extract

    Returns
    -------
    ASE Atoms object, or None if timestep not found
    """
    import numpy as np
    from ase import Atoms

    def _read_cell(data: str) -> np.ndarray:
        xlo = xhi = ylo = yhi = zlo = zhi = xy = xz = yz = 0.0
        with open(data) as f:
            for line in f:
                if "xlo xhi" in line:
                    xlo, xhi = map(float, line.split()[:2])
                elif "ylo yhi" in line:
                    ylo, yhi = map(float, line.split()[:2])
                elif "zlo zhi" in line:
                    zlo, zhi = map(float, line.split()[:2])
                elif "xy xz yz" in line:
                    xy, xz, yz = map(float, line.split()[:3])
        lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
        return np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])

    cell = _read_cell(data_path)

    with open(xyz_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1].strip())
            n_atoms = int(lines[i + 3].strip())
            header_end = i + 9
            if step == timestep:
                symbols, positions = [], []
                for j in range(n_atoms):
                    fields = lines[header_end + j].split()
                    symbols.append(fields[2])
                    positions.append(list(map(float, fields[3:6])))
                return Atoms(symbols, positions, cell=cell, pbc=True)
            i = header_end + n_atoms
        else:
            i += 1

    return None  # timestep not found
