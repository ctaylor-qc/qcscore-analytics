"""
boltzmann_average.py
---------------------
Boltzmann-weighted ensemble averaging of multi-pose QC Score output.

When multiple docking poses are run through Promethium for the same ligand,
each pose produces its own row in the QC Score CSV. Before running the main
residue interaction analysis, this script collapses those poses into a single
representative row per ligand using Boltzmann weighting over score+strain.

WHY score+strain AS THE BOLTZMANN WEIGHT:
  score+strain = QC Score + ligand strain energy = total energy cost of the pose.
  This is the physically correct quantity for ensemble averaging: it penalises
  both weak protein-ligand interactions (high score) and geometrically strained
  ligand conformations (high strain). The Boltzmann factor exp(-ΔE/RT) assigns
  exponentially lower weight to higher-energy poses.

WHY THIS IS USUALLY BEST-POSE SELECTION:
  At 300K, RT ≈ 0.596 kcal/mol. A pose that is 3 kcal/mol above the best pose
  receives weight ~1/150. In practice most docked poses differ by 5-20 kcal/mol,
  so the weighting collapses to the best valid pose. This is physically correct
  rather than arbitrary — the script will log effective weights so you can verify
  whether genuine blending occurred. If you want softer averaging (e.g. to
  propagate uncertainty from near-degenerate poses), increase boltzmann_temperature.

POSE IDENTIFICATION CONVENTION:
  Original/reference poses:  ligand_X   (no numeric suffix)
  Docked poses:              ligand_X_1, ligand_X_2, ligand_X_3, ...
  Base name is extracted by stripping trailing _N suffixes.
  Configurable via pose_suffix_pattern if your naming differs.

VALIDITY FILTERS (applied before weighting):
  1. score must be non-NaN (failed calculations excluded)
  2. score+strain must be <= clash_score_threshold (default 0)
     Positive score+strain indicates a clashing/repulsive pose.
  Ligands with no valid poses after filtering are excluded entirely
  and reported in the output log.

Usage:
    python boltzmann_average.py --config boltzmann_config.json

Config file (JSON):
    {
        "input_file":             "qcscore_output.csv",
        "output_file":            "qcscore_boltzmann.csv",
        "boltzmann_temperature":  300,
        "clash_score_threshold":  0,
        "pose_suffix_pattern":    "_[0-9]+$"
    }

Outputs:
    <output_file>   — averaged CSV, one row per ligand, ready for analysis script
    <output_file>.log — per-ligand weight report for auditing
"""

import json
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Physical constant: gas constant in kcal/(mol·K)
R_KCAL = 1.987204e-3  # kcal / (mol·K)

DEFAULT_CONFIG = {
    "input_file":            "qcscore_output.csv",
    "output_file":           "qcscore_boltzmann.csv",
    "boltzmann_temperature": 300,
    "clash_score_threshold": 0,
    "pose_suffix_pattern":   r"_[0-9]+$",
}

META_COLS = {"workflow_id", "name", "active", "score", "score_plus_strain", "strain"}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return {**DEFAULT_CONFIG, **cfg}


def extract_base_name(name, pattern):
    """
    Strip trailing pose suffix to get the base ligand name.

    The default pattern (_[0-9]+$) would incorrectly strip meaningful numeric
    suffixes that are part of the ligand identifier (e.g. ligand_17 → ligand).
    To guard against this, we only strip the suffix if the remaining base name
    itself already appears in the dataset as a distinct entry — i.e. we treat
    _N as a pose suffix only when a bare version of the name also exists.
    This check is done at the dataset level in build_base_name_map(); this
    function just applies a pre-computed mapping.
    """
    return re.sub(pattern, "", name)


def build_base_name_map(names, pattern):
    """
    Build a name → base_name mapping that distinguishes ligand identifiers
    (ligand_17, ligand_20) from pose suffixes (ligand_17_1, ligand_17_2).

    Strategy: a trailing _N suffix is treated as a POSE suffix only if the
    stripped version (the candidate base name) also appears in the name list.
    Otherwise the full name is its own base name.

    Example:
      names = [ligand_17, ligand_17_1, ligand_17_2, ligand_17_3]
      ligand_17_1 → strip _1 → ligand_17, which IS in names → base = ligand_17 ✓
      ligand_17   → strip nothing (already a base) → base = ligand_17 ✓

      names = [ligand_20, ligand_20_1, ...]
      ligand_20_1 → strip _1 → ligand_20, which IS in names → base = ligand_20 ✓
      ligand_20   → no strippable suffix that matches → base = ligand_20 ✓
    """
    name_set = set(names)
    mapping = {}
    for name in names:
        candidate = re.sub(pattern, "", name)
        if candidate != name and candidate in name_set:
            # The stripped version exists — this is a pose suffix
            mapping[name] = candidate
        else:
            # No valid base found by stripping — name is its own base
            mapping[name] = name
    return mapping


def boltzmann_weights(energies, RT):
    """
    Compute Boltzmann weights for an array of energies.
    Shifted by minimum for numerical stability — only energy differences matter.
    Returns weight array summing to 1.
    """
    delta = np.array(energies, dtype=float) - np.min(energies)
    w = np.exp(-delta / RT)
    return w / w.sum()


def weighted_average_row(group, weights, numeric_cols):
    """Compute weighted average of all numeric columns for a pose group."""
    row = {}
    for col in numeric_cols:
        vals = group[col].values.astype(float)
        row[col] = float(np.dot(weights, vals))
    return row


def format_weights(names, weights):
    """Format weight vector as a human-readable string for logging."""
    parts = [f"{n}: {w:.4f}" for n, w in zip(names, weights)]
    return "  |  ".join(parts)


def run(cfg):
    input_path = Path(cfg["input_file"])
    output_path = Path(cfg["output_file"])
    log_path = output_path.with_suffix(".log")
    RT = R_KCAL * cfg["boltzmann_temperature"]
    clash_thresh = cfg["clash_score_threshold"]
    suffix_pat = cfg["pose_suffix_pattern"]

    print(f"\nBoltzmann Ensemble Averaging")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Temperature: {cfg['boltzmann_temperature']} K  (RT = {RT:.4f} kcal/mol)")
    print(f"  Clash threshold: score+strain > {clash_thresh} kcal/mol → excluded")

    # ── Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    required = {"name", "score", "score_plus_strain", "strain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    residue_cols = [c for c in df.columns
                    if c not in META_COLS
                    and c not in {"SMILES", "IC50"}
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    numeric_cols = ["score", "score_plus_strain", "strain"] + residue_cols

    print(f"  Rows loaded: {len(df)}  |  Residue columns: {len(residue_cols)}")

    # ── Assign base names ────────────────────────────────────────────────────
    base_map = build_base_name_map(df["name"].tolist(), suffix_pat)
    df["_base"] = df["name"].map(base_map)
    n_ligands = df["_base"].nunique()
    print(f"  Unique ligands (base names): {n_ligands}")

    # ── Process each ligand group ────────────────────────────────────────────
    log_lines = []
    log_lines.append("=" * 78)
    log_lines.append("Boltzmann Ensemble Averaging — Pose Weight Report")
    log_lines.append("=" * 78)
    log_lines.append(f"Temperature: {cfg['boltzmann_temperature']} K  "
                     f"RT = {RT:.4f} kcal/mol")
    log_lines.append(f"Clash threshold: score+strain > {clash_thresh} kcal/mol\n")

    output_rows = []
    n_excluded = 0
    n_single = 0
    n_blended = 0

    for base, group in df.groupby("_base", sort=True):
        # Step 1: filter failed calculations
        valid = group[group["score"].notna()].copy()

        # Step 2: filter clashing poses
        clashing = valid[valid["score_plus_strain"] > clash_thresh]
        valid = valid[valid["score_plus_strain"] <= clash_thresh].copy()

        log_lines.append(f"Ligand: {base}")
        log_lines.append(f"  Total poses: {len(group)}  |  "
                         f"Failed (NaN): {group['score'].isna().sum()}  |  "
                         f"Clashing: {len(clashing)}  |  "
                         f"Valid: {len(valid)}")

        if len(clashing) > 0:
            for _, cr in clashing.iterrows():
                log_lines.append(f"    Excluded (clash): {cr['name']}  "
                                  f"score+strain={cr['score_plus_strain']:.2f}")

        if len(valid) == 0:
            log_lines.append("  *** ALL poses invalid — ligand excluded from output ***\n")
            n_excluded += 1
            continue

        # Step 3: Boltzmann weights over score+strain
        energies = valid["score_plus_strain"].values.astype(float)
        weights = boltzmann_weights(energies, RT)
        pose_names = valid["name"].tolist()

        log_lines.append(f"  Weights ({cfg['boltzmann_temperature']}K Boltzmann):")
        for pname, w, e in zip(pose_names, weights, energies):
            dominant = " ◄" if w > 0.99 else (" ≈" if w > 0.01 else "")
            log_lines.append(f"    {pname:30s}  score+strain={e:8.3f}  w={w:.6f}{dominant}")

        # Effective number of poses contributing meaningfully
        n_eff = 1.0 / np.sum(weights**2)
        log_lines.append(f"  Effective poses (1/Σw²): {n_eff:.2f}")

        if n_eff < 1.05:
            n_single += 1
            log_lines.append("  → Effectively single-pose (best pose dominates)")
        else:
            n_blended += 1
            log_lines.append(f"  → Genuine blending across {n_eff:.1f} effective poses")

        # Step 4: weighted average of all numeric columns
        averaged = weighted_average_row(valid, weights, numeric_cols)
        averaged["name"] = base
        averaged["_n_poses_valid"] = len(valid)
        averaged["_n_poses_blended"] = round(n_eff, 2)
        averaged["_best_pose"] = pose_names[int(np.argmax(weights))]

        # Preserve non-numeric metadata from best pose
        best_row = valid.iloc[int(np.argmax(weights))]
        if "workflow_id" in df.columns:
            averaged["workflow_id"] = best_row["workflow_id"]
        if "active" in df.columns:
            averaged["active"] = best_row.get("active", None)

        output_rows.append(averaged)
        log_lines.append("")

    # ── Assemble output DataFrame ────────────────────────────────────────────
    col_order = (
        ["name", "workflow_id", "active"]
        + ["score", "score_plus_strain", "strain"]
        + residue_cols
        + ["_n_poses_valid", "_n_poses_blended", "_best_pose"]
    )
    col_order = [c for c in col_order if c in output_rows[0]]

    out_df = pd.DataFrame(output_rows)[col_order]
    out_df.to_csv(output_path, index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    log_lines.append("=" * 78)
    log_lines.append("Summary")
    log_lines.append("=" * 78)
    log_lines.append(f"  Input ligands:              {n_ligands}")
    log_lines.append(f"  Excluded (all poses bad):   {n_excluded}")
    log_lines.append(f"  Single-pose (w > 0.99):     {n_single}")
    log_lines.append(f"  Genuinely blended:          {n_blended}")
    log_lines.append(f"  Output ligands:             {len(out_df)}")
    log_lines.append(f"\n  Output written to: {output_path}")
    log_lines.append(f"  Weight log written to: {log_path}")

    summary = "\n".join(log_lines)
    print("\n" + "\n".join(log_lines[-12:]))  # print summary block to console

    with open(log_path, "w") as f:
        f.write(summary)

    print(f"\nDone. {len(out_df)} ligands written to {output_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Boltzmann ensemble averaging of multi-pose QC Score output")
    parser.add_argument("--config", default="boltzmann_config.json",
                        help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)
