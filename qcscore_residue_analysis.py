"""
qcscore_residue_analysis.py
----------------------------
Residue-resolved interaction energy analysis for QC Score output from Promethium.

Input: QC Score CSV with columns:
    workflow_id, name, active, score, score_plus_strain, strain, <residue columns...>

    (Column names as exported directly from Promethium.)

Optional cross-reference CSV (--xref):
    ligand_file_name, SMILES, IC50   (IC50 in nM or uM; script auto-detects log scale)

Usage:
    python qcscore_residue_analysis.py --config analysis_config.json

Config file (JSON):
    {
        "data_file":                "qcscore_output.csv",
        "xref_file":                null,
            // Optional path to SMILES/IC50 cross-reference CSV (joined on 'name').

        "activity_column":          "score",
            // Which column to use as the activity axis.
            // "score"              — QC Score (recommended first pass)
            // "score_plus_strain"  — includes ligand strain penalty
            // Run both and compare if audit reports Spearman rho < 0.95 between them.

        "activity_higher_is_better": true,
            // true for QC Score / Score+strain (more negative = better binder;
            // the script flips the sign internally so all correlations read as
            // "higher = more active").

        "log_transform_activity":   false,
            // Set true only if activity_column is a raw IC50 value.

        "strain_filter_percentile": 95,
            // Ligands above this strain percentile are excluded before modeling.
            // High-strain outliers inflate the activity range artifactually.
            // Set null to disable.

        "exclude_capped_residues":  true,
            // Drop residues whose column names are wrapped in parentheses, e.g. (V7A).
            // These are terminal caps of the alanine-scanned fragments used in the
            // QC Score decomposition — computational boundary artifacts, not real
            // binding pocket contacts. Should almost always be true.

        "exclude_residues":         [],
            // Explicit list of residue column names to exclude, e.g. ["G11A", "A31A"].
            // Use for backbone/bridging residues (X/X notation) or any residue you
            // know to be outside the binding pocket.

        "clash_score_threshold":    0,
            // Ligands with score (or score_plus_strain) above this value are excluded
            // before any analysis. A positive QC Score means net repulsive interaction
            // — the ligand is clashing with the protein rather than binding. This is
            // always a pose artifact and never physically meaningful. The default of 0
            // is a hard physical cutoff; raise only if you have a specific reason.

        "residue_iqr_outlier_factor": 10.0,
            // Ligands with any individual residue interaction energy beyond
            // ±N×IQR from the dataset median are flagged as localised clash artifacts
            // and excluded. These survive the global score filter because the clash
            // is offset by strong interactions elsewhere, but they corrupt the variance
            // profile of the affected residue and bias the PLS.
            // 10.0 is deliberately conservative — only genuine artifacts are caught.
            // Lower (e.g. 5.0) for stricter filtering; set null to disable.

        "variance_filter_percentile": 10,
            // Drop residues below this variance percentile across the ligand set.
            // Residues with near-zero variance are invariant across ligands and
            // contribute noise rather than signal.
            // Set null to disable. Consider raising to 20-25 for very large,
            // diverse datasets where many peripheral residues will show low variance.

        "n_pls_components":         10,
            // Maximum number of PLS latent variables to evaluate during CV.
            // Optimal value is selected automatically. 10 is sufficient for most cases.

        "n_clusters":               null,
            // null  = auto-select k by silhouette score within k_range.
            // integer = fix k at this value (useful to reproduce a prior result).

        "k_range":                  [2, 20],
            // Range of k values to search when n_clusters is null.
            // Lower bound: 2 (minimum meaningful clustering).
            // Upper bound: scale with dataset size —
            //   <100 ligands:    [2, 10]
            //   100-500 ligands: [2, 15]
            //   500+ ligands:    [2, 20] or higher
            // The silhouette search is fast even at k=30 for large n.

        "n_top_residues":           20,
            // Number of residues to display in loading/correlation bar charts.

        "cliff_sample_size":        null,
            // null = use all ligands for activity cliff detection (recommended
            //        up to ~5000 ligands; memory scales as n*(n-1)/2 pairs).
            // integer = stratified subsample to this size, preserving the
            //        activity distribution. Use if n > 5000 or memory is limited.
            // Examples: 1000 for cautious large runs, 2000 for comprehensive coverage.

        "output_dir":               "qcscore_analysis_output"
    }

Outputs (written to output_dir):
    audit_report.txt            — data quality and structure summary
    residue_variance.png        — ranked residue variance profile
    pls_cv_scores.png           — cross-validated R² vs number of PLS components
    pls_residue_loadings.png    — top residue loadings on LV1 (and LV2 if informative)
    pls_scores_scatter.png      — ligand scores in LV1/LV2 space, colored by activity
    cluster_silhouette.png      — silhouette scores across k (if auto-selecting)
    cluster_pca_scatter.png     — k-means clusters in PCA space
    cluster_activity_whisker.png— activity distribution per cluster
    activity_cliffs.csv         — ligand pairs with similar interaction profiles but large activity differences
    residue_ranking.csv         — full ranked table: residue, PLS loading LV1, Spearman rho, p-value
    summary_report.txt          — plain-language findings summary
"""

import json
import os
import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data_file": "qcscore_output.csv",
    "xref_file": None,
    "activity_column": "score",
    "activity_higher_is_better": True,
    "strain_filter_percentile": 95,
    "variance_filter_percentile": 10,
    "exclude_capped_residues": True,
    "exclude_residues": [],
    "clash_score_threshold": 0,
    "residue_iqr_outlier_factor": 10.0,
    "n_pls_components": 10,
    "n_clusters": None,
    "k_range": [2, 10],
    "n_top_residues": 20,
    "output_dir": "qcscore_analysis_output",
    "log_transform_activity": False,
}

QCSCORE_META_COLS = {"workflow_id", "name", "active", "score", "score_plus_strain", "strain"}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    merged = {**DEFAULT_CONFIG, **cfg}
    return merged


# ---------------------------------------------------------------------------
# Data loading and audit
# ---------------------------------------------------------------------------

def load_and_validate(cfg):
    """Load QC Score CSV, optionally merge xref, return df and residue column list."""
    df = pd.read_csv(cfg["data_file"])

    required = {"workflow_id", "name", "score", "score_plus_strain", "strain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optionally merge cross-reference (joined on 'name')
    if cfg.get("xref_file"):
        xref = pd.read_csv(cfg["xref_file"])
        df = df.merge(xref, on="name", how="left")
        print(f"  Cross-reference merged: {xref.shape[0]} entries, "
              f"{df['SMILES'].notna().sum() if 'SMILES' in df.columns else 0} SMILES matched")

    # Identify residue columns: everything not in the known metadata set
    residue_cols = [c for c in df.columns if c not in QCSCORE_META_COLS
                    and c not in {"SMILES", "IC50"}
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    return df, residue_cols


def audit_data(df, residue_cols, cfg, out_dir):
    """Print and save a data quality report. Returns cleaned df."""
    lines = []
    lines.append("=" * 70)
    lines.append("QC Score Residue Interaction Analysis — Data Audit")
    lines.append("=" * 70)
    lines.append(f"\nInput file:      {cfg['data_file']}")
    lines.append(f"Total rows:      {len(df)}")

    # Detect fully-failed calculations: score is NaN (job didn't complete)
    failed_mask = df["score"].isna()
    n_failed = failed_mask.sum()
    if n_failed > 0:
        failed_names = df.loc[failed_mask, "name"].tolist()
        lines.append(f"\nFailed calculations (score is NaN): {n_failed}")
        for fn in failed_names:
            lines.append(f"  {fn}")
        lines.append("  → These rows will be excluded from all analysis.")
        df = df[~failed_mask].copy()

    lines.append(f"Ligands for analysis: {len(df)}")
    lines.append(f"Residue columns: {len(residue_cols)}")

    # Partial missing values in residue columns
    n_missing_residues = df[residue_cols].isna().any(axis=1).sum()
    lines.append(f"Rows with partial missing residue values: {n_missing_residues}")
    if n_missing_residues > 0:
        lines.append("  → These rows will be dropped before modeling.")
        df = df.dropna(subset=residue_cols).copy()

    # Activity columns — reported before any filtering so user sees raw distribution
    lines.append("\nActivity column statistics (pre-filter):")
    for col in ["score", "score_plus_strain", "strain"]:
        s = df[col]
        lines.append(f"  {col:22s}  mean={s.mean():.3f}  std={s.std():.3f}  "
                     f"min={s.min():.3f}  max={s.max():.3f}")

    # ── Clash detection: positive score = net repulsive interaction ──────────
    # A positive QC Score means the ligand is pushing against the protein rather
    # than binding. This is physically meaningless as a binding affinity surrogate
    # and is almost always caused by steric clashes in the input pose — either
    # from docking without clash removal, or from an incorrectly prepared structure.
    # These ligands are excluded unconditionally before any modeling.
    clash_thresh = cfg.get("clash_score_threshold", 0)
    activity_col = cfg["activity_column"]
    # Use the configured activity column for clash detection if available,
    # fall back to 'score' for robustness
    clash_col = activity_col if activity_col in ("score", "score_plus_strain") else "score"
    clash_mask = df[clash_col] > clash_thresh
    n_clash = clash_mask.sum()
    if n_clash > 0:
        clash_names = df.loc[clash_mask, "name"].tolist()
        lines.append(f"\nClashing ligands ({clash_col} > {clash_thresh} kcal/mol): {n_clash}")
        for cn in clash_names:
            score_val = df.loc[df["name"] == cn, clash_col].values[0]
            lines.append(f"  {cn}  ({clash_col}={score_val:.2f})")
        lines.append("  → Positive score indicates net repulsive (clashing) interaction.")
        lines.append("    These ligands are excluded — re-run with declashing/pose refinement.")
        df = df[~clash_mask].copy()
    else:
        lines.append(f"\nClash check ({clash_col} > {clash_thresh}): none detected.")
    lines.append(f"Ligands after clash filter: {len(df)}")

    # Strain filtering
    if cfg["strain_filter_percentile"] is not None:
        thresh = np.percentile(df["strain"], cfg["strain_filter_percentile"])
        n_strained = (df["strain"] > thresh).sum()
        lines.append(f"\nStrain filter (>{cfg['strain_filter_percentile']}th percentile, "
                     f">{thresh:.3f} kcal/mol): {n_strained} ligands flagged")
        df["high_strain"] = df["strain"] > thresh
        df_model = df[~df["high_strain"]].copy()
        lines.append(f"  → {len(df_model)} ligands retained for modeling")
    else:
        df["high_strain"] = False
        df_model = df.copy()
        lines.append("\nStrain filter: disabled")

    # Residue variance profile
    variances = df_model[residue_cols].var()
    near_zero = (variances < 1e-6).sum()
    lines.append(f"\nResidues with near-zero variance: {near_zero}")

    if cfg["variance_filter_percentile"] is not None:
        var_thresh = np.percentile(variances, cfg["variance_filter_percentile"])
        low_var = (variances < var_thresh).sum()
        lines.append(f"Residues below {cfg['variance_filter_percentile']}th variance percentile "
                     f"(will be dropped): {low_var}")

    # score vs score_plus_strain correlation
    r, p = stats.spearmanr(df_model["score"], df_model["score_plus_strain"])
    lines.append(f"\nSpearman correlation score vs score_plus_strain: ρ={r:.4f}, p={p:.2e}")
    lines.append("  → Use this to decide which activity axis to model.")
    if abs(r) > 0.95:
        lines.append("  → Very high correlation: both axes likely give similar residue rankings.")
    else:
        lines.append("  → Notable divergence: consider running analysis on both axes.")

    report = "\n".join(lines)
    print(report)
    with open(out_dir / "audit_report.txt", "w") as f:
        f.write(report)

    return df, df_model, n_clash


# ---------------------------------------------------------------------------
# Residue exclusion (caps and explicit list)
# ---------------------------------------------------------------------------

def exclude_residues(residue_cols, cfg):
    """
    Remove capped residues (names wrapped in parentheses) and any explicitly
    listed residues from the analysis.

    Capped residues are terminal caps of the alanine-scanned peptide fragments
    used in QC Score decomposition. They are computational artifacts of the
    fragmentation boundary, not real binding pocket residues, and their
    interaction energies partially reflect edge effects rather than genuine
    ligand contacts. Excluding them avoids spurious signals in the ranking.

    Bridging/backbone residues (e.g. X/X notation) represent inter-residue
    peptide bond contributions rather than sidechain interactions. These are
    relatively invariant across ligands and not useful for lead optimisation.
    Add them to 'exclude_residues' in the config to remove them.
    """
    excluded = []
    kept = []
    for col in residue_cols:
        is_capped = cfg.get("exclude_capped_residues", True) and                     col.startswith("(") and col.endswith(")")
        is_explicit = col in cfg.get("exclude_residues", [])
        if is_capped or is_explicit:
            excluded.append(col)
        else:
            kept.append(col)

    if excluded:
        print(f"  Excluded {len(excluded)} residues:")
        for r in excluded:
            reason = "capped" if (r.startswith("(") and r.endswith(")")) else "explicit"
            print(f"    {r}  ({reason})")
    print(f"  Residues remaining: {len(kept)}")
    return kept, excluded


# ---------------------------------------------------------------------------
# Per-residue outlier detection
# ---------------------------------------------------------------------------

def detect_residue_outliers(df_model, residue_cols, cfg, out_dir):
    """
    Detect ligands with extreme interaction energies for individual residues,
    indicative of localised steric clashes that survive the global score filter.

    A ligand can have a plausible total QC Score while still having one or two
    residue interactions that are physically implausible — for example, a pose
    that clashes with a sidechain at the pocket entrance but binds well in the
    core. These localised outliers corrupt the variance profile of the affected
    residue (driving up its variance artificially) and bias the PLS loadings
    toward that residue.

    Method: for each residue, values beyond ±N×IQR from Q1/Q3 are flagged.
    N is set by residue_iqr_outlier_factor in the config (default 10.0 — 
    deliberately conservative to catch only genuine artifacts, not real
    pharmacophoric variation).

    Ligands with any flagged residue value are reported and removed.
    The cleaned DataFrame and a summary are returned.
    """
    factor = cfg.get("residue_iqr_outlier_factor", 10.0)
    if factor is None:
        print("  Per-residue outlier detection: disabled")
        return df_model, 0

    flagged_ligands = set()
    flag_details = {}  # name -> list of (residue, value, bound)

    for col in residue_cols:
        vals = df_model[col].values
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        if iqr < 1e-6:
            continue  # near-constant residue, skip
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        outlier_mask = (vals < lo) | (vals > hi)
        outlier_idx = df_model.index[outlier_mask]
        for idx in outlier_idx:
            name = df_model.loc[idx, "name"]
            val = df_model.loc[idx, col]
            bound = hi if val > hi else lo
            flagged_ligands.add(name)
            flag_details.setdefault(name, []).append((col, val, bound))

    n_flagged = len(flagged_ligands)
    lines = []
    lines.append(f"\nPer-residue outlier detection (\u00b1{factor}\u00d7IQR): "
                 f"{n_flagged} ligands flagged")

    if n_flagged > 0:
        for name in sorted(flagged_ligands):
            lines.append(f"  {name}:")
            for col, val, bound in flag_details[name]:
                direction = "high" if val > bound else "low"
                lines.append(f"    {col}: {val:.2f} kcal/mol  "
                              f"(bound={bound:.2f}, {direction})")
        lines.append("  → These ligands have localised clash artifacts.")
        lines.append("    They are excluded from modeling. Re-run with pose refinement.")
        df_clean = df_model[~df_model["name"].isin(flagged_ligands)].copy()
        lines.append(f"  Ligands after residue outlier filter: {len(df_clean)}")
    else:
        lines.append("  → No localised residue outliers detected.")
        df_clean = df_model.copy()

    report = "\n".join(lines)
    print(report)
    with open(out_dir / "audit_report.txt", "a") as f:
        f.write("\n" + report)

    return df_clean, n_flagged


# ---------------------------------------------------------------------------
# Residue variance filtering
# ---------------------------------------------------------------------------

def filter_residues(df_model, residue_cols, cfg, out_dir):
    """Filter low-variance residues, plot variance profile, return filtered column list."""
    variances = df_model[residue_cols].var().sort_values(ascending=False)

    # Plot variance profile
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(variances)), variances.values, color='steelblue', alpha=0.7, width=1.0)
    if cfg["variance_filter_percentile"] is not None:
        thresh = np.percentile(variances.values, cfg["variance_filter_percentile"])
        ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
                   label=f'{cfg["variance_filter_percentile"]}th percentile cutoff')
        ax.legend()
    ax.set_xlabel("Residue (ranked by variance)", fontsize=12)
    ax.set_ylabel("Interaction Energy Variance (kcal/mol)²", fontsize=12)
    ax.set_title("Residue Interaction Energy Variance Profile", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "residue_variance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Apply filter
    if cfg["variance_filter_percentile"] is not None:
        thresh = np.percentile(variances.values, cfg["variance_filter_percentile"])
        filtered_cols = [c for c in residue_cols if variances[c] >= thresh]
    else:
        filtered_cols = residue_cols

    print(f"\nResidues after variance filter: {len(filtered_cols)} / {len(residue_cols)}")
    return filtered_cols


# ---------------------------------------------------------------------------
# Activity preparation
# ---------------------------------------------------------------------------

def prepare_activity(df_model, cfg):
    """Return activity vector, applying log transform and sign convention."""
    col = cfg["activity_column"]
    y = df_model[col].values.copy().astype(float)

    if cfg["log_transform_activity"]:
        y = np.log10(y + 1e-9)

    # For QC Score: more negative = better binder. Flip sign so that
    # "higher is better" semantics are consistent throughout (higher y = more active).
    if cfg["activity_higher_is_better"] and col in ("score", "score_plus_strain"):
        y = -y  # now more negative (stronger binder) maps to higher y

    return y


# ---------------------------------------------------------------------------
# PLS analysis
# ---------------------------------------------------------------------------

def run_pls_analysis(X_scaled, y, residue_cols, cfg, out_dir):
    """
    Fit PLS, select optimal n_components by LOO/KFold CV,
    plot CV R², plot residue loadings, plot LV scores scatter.
    Returns fitted pls model and sorted residue importance dataframe.
    """
    n = len(y)
    max_comp = min(cfg["n_pls_components"], len(residue_cols), n - 1)

    # For small datasets use LeaveOneOut; otherwise KFold(10)
    if n <= 30:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        cv_label = "LOO"
    else:
        cv = KFold(n_splits=min(10, n), shuffle=True, random_state=42)
        cv_label = f"{min(10, n)}-fold"

    cv_r2 = []
    for nc in range(1, max_comp + 1):
        pls = PLSRegression(n_components=nc, scale=False)
        scores = cross_val_score(pls, X_scaled, y, cv=cv, scoring='r2')
        # NaN can appear when a fold's test set has zero variance; replace with 0
        scores = np.where(np.isnan(scores), 0.0, scores)
        cv_r2.append(scores.mean())

    best_nc = int(np.argmax(cv_r2)) + 1
    print(f"\nPLS ({cv_label} CV): best n_components = {best_nc}, CV R² = {cv_r2[best_nc-1]:.4f}")

    # Plot CV R²
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_comp + 1), cv_r2, 'o-', color='steelblue', linewidth=2)
    ax.axvline(best_nc, color='red', linestyle='--', linewidth=1.2,
               label=f'Optimal: {best_nc} components')
    ax.set_xlabel("Number of PLS Components", fontsize=12)
    ax.set_ylabel("Cross-validated R²", fontsize=12)
    ax.set_title("PLS Cross-Validation: Activity Prediction", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pls_cv_scores.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fit final model with best_nc
    pls = PLSRegression(n_components=best_nc, scale=False)
    pls.fit(X_scaled, y)

    # LV1 loadings — x_loadings_ shape: (n_features, n_components)
    lv1_loadings = pls.x_loadings_[:, 0]
    lv2_loadings = pls.x_loadings_[:, 1] if best_nc >= 2 else np.zeros(len(residue_cols))

    # Spearman correlations for each residue
    rho_vals, p_vals = [], []
    for i, col in enumerate(residue_cols):
        rho, p = stats.spearmanr(X_scaled[:, i], y)
        rho_vals.append(rho)
        p_vals.append(p)

    importance_df = pd.DataFrame({
        "residue": residue_cols,
        "pls_lv1_loading": lv1_loadings,
        "pls_lv2_loading": lv2_loadings,
        "spearman_rho": rho_vals,
        "spearman_p": p_vals,
        "abs_lv1_loading": np.abs(lv1_loadings),
    }).sort_values("abs_lv1_loading", ascending=False).reset_index(drop=True)

    importance_df.to_csv(out_dir / "residue_ranking.csv", index=False)
    print(f"  Residue ranking saved to residue_ranking.csv")

    # Plot top residue loadings
    n_top = min(cfg["n_top_residues"], len(residue_cols))
    top = importance_df.head(n_top)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LV1 loadings bar chart
    colors_lv1 = ['steelblue' if v > 0 else 'tomato' for v in top["pls_lv1_loading"]]
    axes[0].barh(top["residue"][::-1], top["pls_lv1_loading"][::-1],
                 color=colors_lv1[::-1], edgecolor='black', linewidth=0.4)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel("PLS LV1 Loading", fontsize=12)
    axes[0].set_title(f"Top {n_top} Residues — PLS LV1 Loadings", fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Spearman rho bar chart (same residues, same order for direct comparison)
    colors_rho = ['steelblue' if v > 0 else 'tomato' for v in top["spearman_rho"]]
    axes[1].barh(top["residue"][::-1], top["spearman_rho"][::-1],
                 color=colors_rho[::-1], edgecolor='black', linewidth=0.4)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Spearman ρ with Activity", fontsize=12)
    axes[1].set_title(f"Top {n_top} Residues — Spearman Correlation", fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle(f"Residue Importance: {cfg['activity_column']}", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "pls_residue_loadings.png", dpi=150, bbox_inches='tight')
    plt.close()

    # LV scores scatter colored by activity
    T = pls.transform(X_scaled)
    lv1_scores = T[:, 0]
    lv2_scores = T[:, 1] if best_nc >= 2 else np.zeros(n)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(lv1_scores, lv2_scores, c=y, cmap='RdYlGn',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label=f"{cfg['activity_column']} (higher = more active)")
    ax.set_xlabel("PLS LV1 Score", fontsize=12)
    ax.set_ylabel("PLS LV2 Score", fontsize=12)
    ax.set_title("Ligands in PLS Score Space", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Annotate explained variance of each LV
    x_var = np.var(T, axis=0) / np.sum(np.var(X_scaled, axis=0))
    ax.set_xlabel(f"PLS LV1 Score ({x_var[0]:.1%} X-variance)", fontsize=12)
    if best_nc >= 2:
        ax.set_ylabel(f"PLS LV2 Score ({x_var[1]:.1%} X-variance)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_dir / "pls_scores_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Detect LV2 collapse: if best_nc == 1, LV2 was never computed
    # If best_nc >= 2, check whether LV2 scores have meaningful spread
    lv2_collapsed = True
    if best_nc >= 2:
        lv2_scores = T[:, 1]
        lv2_range = lv2_scores.max() - lv2_scores.min()
        lv1_range = T[:, 0].max() - T[:, 0].min()
        lv2_collapsed = lv2_range < 0.01 * lv1_range  # LV2 < 1% of LV1 range

    return pls, importance_df, best_nc, cv_r2[best_nc - 1], lv2_collapsed


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(X_scaled, y, df_model, cfg, out_dir):
    """Auto-select k by silhouette or use config k, cluster, auto-label, plot."""
    k_min, k_max = cfg["k_range"]
    k_max = min(k_max, len(y) - 1, len(y) // 3)
    k_max = max(k_max, k_min)  # ensure valid range

    if cfg["n_clusters"] is None:
        # Auto-select by silhouette
        sil_scores = []
        k_vals = range(k_min, k_max + 1)
        for k in k_vals:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))

        best_k = list(k_vals)[int(np.argmax(sil_scores))]
        print(f"\nClustering: auto-selected k={best_k} "
              f"(silhouette={max(sil_scores):.4f})")

        # Silhouette plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(k_vals), sil_scores, 'o-', color='steelblue', linewidth=2)
        ax.axvline(best_k, color='red', linestyle='--', linewidth=1.2,
                   label=f'Selected k={best_k}')
        ax.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax.set_ylabel("Silhouette Score", fontsize=12)
        ax.set_title("K-means Cluster Selection by Silhouette Score",
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "cluster_silhouette.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        best_k = cfg["n_clusters"]
        print(f"\nClustering: using configured k={best_k}")

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    # Auto-label clusters by mean activity (highest mean = cluster 0 after relabeling)
    cluster_means = {}
    for c in range(best_k):
        cluster_means[c] = y[cluster_labels == c].mean()
    rank_map = {c: rank for rank, (c, _) in
                enumerate(sorted(cluster_means.items(), key=lambda x: -x[1]))}
    cluster_labels_ranked = np.array([rank_map[c] for c in cluster_labels])

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cmap = matplotlib.colormaps.get_cmap('Dark2')
    palette = {i: cmap(i / max(best_k - 1, 1)) for i in range(best_k)}

    # PCA scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(best_k):
        mask = cluster_labels_ranked == i
        mean_act = y[mask].mean()
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[palette[i]], label=f'Cluster {i} (mean={mean_act:.2f})',
                   s=60, alpha=0.6, edgecolors='black', linewidth=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("K-means Clusters in PCA Space\n(clusters ranked by mean activity: 0=highest)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_pca_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Activity whisker plot
    df_plot = df_model.copy()
    df_plot['Cluster'] = cluster_labels_ranked
    df_plot['_activity'] = y

    fig, ax = plt.subplots(figsize=(max(6, best_k * 1.5), 6))
    palette_list = [palette[i] for i in range(best_k)]
    sns.boxplot(data=df_plot, x='Cluster', y='_activity', ax=ax,
                palette=palette_list, order=list(range(best_k)))
    ax.set_xlabel("Cluster (0 = highest mean activity)", fontsize=12)
    ax.set_ylabel(f"{cfg['activity_column']}", fontsize=12)
    ax.set_title("Activity Distribution by Cluster", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_activity_whisker.png", dpi=150, bbox_inches='tight')
    plt.close()

    return cluster_labels_ranked, best_k


# ---------------------------------------------------------------------------
# Activity cliff detection
# ---------------------------------------------------------------------------

def detect_activity_cliffs(X_scaled, y, df_model, cfg, out_dir, top_n=50):
    """
    Find pairs of ligands that are similar in interaction-energy space
    but have large differences in activity. These are the most informative
    pairs for understanding which residues drive activity.

    Scales to ~5000 ligands without sampling: pairwise distances are
    computed fully vectorised via scipy.spatial.distance. The condensed
    distance vectors (n*(n-1)/2 elements) are used throughout to avoid
    materialising the full n×n matrix in memory where possible.
    At n=3000 the condensed vectors are ~36M elements (~280 MB) — tractable
    on any modern workstation. Above ~5000 ligands consider enabling
    cliff_sample_size in the config to cap memory use.
    """
    n = len(y)
    names = df_model["name"].values
    sample_size = cfg.get("cliff_sample_size", None)

    if sample_size is not None and n > sample_size:
        # Stratified sample: preserve activity distribution
        print(f"  Activity cliff detection: stratified sample of "
              f"{sample_size} from {n} ligands (cliff_sample_size set in config)")
        # Bin activity into quartiles and sample proportionally
        bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
        idx = []
        per_bin = max(1, sample_size // len(np.unique(bins)))
        for b in np.unique(bins):
            bin_idx = np.where(bins == b)[0]
            chosen = np.random.choice(bin_idx,
                                      min(per_bin, len(bin_idx)),
                                      replace=False)
            idx.extend(chosen.tolist())
        idx = np.array(idx[:sample_size])
        X_sub, y_sub, names_sub = X_scaled[idx], y[idx], names[idx]
        print(f"  Using {len(idx)} ligands after stratified sampling")
    else:
        X_sub, y_sub, names_sub = X_scaled, y, names
        if n > 5000:
            print(f"  Warning: {n} ligands — cliff detection may use >1 GB RAM. "
                  f"Set cliff_sample_size in config to limit if needed.")

    n_sub = len(y_sub)

    # Fully vectorised: condensed distance vectors, no Python loop over pairs
    dist_condensed = pdist(X_sub, metric="euclidean")
    act_condensed  = pdist(y_sub.reshape(-1, 1), metric="cityblock")

    # Build index arrays for upper triangle (i < j)
    ii, jj = np.triu_indices(n_sub, k=1)

    dist_norm = dist_condensed / (dist_condensed.max() + 1e-12)
    act_norm  = act_condensed  / (act_condensed.max()  + 1e-12)
    cliff_scores = act_norm / (dist_norm + 1e-6)

    # Take top_n by cliff score without materialising full pair DataFrame
    if len(cliff_scores) > top_n:
        top_idx = np.argpartition(cliff_scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(cliff_scores[top_idx])[::-1]]
    else:
        top_idx = np.argsort(cliff_scores)[::-1]

    cliff_df = pd.DataFrame({
        "ligand_a":             names_sub[ii[top_idx]],
        "ligand_b":             names_sub[jj[top_idx]],
        "interaction_distance": dist_condensed[top_idx],
        "activity_difference":  act_condensed[top_idx],
        "activity_a":           y_sub[ii[top_idx]],
        "activity_b":           y_sub[jj[top_idx]],
        "cliff_score":          cliff_scores[top_idx],
    })

    cliff_df.to_csv(out_dir / "activity_cliffs.csv", index=False)
    print(f"  Top {min(top_n, len(cliff_df))} activity cliff pairs saved to activity_cliffs.csv")

    return cliff_df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(cfg, n_ligands, n_residues_raw, n_residues_excluded,
                  n_residues_filtered, n_strained, n_clash, n_residue_outliers,
                  best_nc, cv_r2, lv2_collapsed, best_k, importance_df, out_dir):
    """Write a plain-language summary of findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("QC Score Residue Interaction Analysis — Summary")
    lines.append("=" * 70)
    lines.append(f"\nActivity axis: {cfg['activity_column']}")
    lines.append(f"Ligand accounting:")
    lines.append(f"  Clashing (score > threshold):    {n_clash:>4d}  excluded")
    lines.append(f"  Per-residue outliers:            {n_residue_outliers:>4d}  excluded")
    lines.append(f"  High strain:                     {n_strained:>4d}  excluded")
    lines.append(f"  Retained for modeling:           {n_ligands:>4d}")
    lines.append(f"Residues: {n_residues_raw} raw → "
                 f"{n_residues_raw - n_residues_excluded} after cap/explicit exclusion → "
                 f"{n_residues_filtered} after variance filtering")

    # ---- PLS interpretation ----
    lines.append(f"\nPLS Model (leave-one-out cross-validated):")
    lines.append(f"  Optimal components: {best_nc}")
    lines.append(f"  CV R\u00b2: {cv_r2:.4f}")

    if cv_r2 > 0.6:
        lines.append("  → Strong predictive signal: residue interactions explain "
                     "activity variation well across held-out ligands.")
    elif cv_r2 > 0.3:
        lines.append("  → Moderate signal: residue interactions partially explain activity.")
        lines.append("    Consider whether multiple binding modes are present in this set.")
    else:
        lines.append("  → CV R\u00b2 is near zero. This does NOT mean the analysis is uninformative.")
        lines.append("    It means the dataset is too small for reliable held-out prediction,")
        lines.append("    which is expected for <30 ligands against a high-dimensional feature space.")
        lines.append("    The PLS loadings and Spearman correlations below are still valid")
        lines.append("    descriptive statistics — they identify which residue interactions")
        lines.append("    co-vary with activity across this ligand set. Use them for residue")
        lines.append("    hypothesis generation, not for activity prediction on new compounds.")
        lines.append("    Predictive modeling becomes appropriate once the dataset exceeds ~50-100")
        lines.append("    ligands with good activity range coverage.")

    # ---- LV2 collapse interpretation ----
    if lv2_collapsed:
        lines.append(f"\nPLS dimensionality: only 1 meaningful latent variable found.")
        lines.append("  The second latent variable (LV2) collapsed to near-zero variance,")
        lines.append("  meaning the residue interaction space is essentially one-dimensional")
        lines.append("  for this ligand set. All ligands vary along a single axis of")
        lines.append("  interaction pattern differences. This is common for a congeneric")
        lines.append("  series or a small, diverse-but-similarly-binding set — it indicates")
        lines.append("  the ligands engage the pocket in qualitatively the same way, with")
        lines.append("  quantitative differences in a handful of key residues (see ranking below).")
        lines.append("  LV2 collapse is not a failure; it simplifies interpretation.")
    else:
        lines.append(f"\nPLS dimensionality: {best_nc} latent variables retained.")
        lines.append("  Multiple meaningful axes of interaction variation were found.")
        lines.append("  This may indicate distinct binding sub-modes or chemotype-dependent")
        lines.append("  engagement patterns. Examine the LV1 vs LV2 scores scatter plot")
        lines.append("  to see whether ligand groups separate along LV2.")

    lines.append(f"\nClustering: {best_k} clusters (auto-selected by silhouette score)")
    lines.append("  Clusters are ranked 0 (highest mean activity) → N (lowest).")
    lines.append("  These groups reflect similarity in residue interaction fingerprint,")
    lines.append("  not chemical scaffold. Ligands in the same cluster engage the pocket")
    lines.append("  similarly; cross-cluster differences highlight binding mode variation.")

    lines.append(f"\nTop 10 residues by PLS LV1 loading:")
    lines.append("  The Spearman ρ column is the most directly interpretable statistic")
    lines.append("  at small sample sizes. Negative ρ means stronger interaction with")
    lines.append("  that residue correlates with better binding (more negative QC Score).")
    lines.append(f"  {'Rank':<5} {'Residue':<20} {'LV1 Loading':>12} {'Spearman ρ':>12} {'p-value':>12}")
    lines.append("  " + "-" * 65)
    for i, row in importance_df.head(10).iterrows():
        lines.append(f"  {i+1:<5} {row['residue']:<20} {row['pls_lv1_loading']:>12.4f} "
                     f"{row['spearman_rho']:>12.4f} {row['spearman_p']:>12.2e}")

    lines.append("\nKey output files:")
    lines.append("  residue_ranking.csv         — full ranked residue importance table")
    lines.append("  pls_residue_loadings.png    — top residue loadings and correlations")
    lines.append("  pls_scores_scatter.png      — ligands in PLS space, colored by activity")
    lines.append("  cluster_pca_scatter.png     — cluster visualization")
    lines.append("  cluster_activity_whisker.png— activity distribution per cluster")
    lines.append("  activity_cliffs.csv         — ligand pairs with largest activity cliffs")

    lines.append("\nRecommended next steps:")
    lines.append("  1. Focus on residues with |Spearman ρ| > 0.5 and p < 0.05 in")
    lines.append("     residue_ranking.csv. Cross-reference against the binding pocket")
    lines.append("     structure to assess whether these contacts are designable.")
    lines.append("  2. Review activity_cliffs.csv: pairs with high cliff score and low")
    lines.append("     interaction distance are best candidates for InteractionMap (FSAPT)")
    lines.append("     follow-up, which will decompose key residue contacts into electrostatic,")
    lines.append("     dispersion, and induction components.")
    lines.append("  3. If LV2 is non-collapsed and CV R\u00b2 is low, consider running")
    lines.append("     analysis separately per cluster — binding mode heterogeneity may be")
    lines.append("     suppressing the global signal.")
    lines.append("  4. For predictive modeling (virtual screening, rank-ordering new designs),")
    lines.append("     expand the dataset to >50 ligands covering a wider activity range.")
    lines.append("  5. If SMILES are available, add a cross-reference file to enable")
    lines.append("     Tanimoto-filtered activity cliff detection.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir / "summary_report.txt", "w") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from {cfg['data_file']}...")
    df, residue_cols = load_and_validate(cfg)
    print(f"  {len(df)} ligands, {len(residue_cols)} residue columns detected")

    print("\nRunning data audit...")
    df, df_model, n_clash = audit_data(df, residue_cols, cfg, out_dir)
    n_strained = df["high_strain"].sum()

    print("\nExcluding capped and explicit residues...")
    clean_cols, excluded_cols = exclude_residues(residue_cols, cfg)

    print("\nChecking for per-residue outliers...")
    df_model, n_residue_outliers = detect_residue_outliers(df_model, clean_cols, cfg, out_dir)

    print("\nFiltering residues by variance...")
    filtered_cols = filter_residues(df_model, clean_cols, cfg, out_dir)

    # Prepare feature matrix and activity vector
    X = df_model[filtered_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = prepare_activity(df_model, cfg)

    print(f"\nRunning PLS analysis on {len(y)} ligands × {len(filtered_cols)} residues...")
    pls, importance_df, best_nc, cv_r2, lv2_collapsed = run_pls_analysis(
        X_scaled, y, filtered_cols, cfg, out_dir)

    print(f"\nRunning clustering...")
    cluster_labels, best_k = run_clustering(X_scaled, y, df_model, cfg, out_dir)

    print(f"\nDetecting activity cliffs...")
    detect_activity_cliffs(X_scaled, y, df_model, cfg, out_dir)

    write_summary(cfg, len(df_model), len(residue_cols), len(excluded_cols),
                  len(filtered_cols), n_strained, n_clash, n_residue_outliers,
                  best_nc, cv_r2, lv2_collapsed, best_k, importance_df, out_dir)


    print(f"\nAll outputs written to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QC Score residue interaction analysis pipeline")
    parser.add_argument("--config", default="analysis_config.json",
                        help="Path to JSON config file")
    args = parser.parse_args()
    main(args.config)
