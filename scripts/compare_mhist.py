"""Comparative EmbedKitAnalyzer on three MHIST embedding models.

Models compared:
  - ResNet18  : 3152 × 128
  - CONCH     : 3152 × 512
  - UNI       : 3152 × 1024

Usage:
    python examples/compare_mhist.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from embedkit import EmbedKitAnalyzer
from embedkit.visualization.plots import AnalysisPlotter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / "mhist"
MODELS = {
    "ResNet18": DATA_DIR / "features_mhist_resnet18.npz",
    "CONCH":    DATA_DIR / "features_mhist_conch.npz",
    "UNI":      DATA_DIR / "features_mhist_uni.npz",
}
K = 10
ID_METHODS = ["TwoNN"]   # TwoNN es el más rápido; MLE/lPCA son O(n²) y lentos en n>2k
N_MAX = 2_000            # subsample para kNN y ID (3152 → 2000 es representativo)
OUT_DIR = Path("mhist_analysis")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
def load_embeddings(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    X = d["embeddings"].astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# Run analysis on each model
# ---------------------------------------------------------------------------
reports: dict[str, object] = {}
for model_name, npz_path in MODELS.items():
    print(f"Analyzing {model_name} ({npz_path.name}) ...", flush=True)
    X = load_embeddings(npz_path)
    report = EmbedKitAnalyzer(k=K, id_methods=ID_METHODS, n_max=N_MAX).fit(X)
    reports[model_name] = report
    print(f"  done — shape={X.shape}, severity={report.severity}")


# ---------------------------------------------------------------------------
# Side-by-side comparison table
# ---------------------------------------------------------------------------
def extract_metrics(report) -> dict:
    hub = report.hubness
    iso = report.geometry.isotropy
    dc  = report.geometry.distance_concentration
    nc  = report.geometry.neighbor_consistency
    ker = report.kernel
    n, d = report.input_shape

    return {
        # Shape
        "n_samples":            n,
        "ambient_dim":          d,
        # Intrinsic dimension
        "ID_consensus":         round(report.intrinsic_dim.consensus,  2),
        "ID_TwoNN":             round(report.intrinsic_dim.estimates.get("TwoNN", float("nan")), 2),
        "ID_uncertainty (±)":   round(report.intrinsic_dim.uncertainty, 2),
        "ID/ambient_ratio":     round(report.intrinsic_dim.consensus / d, 3),
        # Hubness
        "k_skewness":           round(hub.k_skewness,        3),
        "robinhood_index":      round(hub.robinhood_index,   3),
        "hub_ratio":            round(hub.hub_ratio,         3),
        "antihub_ratio":        round(hub.antihub_ratio,     3),
        "hub_contamination":    round(hub.hub_contamination, 3),
        # Geometry
        "participation_ratio":  round(iso.participation_ratio, 2),
        "isotropy_score":       round(iso.isotropy_score,      3),
        "relative_contrast":    round(dc.relative_contrast,    3),
        "concentration_ratio":  round(dc.concentration_ratio,  3),
        "neighbor_consistency": round(nc.mean_consistency,     3),
        "uniformity":           round(report.geometry.uniformity.uniformity, 4),
        # Kernel
        "effective_rank":       round(ker.effective_rank, 2),
        "spectral_gap":         round(ker.spectral_gap,   3),
        "condition_number":     f"{ker.condition_number:.2e}",
        # Auto-config
        "suggested_target_dim": report.suggested_target_dim,
        "severity":             report.severity,
    }


rows = {name: extract_metrics(rep) for name, rep in reports.items()}
df = pd.DataFrame(rows)
df.index.name = "metric"

SEPARATOR = {
    "n_samples":            "--- Shape ---",
    "ID_consensus":         "--- Intrinsic Dimension ---",
    "k_skewness":           "--- Hubness ---",
    "participation_ratio":  "--- Geometry ---",
    "effective_rank":       "--- Kernel ---",
    "suggested_target_dim": "--- Auto-config ---",
}

col_w = 16
header = f"{'metric':<28}" + "".join(f"{m:>{col_w}}" for m in df.columns)
rule   = "-" * len(header)

print("\n")
print("=" * len(header))
print("  MHIST Embedding Geometry Comparison".center(len(header)))
print("=" * len(header))
print(header)
print(rule)

for metric, row in df.iterrows():
    if metric in SEPARATOR:
        print(f"\n  {SEPARATOR[metric]}")
    vals = "".join(f"{str(v):>{col_w}}" for v in row)
    print(f"  {metric:<26}{vals}")

print(rule)


# ---------------------------------------------------------------------------
# Per-model recommendation blocks
# ---------------------------------------------------------------------------
print("\n\n  RECOMMENDATIONS")
print(rule)
for model_name, report in reports.items():
    print(f"\n  [{model_name}]  severity={report.severity}")
    for rec in report.recommendations:
        wrapped = textwrap.fill(rec, width=90, subsequent_indent="      ")
        print(f"    • {wrapped}")


# ---------------------------------------------------------------------------
# Save individual analysis plots
# ---------------------------------------------------------------------------
print("\n\nSaving per-model plots ...")
for model_name, report in reports.items():
    plotter = AnalysisPlotter(report)
    fig = plotter.plot_full_report()
    out_path = OUT_DIR / f"analysis_{model_name.lower()}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved {out_path}")

# Save comparison table as CSV
csv_path = OUT_DIR / "comparison_metrics.csv"
df.to_csv(csv_path)
print(f"  Saved {csv_path}")

print("\nDone.")
