"""Supervised EmbedKit refinement + KNN/LogReg evaluation on MHIST.

Models: ResNet18 (128-d), CONCH (512-d), UNI (1024-d)
Labels: HP=0 / SSA=1  (binary)

For each model the script:
  1. Aligns NPZ embeddings with the CSV train/val/test partition.
  2. Evaluates KNN (k=5) and Logistic Regression on raw embeddings.
  3. Fits EmbedKit in supervised mode on train+val split.
  4. Re-evaluates on refined embeddings.
  5. Prints a single comparison table (accuracy, macro-F1, ROC-AUC).

Usage:
    python examples/refine_and_evaluate_mhist.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from embedkit import EmbedKit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / "mhist"
CSV_PATH = DATA_DIR / "annotations_train_val_test.csv"
MODELS = {
    "ResNet18": DATA_DIR / "features_mhist_resnet18.npz",
    "CONCH":    DATA_DIR / "features_mhist_conch.npz",
    "UNI":      DATA_DIR / "features_mhist_uni.npz",
}
KNN_K      = 5
LR_MAX_ITER = 1000
EPOCHS     = 150
RANDOM_STATE = 42
OUT_DIR    = Path("mhist_analysis")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_data(npz_path: Path, csv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, y_labels, partition_array) aligned with CSV."""
    d = np.load(npz_path, allow_pickle=True)
    X = d["embeddings"].astype(np.float32)
    paths = d["paths"].astype(str)

    # Extract bare image filename from path (handles both 'MHIST_xxx.png' and
    # 'data/patches/train/HP/MHIST_xxx.png' styles)
    img_names = [Path(p).name for p in paths]

    lookup = csv.set_index("Image Name")[["y", "Partition"]]
    matched = lookup.reindex(img_names)

    if matched["y"].isna().any():
        n_miss = matched["y"].isna().sum()
        raise ValueError(f"{n_miss} embeddings could not be matched to CSV rows in {npz_path.name}")

    y = matched["y"].values.astype(np.int64)
    partitions = matched["Partition"].values
    return X, y, partitions


def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray,  y_test: np.ndarray,
             label: str) -> dict:
    """Fit KNN and LogReg; return metrics dict."""

    # Standardize for LogReg
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    results = {}

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    knn.fit(X_train, y_train)
    y_pred_knn   = knn.predict(X_test)
    y_prob_knn   = knn.predict_proba(X_test)[:, 1]
    results[f"{label}_KNN_acc"]      = accuracy_score(y_test, y_pred_knn)
    results[f"{label}_KNN_f1"]       = f1_score(y_test, y_pred_knn, average="macro")
    results[f"{label}_KNN_roc_auc"]  = roc_auc_score(y_test, y_prob_knn)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    lr.fit(X_tr_sc, y_train)
    y_pred_lr    = lr.predict(X_te_sc)
    y_prob_lr    = lr.predict_proba(X_te_sc)[:, 1]
    results[f"{label}_LR_acc"]       = accuracy_score(y_test, y_pred_lr)
    results[f"{label}_LR_f1"]        = f1_score(y_test, y_pred_lr, average="macro")
    results[f"{label}_LR_roc_auc"]   = roc_auc_score(y_test, y_prob_lr)

    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
csv = pd.read_csv(CSV_PATH)
all_rows: list[dict] = []

for model_name, npz_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    X, y, partitions = load_model_data(npz_path, csv)

    train_mask = np.isin(partitions, ["train", "val"])
    test_mask  = partitions == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    print(f"  Split → train+val: {X_train.shape[0]}  |  test: {X_test.shape[0]}")

    # ── Before refinement ────────────────────────────────────────────────
    print("  Evaluating RAW embeddings ...", end=" ", flush=True)
    t0 = time.time()
    raw_metrics = evaluate(X_train, y_train, X_test, y_test, label="raw")
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Supervised refinement ────────────────────────────────────────────
    print(f"  Fitting EmbedKit supervised (epochs={EPOCHS}) ...", flush=True)
    t0 = time.time()
    ek = EmbedKit(
        mode="supervised",
        epochs=EPOCHS,
        batch_size=256,
        lr=3e-4,
        scheduler="cosine",
        early_stopping_patience=20,
        random_state=RANDOM_STATE,
    )
    X_train_ref = ek.fit_transform(X_train, y=y_train)
    X_test_ref  = ek.transform(X_test)
    elapsed = time.time() - t0
    print(f"  Refinement done ({elapsed:.1f}s) — "
          f"raw_dim={X_train.shape[1]} → refined_dim={X_train_ref.shape[1]}")

    # ── After refinement ─────────────────────────────────────────────────
    print("  Evaluating REFINED embeddings ...", end=" ", flush=True)
    t0 = time.time()
    ref_metrics = evaluate(X_train_ref, y_train, X_test_ref, y_test, label="refined")
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Save refined embeddings ───────────────────────────────────────────
    out_npz = OUT_DIR / f"refined_{model_name.lower()}.npz"
    np.savez(out_npz, embeddings_train=X_train_ref, embeddings_test=X_test_ref,
             y_train=y_train, y_test=y_test)
    print(f"  Saved refined embeddings → {out_npz}")

    # ── Collect row ───────────────────────────────────────────────────────
    row = {"model": model_name}
    row.update(raw_metrics)
    row.update(ref_metrics)
    all_rows.append(row)


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
df = pd.DataFrame(all_rows).set_index("model")

# Compute deltas
for clf in ("KNN", "LR"):
    for metric in ("acc", "f1", "roc_auc"):
        raw_col = f"raw_{clf}_{metric}"
        ref_col = f"refined_{clf}_{metric}"
        df[f"Δ_{clf}_{metric}"] = df[ref_col] - df[raw_col]

# Friendly column labels for printing
RENAME = {
    "raw_KNN_acc":      "KNN Acc (raw)",
    "raw_KNN_f1":       "KNN F1 (raw)",
    "raw_KNN_roc_auc":  "KNN AUC (raw)",
    "refined_KNN_acc":  "KNN Acc (ref)",
    "refined_KNN_f1":   "KNN F1 (ref)",
    "refined_KNN_roc_auc": "KNN AUC (ref)",
    "Δ_KNN_acc":        "Δ KNN Acc",
    "Δ_KNN_f1":         "Δ KNN F1",
    "Δ_KNN_roc_auc":    "Δ KNN AUC",
    "raw_LR_acc":       "LR Acc (raw)",
    "raw_LR_f1":        "LR F1 (raw)",
    "raw_LR_roc_auc":   "LR AUC (raw)",
    "refined_LR_acc":   "LR Acc (ref)",
    "refined_LR_f1":    "LR F1 (ref)",
    "refined_LR_roc_auc": "LR AUC (ref)",
    "Δ_LR_acc":         "Δ LR Acc",
    "Δ_LR_f1":          "Δ LR F1",
    "Δ_LR_roc_auc":     "Δ LR AUC",
}
df_print = df.rename(columns=RENAME)

# Print table grouped by classifier
print("\n\n" + "="*70)
print("  MHIST SUPERVISED REFINEMENT — RESULTS COMPARISON".center(70))
print("="*70)

for clf_label in ("KNN", "LR"):
    clf_long = "K-Nearest Neighbors (k=5)" if clf_label == "KNN" else "Logistic Regression"
    print(f"\n  ── {clf_long} ──")
    cols = [
        f"{clf_label} Acc (raw)", f"{clf_label} Acc (ref)", f"Δ {clf_label} Acc",
        f"{clf_label} F1 (raw)",  f"{clf_label} F1 (ref)",  f"Δ {clf_label} F1",
        f"{clf_label} AUC (raw)", f"{clf_label} AUC (ref)", f"Δ {clf_label} AUC",
    ]
    sub = df_print[cols].copy()
    print(sub.round(4).to_string())

# CSV export
csv_out = OUT_DIR / "refinement_results.csv"
df.round(4).to_csv(csv_out)
print(f"\n  Saved full table → {csv_out}")
print("\nDone.")
