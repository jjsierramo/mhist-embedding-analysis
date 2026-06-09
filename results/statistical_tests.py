"""Statistical significance tests for embedding refinement results on MHIST.

For each model (ResNet18, CONCH, UNI) and classifier (KNN, LogReg):
  - McNemar's test: raw vs supervised, raw vs unsupervised
  - Bootstrap 95% CI (1000 iterations): accuracy difference

Saves results to mhist_analysis/statistical_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR     = Path.home() / "mhist"
ANALYSIS_DIR = Path(__file__).parent
CSV_PATH     = DATA_DIR / "annotations_train_val_test.csv"
KNN_K        = 5
LR_MAX_ITER  = 1000
RANDOM_STATE = 42
N_BOOTSTRAP  = 1000
RNG          = np.random.default_rng(RANDOM_STATE)

MODELS = ["ResNet18", "CONCH", "UNI"]
RAW_NPZ = {
    "ResNet18": DATA_DIR / "features_mhist_resnet18.npz",
    "CONCH":    DATA_DIR / "features_mhist_conch.npz",
    "UNI":      DATA_DIR / "features_mhist_uni.npz",
}
SUP_NPZ = {m: ANALYSIS_DIR / f"refined_{m.lower()}.npz"       for m in MODELS}
UNS_NPZ = {m: ANALYSIS_DIR / f"unsup_refined_{m.lower()}.npz" for m in MODELS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw_splits(npz_path: Path, csv: pd.DataFrame):
    """Return (X_train, y_train, X_test, y_test) aligned with CSV partitions."""
    d = np.load(npz_path, allow_pickle=True)
    X = d["embeddings"].astype(np.float32)
    img_names = [Path(p).name for p in d["paths"].astype(str)]

    lookup  = csv.set_index("Image Name")[["y", "Partition"]]
    matched = lookup.reindex(img_names)

    y          = matched["y"].values.astype(np.int64)
    partitions = matched["Partition"].values

    train_mask = np.isin(partitions, ["train", "val"])
    test_mask  = partitions == "test"
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def load_refined_splits(npz_path: Path):
    """Return (X_train, y_train, X_test, y_test) from a refined NPZ."""
    d = np.load(npz_path, allow_pickle=True)
    return (
        d["embeddings_train"].astype(np.float32),
        d["y_train"].astype(np.int64),
        d["embeddings_test"].astype(np.float32),
        d["y_test"].astype(np.int64),
    )


def get_predictions(X_train, y_train, X_test):
    """Fit KNN and LogReg; return (y_pred_knn, y_pred_lr)."""
    # KNN — no scaling needed
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    knn.fit(X_train, y_train)
    y_knn = knn.predict(X_test)

    # Logistic Regression — standardize
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    lr.fit(scaler.fit_transform(X_train), y_train)
    y_lr = lr.predict(scaler.transform(X_test))

    return y_knn, y_lr


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test comparing two classifiers on the same test set.

    Returns (statistic, p_value) using the exact mid-p correction when
    discordant pairs are small, otherwise the standard chi-squared statistic.
    Uses Edwards' continuity correction for the chi-squared form.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # Contingency table
    n10 = np.sum(correct_a & ~correct_b)  # A right, B wrong
    n01 = np.sum(~correct_a & correct_b)  # A wrong, B right
    b, c = n10, n01
    n_disc = b + c

    if n_disc == 0:
        return 0.0, 1.0  # identical predictions

    # McNemar with continuity correction (Edwards)
    statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value)


def bootstrap_acc_diff(y_true, pred_a, pred_b, n_boot=N_BOOTSTRAP):
    """Bootstrap 95% CI for accuracy(B) - accuracy(A).

    Returns (observed_diff, ci_lower, ci_upper).
    """
    n = len(y_true)
    correct_a = (pred_a == y_true).astype(float)
    correct_b = (pred_b == y_true).astype(float)

    boot_diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_diffs[i] = correct_b[idx].mean() - correct_a[idx].mean()

    observed = correct_b.mean() - correct_a.mean()
    ci_lo    = float(np.percentile(boot_diffs, 2.5))
    ci_hi    = float(np.percentile(boot_diffs, 97.5))
    return float(observed), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
csv = pd.read_csv(CSV_PATH)
rows: list[dict] = []

for model in MODELS:
    print(f"\n{'='*60}\n  {model}\n{'='*60}")

    # Load all three embedding sets
    Xr_tr, yr_tr, Xr_te, yr_te = load_raw_splits(RAW_NPZ[model], csv)
    Xs_tr, ys_tr, Xs_te, ys_te = load_refined_splits(SUP_NPZ[model])
    Xu_tr, yu_tr, Xu_te, yu_te = load_refined_splits(UNS_NPZ[model])

    # Sanity-check label alignment
    assert np.array_equal(yr_te, ys_te), f"{model}: raw/supervised test labels differ!"
    assert np.array_equal(yr_te, yu_te), f"{model}: raw/unsupervised test labels differ!"
    y_true = yr_te
    print(f"  Test set: {len(y_true)} samples  "
          f"(HP={np.sum(y_true==0)}, SSA={np.sum(y_true==1)})")

    # Get predictions for each scenario
    print("  Generating predictions ...", end=" ", flush=True)
    raw_knn, raw_lr   = get_predictions(Xr_tr, yr_tr, Xr_te)
    sup_knn, sup_lr   = get_predictions(Xs_tr, ys_tr, Xs_te)
    uns_knn, uns_lr   = get_predictions(Xu_tr, yu_tr, Xu_te)
    print("done")

    for clf_name, pred_raw, pred_sup, pred_uns in [
        ("KNN",   raw_knn, sup_knn, uns_knn),
        ("LogReg", raw_lr,  sup_lr,  uns_lr),
    ]:
        for refinement, pred_ref in [("supervised", pred_sup), ("unsupervised", pred_uns)]:
            print(f"  {clf_name} | {refinement} ...", end=" ", flush=True)

            # McNemar
            stat, pval = mcnemar_test(y_true, pred_raw, pred_ref)

            # Bootstrap CI for accuracy difference (refined - raw)
            obs_diff, ci_lo, ci_hi = bootstrap_acc_diff(y_true, pred_raw, pred_ref)

            acc_raw = (pred_raw == y_true).mean()
            acc_ref = (pred_ref == y_true).mean()

            significant = pval < 0.05
            conclusion = "SIGNIFICANT" if significant else "not significant"
            ci_contains_zero = ci_lo <= 0 <= ci_hi
            print(f"Δacc={obs_diff:+.4f}  p={pval:.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  → {conclusion}")

            rows.append({
                "model":           model,
                "classifier":      clf_name,
                "refinement":      refinement,
                "acc_raw":         round(acc_raw, 4),
                "acc_refined":     round(acc_ref, 4),
                "acc_diff":        round(obs_diff, 4),
                "mcnemar_stat":    round(stat, 4),
                "mcnemar_pval":    round(pval, 4),
                "boot_ci_lower":   round(ci_lo, 4),
                "boot_ci_upper":   round(ci_hi, 4),
                "ci_contains_zero": ci_contains_zero,
                "significant_p05": significant,
                "conclusion":      f"{'SIGNIFICANT' if significant else 'not significant'} (p={pval:.4f})",
            })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)
out_csv = ANALYSIS_DIR / "statistical_tests.csv"
df.to_csv(out_csv, index=False)
print(f"\n  Saved → {out_csv}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("  STATISTICAL SIGNIFICANCE SUMMARY".center(80))
print("="*80)
print(f"{'Model':<12} {'Classifier':<10} {'Refinement':<15} "
      f"{'Δacc':>8} {'p-value':>9} {'95% CI':>22} {'Conclusion'}")
print("-"*80)
for _, r in df.iterrows():
    ci_str = f"[{r.boot_ci_lower:+.4f}, {r.boot_ci_upper:+.4f}]"
    print(f"{r.model:<12} {r.classifier:<10} {r.refinement:<15} "
          f"{r.acc_diff:>+8.4f} {r.mcnemar_pval:>9.4f} {ci_str:>22}  {r.conclusion}")
print("="*80)
