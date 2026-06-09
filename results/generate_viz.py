#!/usr/bin/env python3
"""
Genera visualizaciones UMAP y PCA antes/después del refinamiento
para ResNet-18, CONCH y UNI sobre MHIST (HP vs SSA).
Guarda 6 PNG en ~/embedding-kit/mhist_analysis/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from embedkit.visualization.embedding_viz import EmbeddingVisualizer

OUT_DIR  = Path("/home/jjsierramo/embedding-kit/mhist_analysis")
CSV_PATH = "/home/jjsierramo/mhist/annotations_train_val_test.csv"

CLASS_COLORS = {0: "#2196F3", 1: "#FF5722"}   # azul = HP, rojo-naranja = SSA
CLASS_NAMES  = {0: "HP",      1: "SSA"}

MODELS = {
    "resnet18": {
        "display":      "ResNet-18",
        "raw_path":     "/home/jjsierramo/mhist/features_mhist_resnet18.npz",
        "refined_path": str(OUT_DIR / "refined_resnet18.npz"),
        "raw_dim":      128,
        "refined_dim":  14,
    },
    "conch": {
        "display":      "CONCH",
        "raw_path":     "/home/jjsierramo/mhist/features_mhist_conch.npz",
        "refined_path": str(OUT_DIR / "refined_conch.npz"),
        "raw_dim":      512,
        "refined_dim":  17,
    },
    "uni": {
        "display":      "UNI",
        "raw_path":     "/home/jjsierramo/mhist/features_mhist_uni.npz",
        "refined_path": str(OUT_DIR / "refined_uni.npz"),
        "raw_dim":      1024,
        "refined_dim":  20,
    },
}

# Tabla de lookup: Image Name → clase (0=HP, 1=SSA)
_csv = pd.read_csv(CSV_PATH)
_IMG2Y = dict(zip(_csv["Image Name"], _csv["y"].astype(int)))


def load_raw(path: str):
    """Carga embeddings y deriva etiquetas de clase desde paths o CSV."""
    d = np.load(path, allow_pickle=True)
    X = d["embeddings"]
    paths = d["paths"]

    # Extraer nombre de archivo limpio (e.g. "MHIST_aag.png")
    img_names = [Path(str(p)).name for p in paths]

    # Intentar mapear desde CSV; fallback a labels del npz si ya son binarias
    y = np.array([_IMG2Y.get(n, -1) for n in img_names], dtype=int)
    if (y == -1).any():
        # Si algún nombre no se encontró, usar labels del npz directamente
        raw_labels = d["labels"].astype(int)
        unique = np.unique(raw_labels)
        if set(unique).issubset({0, 1}):
            y = raw_labels
        else:
            raise ValueError(f"No se pudieron mapear todas las etiquetas en {path}")
    return X, y


def load_refined(path: str):
    d = np.load(path)
    X = np.concatenate([d["embeddings_train"], d["embeddings_test"]], axis=0)
    y = np.concatenate([d["y_train"],          d["y_test"]],          axis=0).astype(int)
    return X, y


def scatter_panel(ax, emb, labels, title):
    colors = [CLASS_COLORS[int(l)] for l in labels]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
    for k in sorted(CLASS_NAMES)
]

for method in ["umap", "pca"]:
    print(f"\n{'='*52}")
    print(f"  {method.upper()} — projecting all models")
    print(f"{'='*52}")

    viz = EmbeddingVisualizer(method=method, random_state=42)

    for key, cfg in MODELS.items():
        print(f"\n  [{cfg['display']}]")

        X_raw, y_raw = load_raw(cfg["raw_path"])
        X_ref, y_ref = load_refined(cfg["refined_path"])
        print(f"    raw   : {X_raw.shape}  |  refined: {X_ref.shape}")

        print("    projecting raw ...", end=" ", flush=True)
        emb_raw = viz._reduce(X_raw)
        print("done")

        print("    projecting refined ...", end=" ", flush=True)
        emb_ref = viz._reduce(X_ref)
        print("done")

        # ── figura ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("white")

        scatter_panel(
            axes[0], emb_raw, y_raw,
            f"Embeddings crudos  ({cfg['raw_dim']} dim)",
        )
        scatter_panel(
            axes[1], emb_ref, y_ref,
            f"Embeddings refinados  ({cfg['refined_dim']} dim)",
        )

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=2,
            fontsize=11,
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            f"{cfg['display']} — proyección {method.upper()}   |   MHIST (HP vs SSA)",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )

        out_path = OUT_DIR / f"viz_{key}_{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    → guardado: {out_path.name}")

print("\n✓ 6 figuras generadas.")
