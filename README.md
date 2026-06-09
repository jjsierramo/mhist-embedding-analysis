# Análisis y Refinamiento de Espacios de Embeddings en MHIST

Experimentos de análisis geométrico y refinamiento de embeddings para clasificación de pólipos colorrectales usando el dataset MHIST y la librería **EmbedKit v0.1.1**.

---

## Dataset: MHIST

[MHIST](https://bmirds.github.io/MHIST/) es un dataset de histopatología para clasificación binaria de pólipos colorrectales:

| Clase | Descripción | Significado clínico |
|-------|-------------|---------------------|
| **HP** (0) | Pólipo hiperplásico | Benigno, bajo riesgo |
| **SSA** (1) | Adenoma serrado sésil | Precanceroso, requiere seguimiento |

- **Total:** 3 152 imágenes de parches histológicos (224 × 224 px)
- **Partición:** train+val / test según el CSV oficial `annotations_train_val_test.csv`
- Los embeddings fueron extraídos con pesos congelados (sin fine-tuning)

---

## Modelos evaluados como extractores de features

| Modelo | Dimensión | Dominio de preentrenamiento |
|--------|-----------|-----------------------------|
| **ResNet-18** | 128-d | ImageNet (visión general) |
| **CONCH** | 512-d | Patología computacional |
| **UNI** | 1024-d | Patología computacional (grande) |

---

## Herramienta: EmbedKit v0.1.1

[EmbedKit](https://github.com/jjsierramo/embedding-kit) es una librería para diagnóstico y refinamiento de espacios de embeddings. Proporciona:

- **`EmbedKitAnalyzer`** — análisis geométrico completo (dimensión intrínseca, hubness, isotropy, concentración de distancias, etc.)
- **`EmbedKit`** — refinamiento supervisado y no supervisado mediante aprendizaje contrastivo
- **`AnalysisPlotter`** — visualizaciones PCA y UMAP de los espacios

---

## Experimentos realizados

### 1. Análisis geométrico comparativo (`scripts/compare_mhist.py`)

Diagnóstico del espacio de embeddings de cada modelo usando `EmbedKitAnalyzer`:

| Métrica | ResNet-18 | CONCH | UNI |
|---------|-----------|-------|-----|
| Dimensión intrínseca (TwoNN) | 16.86 | 9.75 | 11.32 |
| Ratio ID/dim_ambiente | 0.132 | 0.019 | 0.011 |
| Isotropy score | 0.014 | 0.052 | 0.081 |
| Hub ratio | 0.113 | 0.082 | 0.090 |
| Hub contamination | 0.464 | 0.387 | 0.387 |
| Relative contrast | 4.421 | 1.670 | 1.215 |
| Participation ratio | 1.30 | 14.02 | 26.33 |
| Severidad diagnóstica | **high** | **high** | **high** |

Los tres espacios presentan severidad alta, con patologías distintas:
- **ResNet-18:** alta concentración de distancias (`relative_contrast=4.4`), isotropy muy baja (0.014), dimensión intrínseca elevada relativa al ambient
- **CONCH y UNI:** baja isotropy, alto spectral gap, fuerte compresión de información en pocas dimensiones

### 2. Refinamiento supervisado (`scripts/refine_and_evaluate_mhist.py`)

Se entrena `EmbedKit(mode="supervised")` con `RankNContrastLoss` sobre train+val y se evalúa en test con KNN (k=5) y Regresión Logística.

| Modelo | KNN Acc raw | KNN Acc ref | Δ KNN Acc | LR Acc raw | LR Acc ref | Δ LR Acc |
|--------|-------------|-------------|-----------|------------|------------|----------|
| ResNet-18 | 0.8669 | 0.8598 | −0.0072 | 0.8649 | 0.8598 | −0.0051 |
| CONCH | 0.7165 | 0.7707 | **+0.0542** | 0.7973 | 0.7656 | −0.0317 |
| UNI | 0.7544 | 0.8280 | **+0.0737** | 0.8086 | 0.8280 | +0.0194 |

**Conclusión:** El refinamiento supervisado mejora significativamente KNN para CONCH (+5.4%) y UNI (+7.4%), pero no beneficia a ResNet-18 (ya cercano a su techo) y puede degradar LR en algunos casos.

### 3. Refinamiento no supervisado (`scripts/unsupervised_refine_mhist.py`)

Se entrena `EmbedKit(mode="unsupervised")` sin etiquetas (SimCLR-style).

| Modelo | Δ KNN Acc | Δ LR Acc |
|--------|-----------|----------|
| ResNet-18 | −0.0082 | −0.0573 |
| CONCH | −0.0246 | −0.0266 |
| UNI | −0.0583 | −0.0624 |

**Conclusión:** La contrastividad no supervisada degeneraliza los embeddings en este dataset, probablemente porque las augmentaciones genéricas destruyen señal diagnóstica relevante en histopatología.

---

## Pruebas estadísticas (`results/statistical_tests.csv`)

Se aplicaron **test de McNemar** (significancia) y **Bootstrap 95% CI** (10 000 iteraciones) sobre las diferencias de accuracy en test.

| Modelo | Clasificador | Refinamiento | p-valor McNemar | CI 95% | Significativo |
|--------|-------------|--------------|-----------------|--------|---------------|
| ResNet-18 | KNN | supervisado | 0.3711 | [−0.022, 0.006] | No |
| ResNet-18 | LogReg | **no supervisado** | **<0.001** | [−0.080, −0.036] | **Sí (degradación)** |
| CONCH | KNN | **supervisado** | **0.0002** | [0.026, 0.082] | **Sí (mejora)** |
| CONCH | LogReg | **supervisado** | **0.0102** | [−0.057, −0.009] | **Sí (degradación)** |
| UNI | KNN | **supervisado** | **<0.001** | [0.046, 0.101] | **Sí (mejora)** |
| UNI | KNN | **no supervisado** | **0.0002** | [−0.089, −0.029] | **Sí (degradación)** |
| UNI | LogReg | **no supervisado** | **<0.001** | [−0.089, −0.033] | **Sí (degradación)** |

---

## Estructura del repositorio

```
mhist_embedkit_experiments/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   ├── compare_mhist.py              # Análisis geométrico comparativo
│   ├── refine_and_evaluate_mhist.py  # Refinamiento supervisado + evaluación
│   └── unsupervised_refine_mhist.py  # Refinamiento no supervisado + evaluación
└── results/
    ├── comparison_metrics.csv        # Métricas geométricas de los 3 modelos
    ├── refinement_results.csv        # Resultados supervisados (raw vs refined)
    ├── unsupervised_results.csv      # Resultados no supervisados
    ├── statistical_tests.csv         # Tests de McNemar + Bootstrap CIs
    ├── analysis_resnet18.png         # Radar plot diagnóstico ResNet-18
    ├── analysis_conch.png            # Radar plot diagnóstico CONCH
    ├── analysis_uni.png              # Radar plot diagnóstico UNI
    ├── viz_resnet18_pca.png          # PCA coloreado por clase (ResNet-18)
    ├── viz_resnet18_umap.png         # UMAP coloreado por clase (ResNet-18)
    ├── viz_conch_pca.png             # PCA coloreado por clase (CONCH)
    ├── viz_conch_umap.png            # UMAP coloreado por clase (CONCH)
    ├── viz_uni_pca.png               # PCA coloreado por clase (UNI)
    ├── viz_uni_umap.png              # UMAP coloreado por clase (UNI)
    ├── generate_viz.py               # Script para regenerar visualizaciones
    └── statistical_tests.py          # Script para reproducir pruebas estadísticas
```

---

## Cómo reproducir los experimentos

### Requisitos previos

1. Descargar el dataset MHIST desde [bmirds.github.io/MHIST](https://bmirds.github.io/MHIST/) y colocarlo en `~/mhist/`
2. Generar los embeddings con cada modelo y guardarlos como:
   - `~/mhist/features_mhist_resnet18.npz`
   - `~/mhist/features_mhist_conch.npz`
   - `~/mhist/features_mhist_uni.npz`
   
   Cada `.npz` debe contener las claves `embeddings` (float32) y `paths` (strings con nombres de imagen).

3. El CSV de particiones `annotations_train_val_test.csv` debe estar en `~/mhist/` con columnas `Image Name`, `y` (0=HP, 1=SSA) y `Partition` (train/val/test).

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Ejecución

```bash
# Desde la raíz del proyecto, con los embeddings en ~/mhist/

# 1. Análisis geométrico comparativo
python scripts/compare_mhist.py

# 2. Refinamiento supervisado + evaluación
python scripts/refine_and_evaluate_mhist.py

# 3. Refinamiento no supervisado + evaluación
python scripts/unsupervised_refine_mhist.py

# 4. Regenerar visualizaciones PCA/UMAP (requiere los .npz de embeddings)
python results/generate_viz.py

# 5. Reproducir pruebas estadísticas (requiere los .npz refinados en mhist_analysis/)
python results/statistical_tests.py
```

Los resultados se guardan en `mhist_analysis/` (directorio generado en tiempo de ejecución).

---

## Dependencias principales

| Librería | Versión mínima | Uso |
|----------|----------------|-----|
| `embedding-kit` | 0.1.1 | Análisis y refinamiento de embeddings |
| `numpy` | ≥1.24 | Operaciones con arrays |
| `pandas` | ≥2.0 | Manejo de tablas de resultados |
| `scikit-learn` | ≥1.3 | KNN, Logistic Regression, métricas |
| `torch` | ≥2.0 | Backend para EmbedKit |
| `umap-learn` | ≥0.5 | Visualizaciones UMAP |
| `matplotlib` | ≥3.7 | Gráficos |
| `seaborn` | ≥0.12 | Estilo de gráficos |

---

## Licencia

MIT — ver [LICENSE](LICENSE).
