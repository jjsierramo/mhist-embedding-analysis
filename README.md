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

[EmbedKit](https://github.com/fagonzalezo/embedding-kit) es una librería para diagnóstico y refinamiento de espacios de embeddings. Proporciona:

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

### Paso 1 — Descargar el dataset MHIST

Descarga el dataset desde [bmirds.github.io/MHIST](https://bmirds.github.io/MHIST/) y coloca los archivos en `~/mhist/`:

```
~/mhist/
├── images/                          # 3152 imágenes .png
└── annotations_train_val_test.csv   # particiones y etiquetas
```

El CSV tiene columnas `Image Name`, `Majority Vote Label` (HP/SSA) y `Partition` (train/val/test). Los scripts esperan una columna `y` con 0=HP y 1=SSA; puedes crearla así:

```python
import pandas as pd
csv = pd.read_csv("~/mhist/annotations_train_val_test.csv")
csv["y"] = (csv["Majority Vote Label"] == "SSA").astype(int)
csv.to_csv("~/mhist/annotations_train_val_test.csv", index=False)
```

---

### Paso 2 — Solicitar acceso a CONCH y UNI en HuggingFace

CONCH y UNI son modelos fundacionales del **Mahmood Lab** (Harvard) con acceso restringido para uso académico.

**Para CONCH:**
1. Visita [huggingface.co/MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch)
2. Inicia sesión en HuggingFace y haz clic en **"Access repository"**
3. Acepta los términos de uso (uso académico no comercial)
4. El acceso suele aprobarse en minutos de forma automática

**Para UNI:**
1. Visita [huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)
2. Mismo proceso: **"Access repository"** → acepta términos
3. Acceso automático para uso académico

Una vez aprobado, autentica tu sesión local:

```bash
pip install huggingface_hub
huggingface-cli login   # pega tu token de https://huggingface.co/settings/tokens
```

---

### Paso 3 — Generar embeddings con ResNet-18

ResNet-18 no requiere acceso especial. Se carga desde `torchvision` y se elimina la cabeza de clasificación para obtener embeddings de 512-d que luego se proyectan a 128-d con una capa lineal, o directamente se usan las activaciones del penúltimo bloque.

```python
"""Extrae embeddings de ResNet-18 con pesos congelados (pool global → 512-d)."""
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGES_DIR = Path.home() / "mhist" / "images"
OUT_PATH   = Path.home() / "mhist" / "features_mhist_resnet18.npz"
BATCH_SIZE = 256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo: ResNet-18 preentrenado en ImageNet, sin cabeza de clasificación
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = torch.nn.Identity()   # elimina la fc final → salida 512-d
backbone.eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MHISTDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = sorted(img_dir.glob("*.png"))
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), str(self.paths[i].name)

loader = DataLoader(MHISTDataset(IMAGES_DIR, transform), batch_size=BATCH_SIZE,
                    num_workers=4, pin_memory=True)

all_embeddings, all_paths = [], []
with torch.no_grad():
    for imgs, names in loader:
        feats = backbone(imgs.to(DEVICE)).cpu().float().numpy()
        all_embeddings.append(feats)
        all_paths.extend(names)

np.savez(OUT_PATH,
         embeddings=np.concatenate(all_embeddings).astype(np.float32),
         paths=np.array(all_paths))
print(f"Guardado: {OUT_PATH}  shape={np.load(OUT_PATH)['embeddings'].shape}")
```

> **Nota:** Los scripts de análisis esperan 128-d para ResNet-18. Si usas la capa `avgpool` (512-d) puedes agregar una capa `nn.Linear(512, 128)` sin entrenar, o simplemente usar los 512-d y actualizar la referencia en los scripts.

---

### Paso 4 — Generar embeddings con CONCH (512-d)

CONCH es un modelo visual-lingüístico basado en CoCa entrenado sobre millones de pares imagen-texto de patología. Su encoder visual produce embeddings de 512-d.

```bash
pip install git+https://github.com/mahmoodlab/CONCH.git
```

```python
"""Extrae embeddings de CONCH con pesos congelados — token [CLS] → 512-d."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from conch.open_clip_custom import create_model_from_pretrained

IMAGES_DIR = Path.home() / "mhist" / "images"
OUT_PATH   = Path.home() / "mhist" / "features_mhist_conch.npz"
BATCH_SIZE = 128    # ViT-B/16: cabe bien en 16 GB con batch 128-256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Carga modelo + transform oficial desde HuggingFace Hub
model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16",
    checkpoint_path="hf_hub:MahmoodLab/conch",
)
model.eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad = False

class MHISTDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = sorted(img_dir.glob("*.png"))
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), str(self.paths[i].name)

loader = DataLoader(MHISTDataset(IMAGES_DIR, preprocess), batch_size=BATCH_SIZE,
                    num_workers=4, pin_memory=True)

all_embeddings, all_paths = [], []
with torch.no_grad():
    for imgs, names in loader:
        # encode_image devuelve el embedding del token [CLS] del encoder visual
        # proj_contrast=False → espacio pre-proyección (más rico para downstream)
        # normalize=False     → sin L2-norm, para que EmbedKit tenga escala real
        feats = model.encode_image(imgs.to(DEVICE),
                                   proj_contrast=False,
                                   normalize=False)
        all_embeddings.append(feats.cpu().float().numpy())
        all_paths.extend(names)

np.savez(OUT_PATH,
         embeddings=np.concatenate(all_embeddings).astype(np.float32),
         paths=np.array(all_paths))
print(f"Guardado: {OUT_PATH}  shape={np.load(OUT_PATH)['embeddings'].shape}")
# Esperado: (3152, 512)
```

---

### Paso 5 — Generar embeddings con UNI (1024-d)

UNI es un modelo fundacional basado en ViT-Large entrenado con DINOv2 sobre más de 100 000 WSIs de patología. Produce embeddings de 1024-d a partir del token `[CLS]`.

```bash
pip install timm huggingface_hub
```

```python
"""Extrae embeddings de UNI con pesos congelados — token [CLS] → 1024-d."""
from pathlib import Path
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGES_DIR = Path.home() / "mhist" / "images"
OUT_PATH   = Path.home() / "mhist" / "features_mhist_uni.npz"
BATCH_SIZE = 64     # ViT-L/16: más pesado, 64 es seguro en 16 GB; bajar a 32 si OOM
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Carga UNI desde HuggingFace Hub (requiere acceso aprobado)
model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,       # inicialización LayerScale estable para ViT-L
    dynamic_img_size=True,  # permite entradas de tamaño variable
)
model.eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad = False

# Transform oficial (normalización y resize definidos en pretrained_cfg)
transform = create_transform(
    **resolve_data_config(model.pretrained_cfg, model=model)
)

class MHISTDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = sorted(img_dir.glob("*.png"))
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), str(self.paths[i].name)

loader = DataLoader(MHISTDataset(IMAGES_DIR, transform), batch_size=BATCH_SIZE,
                    num_workers=4, pin_memory=True)

all_embeddings, all_paths = [], []
with torch.no_grad():
    for imgs, names in loader:
        # forward_features devuelve todos los tokens: [B, 197, 1024]
        # (196 parches de 16×16 + 1 token [CLS] al índice 0)
        tokens = model.forward_features(imgs.to(DEVICE))
        cls_token = tokens[:, 0, :]   # [B, 1024] — token [CLS]
        all_embeddings.append(cls_token.cpu().float().numpy())
        all_paths.extend(names)

np.savez(OUT_PATH,
         embeddings=np.concatenate(all_embeddings).astype(np.float32),
         paths=np.array(all_paths))
print(f"Guardado: {OUT_PATH}  shape={np.load(OUT_PATH)['embeddings'].shape}")
# Esperado: (3152, 1024)
```

---

### Paso 6 — Instalar dependencias y ejecutar los análisis

```bash
pip install -r requirements.txt
```

```bash
# Desde la raíz del proyecto, con los embeddings ya en ~/mhist/

# 1. Análisis geométrico comparativo
python scripts/compare_mhist.py

# 2. Refinamiento supervisado + evaluación
python scripts/refine_and_evaluate_mhist.py

# 3. Refinamiento no supervisado + evaluación
python scripts/unsupervised_refine_mhist.py

# 4. Regenerar visualizaciones PCA/UMAP (requiere los .npz originales)
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
