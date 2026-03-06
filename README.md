# SVAMITVA SAM2 Feature Extraction Model

> **End-to-end AI pipeline for extracting geographic features from SVAMITVA drone orthophotos using Meta's Segment Anything Model 2 (SAM2).**

## 🏗️ Architecture

```
Drone Orthophoto (GeoTIFF)
    │
    ▼
┌──────────────────┐
│ SAM2 Hiera B+    │  ← Multi-scale image encoder
│ Image Encoder    │    (pretrained, staged unfreezing)
└──────────────────┘
    │ 4 feature maps (stride 4/8/16/32)
    ▼
┌──────────────────┐
│ FPN Decoder      │  ← CBAM attention at each scale
│ + Attention      │    Top-down + lateral connections
└──────────────────┘
    │ (B, 256, H/4, W/4) unified features
    ▼
┌──────────────────┐
│ Task-Group       │  ← Shared refinement per feature group
│ Refinement       │    (buildings, roads, water, utility, infra)
└──────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ 11 Task Heads                                │
│ ┌──────────┐┌──────────┐┌──────────────────┐ │
│ │ Building ││ Road     ││ Waterbody        │ │
│ │ (mask +  ││ (polygon ││ (polygon + line  │ │
│ │ roof     ││ + center ││ + point/wells)   │ │
│ │ type)    ││ line)    ││                  │ │
│ └──────────┘└──────────┘└──────────────────┘ │
│ ┌──────────┐┌──────────┐┌──────────────────┐ │
│ │ Utility  ││ Bridge   ││ Railway          │ │
│ │ (line +  ││ (polygon)││ (line)           │ │
│ │ polygon) ││          ││                  │ │
│ └──────────┘└──────────┘└──────────────────┘ │
└──────────────────────────────────────────────┘
    │
    ▼
Binary/Multi-class Masks → Vectorized Shapefiles
```

## ✨ Key Production Features

This version of the model has been audited and upgraded to production standards, specifically for India's **SVAMITVA** scheme requirements:

- **Mathematical Logit Blending**: Tiled inference uses logit-space Gaussian accumulation for perfectly seamless feature extraction across 25GB+ orthophotos.
- **GIS-Grade Alignment**: Strict CRS verification and global percentile normalization ensure sub-pixel accuracy between drone imagery and vector masks.
- **Topology-Aware Export**: Uses `representative_point()` and `buffer(0)` cleaning to guarantee valid, finite boundaries for L-shaped or complex rural building geometries.
- **Training Stability**: Integrated Gradient Clipping and OneCycleLR scheduling for rapid, stable convergence to 99% accuracy targets.
- **Nodata-Aware Inference**: Zero-prediction extraction on black/nodata regions for clean, ready-to-use shapefiles.

## 📋 Output Tasks (11)

| Output Key | Target Feature | Geometry |
|---|---|---|
| `building_mask` | Built-up Area | Polygon |
| `roof_type_mask` | Roof Classification (RCC/Tiled/Tin/Others) | Polygon |
| `road_mask` | Road | Polygon |
| `road_centerline_mask` | Road Centre Line | Line |
| `waterbody_mask` | Water Body | Polygon |
| `waterbody_line_mask` | Water Body Line | Line |
| `waterbody_point_mask` | Waterbody Point (Wells) | Point |
| `utility_line_mask` | Utility (Pipeline/Wires) | Line |
| `utility_point_mask` | Utility Point (Transformers/Tanks) | Point |
| `bridge_mask` | Bridge | Polygon |
| `railway_mask` | Railway | Line |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd SAM2_SVAMITVA
pip install -r requirements.txt
```

### 2. Train

```bash
# Quick smoke-test (3 epochs)
python train.py --quick_test --train_dirs /path/to/MAP1 /path/to/MAP2

# Full training
python train.py \
    --train_dirs data/MAP1 data/MAP2 \
    --epochs 100 \
    --batch_size 8 \
    --backbone sam2

# Resume from checkpoint
python train.py --resume checkpoints/best.pt
```

### 3. Inference

```python
from inference.predict import TiledPredictor, load_model_for_inference

model = load_model_for_inference("checkpoints/best.pt")
predictor = TiledPredictor(model, device=torch.device("cuda"))
results = predictor.predict_tif(Path("village.tif"))
```

### 4. Export to Shapefiles

```python
from inference.export import export_predictions

exported = export_predictions(
    results,
    tif_path=Path("village.tif"),
    output_dir=Path("output/village"),
    threshold=0.5,
)
```

### 5. Web App

```bash
streamlit run app.py
```

## 📂 Project Structure

```
SAM2_SVAMITVA/
├── config/default.yaml      # All hyperparameters
├── data/
│   ├── preprocessing.py     # GeoTIFF I/O, tiling, normalization
│   ├── dataset.py           # PyTorch Dataset + DataLoader factory
│   └── augmentation.py      # Albumentations pipelines
├── models/
│   ├── sam2_encoder.py      # SAM2 Hiera encoder wrapper
│   ├── decoder.py           # FPN + CBAM attention decoder
│   ├── heads.py             # 11 task-specific heads
│   ├── model.py             # Full multi-task model assembly
│   └── losses.py            # Multi-task loss functions
├── training/
│   ├── config.py            # TrainingConfig dataclass
│   ├── trainer.py           # Training loop (AMP, scheduling, etc.)
│   └── metrics.py           # IoU, Dice, accuracy per task
├── inference/
│   ├── predict.py           # Tiled inference with Gaussian blending
│   └── export.py            # Mask → Shapefile/GeoJSON export
├── app.py                   # Streamlit web UI
├── train.py                 # CLI training entry point
└── requirements.txt
```

## ⚙️ Training Features

- **SAM2 backbone**: Meta's Hiera B+ image encoder with staged unfreezing
- **Multi-scale FPN decoder**: 4-level feature pyramid with CBAM attention
- **Mixed-precision training**: AMP for faster GPU training
- **Cosine annealing + warmup**: Learning rate scheduling
- **Gradient clipping**: 0.5 max norm for multi-task stability
- **Early stopping**: Monitors avg IoU across all tasks
- **Per-task weighting**: Rare features (utilities, waterbody points) get higher loss weight
- **Multi-task loss**: BCE + Dice + Focal for binary, CE + Dice for roof types

## 📊 Metrics

Per-task metrics computed during validation:
- **IoU** (Intersection over Union)
- **Dice / F1 Score**
- **Precision / Recall**
- **Pixel Accuracy**
- **Roof-type classification accuracy** (per-class)

## 📡 Data

Training data: SVAMITVA scheme drone imagery (GeoTIFFs + shapefiles):
- `MAP1/MAP1.tif` + companion shapefiles
- `MAP2/MAP2.tif` + companion shapefiles

The dataset auto-discovers TIF files and matches shapefiles by name pattern.
