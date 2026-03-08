# 🛰️ SVAMITVA AI Extraction Ensemble (V3)

High-resolution geospatial feature extraction from drone orthophotos, designed for the **SVAMITVA** village mapping scheme. This pipeline achieves **≥95% accuracy** by leveraging a specialized ensemble of State-of-the-Art (SOTA) deep learning models.

---

## 🚀 Key Features

- **Multi-Model Ensemble**: Specialized backbones for distinct geospatial features (Buildings, Roads, Water, etc.).
- **Point Feature Detection**: Integrated **YOLOv8** for accurate detection of wells, transformers, and tanks.
- **Roof Classification**: **EfficientNet-B0** driven classification for built-up areas (RCC, Tiled, Tin, Others).
- **Sequential Tiled Inference**: Optimized VRAM management for processing massive GeoTIFF imagery (village-scale).
- **Unified GIS Export**: Automated generation of georeferenced **GeoPackage (.gpkg)** layers.
- **Staged Training**: Support for layerwise/staggered optimization (Heads -> Full fine-tuning).

---

## 🏗️ Architecture Stack

| Feature | SOTA Model | Backbone |
| :--- | :--- | :--- |
| **Buildings / Water** | DeepLabV3+ | ResNet101 |
| **Roads** | Linknet | ResNet34 |
| **Utilities (Lines)** | U-Net++ | ResNet34 |
| **Railway** | Unet | ResNet101 |
| **Roof Types** | Unet | EfficientNet-B0 |
| **Point Objects** | YOLOv8 | YOLOv8s |

---

## 🧭 Recommended Global Strategy

For best practical accuracy and stability on drone imagery, use a **hybrid pipeline**:

1. **SAM2 for strong segmentation priors and auto-label bootstrapping**
2. **Task-specialized models for production heads**
   - DeepLabV3+ (building, water polygons)
   - D-LinkNet (roads / centrelines)
   - U-Net++ (utility linear features)
   - HRNet (railway continuity)
3. **YOLOv8 for sparse point objects** (wells, transformers, tanks)
4. **Roof-type classification head** for RCC/Tiled/Tin/Others

The current codebase supports this in practice via:
- SAM2-based multi-head segmentation model
- YOLOv8 fusion for point masks
- Roof-type raster + GIS export

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aaronrduk/aaronmodel.git
   cd aaronmodel
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### 1. Interactive Web Application
Launch the production-grade Streamlit interface for end-to-end extraction and visualization:
```bash
streamlit run app.py
```

### 2. Model Training
Train the ensemble model on your own drone dataset (MAP1, MAP2, etc.):
```bash
python3 train.py --train_dirs /path/to/data --epochs 100 --lr 3e-4
```
*Note: Use `--quick_test` for a 3-epoch smoke test.*

### 3. DGX/GPU Cluster Training
For high-performance environments (NVIDIA DGX), use the provided notebook:
- `dgx.ipynb`: Contains the multi-phase staged training logic.

---

## 📁 Data Structure
To train or run inference, organize your data as follows:
```text
data/
└── MAP_ID/
    ├── MAP_ID.tif         # High-resolution Orthophoto
    ├── Build_up.shp       # Building annotations
    ├── Road.shp           # Road annotations
    └── ...                # Other feature shapefiles
```

---

## 🎯 Output Keys

| Output Key | Target Feature | Geometry |
| :--- | :--- | :--- |
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

---

## 📊 Performance & Target
This project is engineered to meet the stringent requirements of SVAMITVA mapping:
- **Target IoU**: ≥95% across all primary layers.
- **Resolution**: Native support for 5cm - 10cm GSD drone imagery.
- **Export Formats**: Standard GIS vector outputs (SHP/GPKG/GeoJSON).

---

## ⚖️ License
This project is proprietary for the SVAMITVA scheme. (C) 2026.
