# SVAMITVA Ensemble AI: Unified Pipeline Architecture (V3)

This document provides a **high-fidelity technical breakdown** of the SVAMITVA Feature Extraction system. The architecture is engineered to solve the specific challenges of rural drone imagery: high class imbalance, thin linear continuity, and multi-scale object detection.

## 1. High-Fidelity Architecture Flowchart

```mermaid
graph TD
    subgraph INPUT ["🌐 DATA INPUT & PREPROCESSING"]
        IMG["<b>Image Tile</b><br/>(512x512x3)<br/>[uint8/uint16/float32]"]
        NORM["<b>Normalization Layer</b><br/>Robust Percentile Stretch<br/>+ ImageNet Mean/Std"]
        IMG --> NORM
    end

    subgraph BACKBONE ["🏢 FOUNDATION BACKBONE: SAM-2 HIERA B+"]
        EN["<b>Transformer Encoder</b><br/>Windowed Attention Blocks"]
        C2["<b>C2</b> (1/4 res)<br/>Spatial Detail"]
        C3["<b>C3</b> (1/8 res)"]
        C4["<b>C4</b> (1/16 res)"]
        C5["<b>C5</b> (1/32 res)<br/>Semantic Context"]
        NORM --> EN
        EN --> C2 & C3 & C4 & C5
    end

    subgraph DECODER ["🛠️ DECODER: FPN + ATTENTION FUSION"]
        FPN_L["<b>Lateral Convs</b><br/>(1x1 Conv, 256ch)"]
        FPN_U["<b>Up-sampling</b><br/>(Bilinear 2x)"]
        CBAM_C["<b>Channel Attention</b><br/>(Avg+Max Pool, MLP)"]
        CBAM_S["<b>Spatial Attention</b><br/>(7x7 Conv, Sigmoid)"]
        
        C2 & C3 & C4 & C5 --> FPN_L
        FPN_L --> FPN_U
        FPN_U --> CBAM_C
        CBAM_C --> CBAM_S
        Unified["<b>Unified Latent Map</b><br/>(128x128x256)"]
        CBAM_S --> Unified
    end

    subgraph PREDICTION_HEADS ["🎯 TASK-SPECIALIZED PREDICTION"]
        subgraph HEAD_POLYGONS ["Built-up & Water Body"]
            BHead["<b>Binary/Dual Head</b><br/>Conv3x3 + BN + ReLU<br/>Sigmoid/Softmax Out"]
        end
        
        subgraph HEAD_LINES ["Roads & Railways"]
            LHead["<b>D-LinkNet Block</b><br/>Dilated Convs [1,2,4,8]<br/>Ensures Line Connectivity"]
        end
        
        subgraph HEAD_POINTS ["Utilities & Wells"]
            PHead["<b>Detection Head</b><br/>Centripetal Style Features<br/>Point Probability Maps"]
        end
        
        Unified --> BHead & LHead & PHead
    end

    subgraph POST_PROCESSING ["🚀 ENSEMBLE FUSION AGENT"]
        YOLO["<b>YOLOv8s Point Detector</b><br/>Custom Trained on SVAMITVA"]
        FUSE["<b>Probabilistic Fusion</b><br/>(Segmentation Map + YOLO Box)<br/>Logical OR / Max Conf"]
        
        IMG -.-> YOLO
        PHead --> FUSE
        YOLO --> FUSE
    end

    subgraph OUTPUTS ["🛰️ GIS-READY LAYERS (GeoPackage)"]
        PolyOut["<b>Polygons</b><br/>Buildings, Roof Types,<br/>Water Body, Road"]
        LineOut["<b>Lines</b><br/>Road Centre Line,<br/>Utility/Water Line, Railway"]
        PointOut["<b>Points</b><br/>Wells, Transformers, Tanks"]
        
        BHead --> PolyOut
        LHead --> LineOut
        FUSE --> PointOut
    end

    %% Styles
    classDef input fill:#111,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef backbone fill:#222,stroke:#4f46e5,stroke-width:2px,color:#fff
    classDef decoder fill:#1a1a1a,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef predictions fill:#111,stroke:#10b981,stroke-width:2px,color:#fff
    classDef outputs fill:#000,stroke:#f59e0b,stroke-width:2px,color:#fff
    class IMG_PATH,IMG input
    class EN,Hiera,C2,C3,C4,C5 backbone
    class FPN_L,FPN_U,CBAM_C,CBAM_S decoder
    class BHead,LHead,PHead predictions
    class PolyOut,LineOut,PointOut outputs
```

##  technical Details for Judges

### 1. Neural Backbone: Foundation Vision Tuning
The model uses the **SAM2 (Segment Anything Model 2)** Hiera backbone. We leverage its **multi-scale feature extraction** (C2 through C5 layers) to capture both micro-level edges (like utility lines) and macro-level features (like entire building clusters).

### 2. Decoder Strategy: Multi-Scale Attention
- **FPN (Feature Pyramid Network)**: Solves the scale-variance problem. Small objects (Wells) are predicted from high-resolution layers, while large objects (Roads) are predicted from high-semantic layers.
- **CBAM (Convolutional Block Attention Module)**: 
    - **Channel Attention**: Learns which spectral features matter most for different tasks.
    - **Spatial Attention**: Suppresses noise from the "Rural Background" (trees, grass) to isolate point-level targets.

### 3. Dedicated Geometry Heads
- **D-LinkNet Configuration (Linework)**: Linear features (Railway/Road) suffer from gaps. Our head uses **Dilated Convolutions (rates: 1, 2, 4, 8)** to fill gaps by looking at the neighborhood context without increasing the parameter count.
- **Dual-Head Footprints**: For buildings, a single forward pass predicts the **Mask** (Binary) and the **Roof Type** (Multi-class), ensuring zero offset between the boundary and the classification layer.

### 4. Ensemble Fusion Logic
The system uses a **Probability-Aware Ensemble**:
- **Segmentation**: Predicts the pixel-level boundary.
- **YOLOv8**: Uses a regression-based approach to get the center-point of utility features.
- **The Fusion**: `FinalMap = max(SegMap, YOLO_PointMap)`. This ensures that even if one model is occluded, the other will "save" the detection, maintaining a **High Recall** standard.

### 5. Multi-Task Training Objective
The model is trained using a **Weighted Composite Loss**:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{CrossEntropy}$$
- **Focal Loss**: Heavily penalizes the model for missing small objects (Transformers/Wells).
- **Dice Loss**: Optimizes for the Overlap (IoU) of polygons and lines.
- **CE Loss**: Used for high-accuracy categorical roof classification.
