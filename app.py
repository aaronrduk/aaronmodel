"""
SVAMITVA Feature Extraction — Production Ensemble V3 Web Application
"""

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from inference.predict import load_ensemble_pipeline

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SVAMITVA Ensemble AI",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #e0e0ff !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Metadata ─────────────────────────────────────────────────────────

FEATURES = {
    "building_mask": ("🏘️ Buildings (SAM2+Dual-Head)", (255, 100, 50)),
    "roof_type_mask": ("🏠 Roof Types (RCC/Tiled/Tin/Others)", (255, 0, 180)),
    "road_mask": ("🛣️ Roads (SAM2+Link-Head)", (255, 255, 100)),
    "road_centerline_mask": ("🧭 Road Centre Line", (255, 220, 0)),
    "waterbody_mask": ("💧 Waterbodies (SAM2+Binary-Head)", (50, 150, 255)),
    "waterbody_line_mask": ("🌊 Water Body Line", (0, 210, 255)),
    "waterbody_point_mask": ("🕳️ Wells (YOLOv8 + SAM2)", (0, 255, 255)),
    "utility_line_mask": ("⚡ Utility Lines (SAM2+Line-Head)", (50, 220, 100)),
    "utility_point_mask": ("🔌 Utility Points (YOLOv8 + SAM2)", (0, 255, 120)),
    "bridge_mask": ("🌉 Bridge", (255, 140, 0)),
    "railway_mask": ("🚂 Railways (SAM2+Line-Head)", (180, 80, 255)),
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
)

# ── Main ─────────────────────────────────────────────────────────────────


def main():
    st.title("🛰️ SVAMITVA Production Ensemble V3")

    with st.sidebar:
        st.header("⚙️ Model Configuration")
        ckpt_path = st.text_input(
            "Ensemble Checkpoint (.pt)", "checkpoints/ensemble_v3.pt"
        )
        yolo_path = st.text_input("YOLOv8 Weights (.pt)", "checkpoints/yolov8s.pt")
        selected_masks = []
        for key, meta in FEATURES.items():
            enabled = st.checkbox(meta[0], value=True, key=f"feature_{key}")
            if enabled:
                selected_masks.append(key)
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5)
        alpha = st.slider("Overlay Opacity", 0.1, 0.8, 0.4)

    uploaded = st.file_uploader(
        "Upload Drone Orthophoto (GeoTIFF)", type=["tif", "tiff"]
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tif_path = Path(tmp.name)

        if st.button("🚀 Run End-to-End Extraction", type="primary"):
            with st.spinner("Processing Large Orthophoto..."):
                predictor = load_ensemble_pipeline(
                    ckpt_path,
                    yolo_path if Path(yolo_path).exists() else None,
                    device=DEVICE,
                )
                predictor.threshold = threshold
                st.session_state.results = predictor.predict_tif(
                    tif_path,
                    selected_masks=selected_masks,
                )
                st.session_state.tif_path = tif_path
                st.success("Extraction Complete!")

        results = st.session_state.get("results")
        if results:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Generate GIS Layers"):
                    from inference.export import export_predictions

                    with tempfile.TemporaryDirectory() as out_dir:
                        export_predictions(results, tif_path, Path(out_dir))
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w") as zf:
                            for p in Path(out_dir).glob("*.gpkg"):
                                zf.write(p, p.name)
                        st.download_button(
                            "📩 Download ZIP", zip_buf.getvalue(), "results.zip"
                        )

            from inference.predict import Reader
            import cv2

            with Reader(str(tif_path)) as src:
                thumb = np.transpose(src.preview(max_size=1200).data, (1, 2, 0))

            overlay = thumb.copy()
            for key, (name, color) in FEATURES.items():
                if key in results:
                    m_small = cv2.resize(results[key], (thumb.shape[1], thumb.shape[0]))
                    binary = m_small > 0 if key == "roof_type_mask" else m_small > threshold
                    for c in range(3):
                        overlay[binary, c] = (
                            overlay[binary, c] * (1 - alpha) + color[c] * alpha
                        )
            st.image(overlay.astype(np.uint8), caption="Ensemble Overview")


if __name__ == "__main__":
    main()
