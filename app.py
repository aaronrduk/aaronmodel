"""
SVAMITVA Feature Extraction — Streamlit Web Application

Upload drone orthophotos (TIF/JPG/PNG) → Extract geographic features
with SAM2-based AI model → Visualize + Download Shapefiles.
"""

import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import rasterio
from rasterio.io import MemoryFile
from data.preprocessing import (
    compute_tile_windows,
    read_tile,
    read_geotiff_meta,
    compute_global_stretch,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SVAMITVA Feature Extraction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main .block-container { padding-top: 2rem; max-width: 1400px; }
    h1 { color: #e0e0ff !important; }
    h2, h3 { color: #c0c0e0 !important; }

    .feature-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 0.5rem;
    }

    .stat-box {
        background: linear-gradient(135deg, rgba(100,100,255,0.15), rgba(50,50,200,0.08));
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(100,100,255,0.2);
    }

    .detected { border-left: 4px solid #4caf50; }
    .not-detected { border-left: 4px solid rgba(255,80,80,0.4); opacity: 0.6; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Feature Metadata ─────────────────────────────────────────────────────────

FEATURES = {
    "building_mask": ("🏘️ Buildings", (255, 100, 50)),
    "road_mask": ("🛣️ Roads", (255, 255, 100)),
    "road_centerline_mask": ("📏 Road Centerlines", (200, 200, 60)),
    "waterbody_mask": ("💧 Waterbodies", (50, 150, 255)),
    "waterbody_line_mask": ("〰️ Water Lines", (80, 180, 255)),
    "waterbody_point_mask": ("📍 Wells / Water Points", (100, 200, 255)),
    "utility_line_mask": ("⚡ Utility Lines", (50, 220, 100)),
    "utility_point_mask": ("🔌 Utility Points (Transformers/Tanks)", (100, 255, 150)),
    "bridge_mask": ("🌉 Bridges", (220, 130, 50)),
    "railway_mask": ("🚂 Railways", (180, 80, 255)),
}

ROOF_TYPES = ["Background", "RCC", "Tiled", "Tin", "Others"]


# ── Model Loading ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


@st.cache_resource
def load_model(ckpt_path: str):
    """Load model from checkpoint."""
    from models.model import SvamitvaModel

    if not Path(ckpt_path).exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)

    # Detect backbone
    backbone = (
        "sam2" if any("encoder.encoder" in k for k in state.keys()) else "resnet50"
    )

    model = SvamitvaModel(backbone=backbone, pretrained=False)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    return model


def find_checkpoints():
    """Find available model checkpoints."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return []
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in pts]


# ── Image Loading ─────────────────────────────────────────────────────────────


def load_image(uploaded_file) -> tuple:
    """Load an uploaded image file. Returns (image_np, geo_meta)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith((".tif", ".tiff")):
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                bands = min(src.count, 3)
                img = src.read(list(range(1, bands + 1)))
                geo_meta = {"crs": src.crs, "transform": src.transform}
    else:
        from PIL import Image
        import io

        img_pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = np.array(img_pil).transpose(2, 0, 1)
        geo_meta = None

    img = np.moveaxis(img, 0, -1)  # (H, W, C)
    if img.dtype != np.uint8:
        # Percentile stretch
        img = img.astype(np.float32)
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            valid = ch[ch > 0]
            if len(valid) > 0:
                lo, hi = np.percentile(valid, [2, 98])
                if hi - lo < 1e-6:
                    hi = lo + 1
                img[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1) * 255
            else:
                img[:, :, c] = 0
        img = img.astype(np.uint8)
    return img, geo_meta


def downsample_for_display(image: np.ndarray, max_size: int = 2000) -> np.ndarray:
    """Resize image for stable browser display if it exceeds max_size."""
    if image is None:
        return None
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image

    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    import cv2

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Inference ─────────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE_SIZE = 512
OVERLAP = 64


@torch.no_grad()
def run_inference(image_np, model, selected_keys):
    """Run model inference on full image using logit-space blending."""
    H, W = image_np.shape[:2]
    results = {}

    # Initial nodata mask
    nodata_mask = image_np.sum(axis=-1) == 0

    # Global stretch sampling (Streamlit version: faster)
    stretch = None
    if image_np.dtype != np.uint8:
        stretch = {}
        for c in range(3):
            ch = image_np[..., c]
            valid = ch[ch > 0]
            if len(valid) > 500:
                stretch[c] = tuple(np.percentile(valid, [2, 98]))

    # Initialize logit accumulators
    logit_accum = {
        k: np.zeros((H, W), dtype=np.float32)
        for k in selected_keys
        if k != "roof_type_mask"
    }
    roof_logit_accum = (
        np.zeros((5, H, W), dtype=np.float32)
        if "roof_type_mask" in selected_keys
        else None
    )
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Gaussian kernel for blending
    x = np.arange(TILE_SIZE) - TILE_SIZE / 2 + 0.5
    k1d = np.exp(-0.5 * (x / (TILE_SIZE / 4)) ** 2)
    kernel = np.outer(k1d, k1d).astype(np.float32)
    kernel /= kernel.max()

    step = TILE_SIZE - OVERLAP
    for y in range(0, H, step):
        for x_pos in range(0, W, step):
            h = min(TILE_SIZE, H - y)
            w = min(TILE_SIZE, W - x_pos)
            if h < TILE_SIZE // 4 or w < TILE_SIZE // 4:
                continue

            tile = image_np[y : y + h, x_pos : x_pos + w].copy()

            # Apply stretch if not uint8
            if stretch:
                tile = tile.astype(np.float32)
                for c in range(3):
                    lo, hi = stretch.get(c, (0, 1))
                    tile[..., c] = (
                        np.clip((tile[..., c] - lo) / (hi - lo + 1e-6), 0, 1) * 255
                    )
                tile = tile.astype(np.uint8)

            if h < TILE_SIZE or w < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:h, :w] = tile
                tile = padded

            img = tile.astype(np.float32) / 255.0
            img = (img - MEAN) / STD
            tensor = (
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            )

            out = model(tensor)

            blend = kernel[:h, :w]
            for key in selected_keys:
                if key in out:
                    logits = out[key].squeeze().cpu().numpy()
                    if key == "roof_type_mask":
                        roof_logit_accum[:, y : y + h, x_pos : x_pos + w] += (
                            logits[:, :h, :w] * blend[np.newaxis]
                        )
                    else:
                        logit_accum[key][y : y + h, x_pos : x_pos + w] += (
                            logits[:h, :w] * blend
                        )

            weight_map[y : y + h, x_pos : x_pos + w] += blend

    # Normalize and activate
    weight_map = np.maximum(weight_map, 1e-8)
    for key in selected_keys:
        if key == "roof_type_mask" and roof_logit_accum is not None:
            for c in range(5):
                roof_logit_accum[c] /= weight_map
            # Assign nodata to background
            roof_logit_accum[0][nodata_mask] = 100.0
            results[key] = roof_logit_accum.argmax(axis=0).astype(np.uint8)
        elif key in logit_accum:
            avg_logits = logit_accum[key] / weight_map
            prob = 1.0 / (1.0 + np.exp(-avg_logits))
            # Mask nodata
            prob[nodata_mask] = 0
            results[key] = prob

    return results


def create_overlay(image, predictions, threshold=0.5, alpha=0.4, target_key=None):
    """Create visualization overlay. If target_key is set, only show that layer."""
    composite = image.copy()
    for key, mask in predictions.items():
        if key not in FEATURES:
            continue
        if target_key is not None and key != target_key:
            continue

        _, color = FEATURES[key]
        binary = mask > threshold if mask.dtype == np.float32 else mask > 0
        if binary.sum() == 0:
            continue
        overlay = np.full_like(composite, color, dtype=np.uint8)
        mask_3d = np.stack([binary] * 3, axis=-1)
        composite = np.where(
            mask_3d,
            (composite * (1 - alpha) + overlay * alpha).astype(np.uint8),
            composite,
        )
    return composite


# ── Main UI ───────────────────────────────────────────────────────────────────


def main():
    st.title("🛰️ SVAMITVA Feature Extraction")
    st.markdown(
        "**AI-powered geographic feature extraction from drone orthophotos using SAM2**"
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        ckpts = find_checkpoints()
        if ckpts:
            ckpt_path = st.selectbox("Model Checkpoint", ckpts)
        else:
            st.warning("No checkpoints found in `checkpoints/`")
            ckpt_path = st.text_input("Checkpoint path", "checkpoints/best.pt")

        st.divider()
        st.subheader("Feature Selection")
        selected = {}
        for key, (label, _) in FEATURES.items():
            selected[key] = st.checkbox(label, value=True, key=key)

        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
        alpha = st.slider("Overlay Opacity", 0.1, 0.8, 0.4, 0.05)

    # Main content
    uploaded = st.file_uploader(
        "Upload Drone Image (TIF, JPG, PNG)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
        help="Supports files up to 25GB. Browser uploads over 25GB may be unstable.",
    )

    if uploaded:
        with st.spinner("Loading image..."):
            image_np, geo_meta = load_image(uploaded)

        if geo_meta:
            with st.sidebar:
                st.success("✅ Georeferenced Image Loaded")
                st.info(f"CRS: {geo_meta['crs']}")

        st.subheader("📸 Visualization")
        col1, col2 = st.columns(2)
        with col1:
            display_img = downsample_for_display(image_np)
            st.image(
                display_img,
                caption=f"View: {image_np.shape[1]}×{image_np.shape[0]} (Downsampled for stability)",
            )

        # Load model
        model = load_model(ckpt_path)
        if model is None:
            st.error("❌ Could not load model. Please check checkpoint path.")
            return

        selected_keys = [k for k, v in selected.items() if v]
        if "roof_type_mask" not in selected_keys and selected.get("building_mask"):
            selected_keys.append("roof_type_mask")

        if st.button("🚀 Extract Features", type="primary", use_container_width=True):
            with st.spinner("Running AI inference..."):
                st.session_state.predictions = run_inference(
                    image_np, model, selected_keys
                )
                st.session_state.processed_image = True

        predictions = st.session_state.get("predictions")
        if predictions:
            # Results
            with col2:
                # Layer selector for visualization
                feat_opts = [
                    FEATURES[k][0] for k in predictions.keys() if k in FEATURES
                ]
                layer_options = ["🏠 All Features"] + feat_opts
                selected_viz = st.selectbox("🎯 Visualization Layer", layer_options)

                target_key = None
                if selected_viz != "🏠 All Features":
                    target_key = [
                        k for k, v in FEATURES.items() if v[0] == selected_viz
                    ][0]

                overlay = create_overlay(
                    image_np, predictions, threshold, alpha, target_key=target_key
                )
                display_overlay = downsample_for_display(overlay)
                st.image(
                    display_overlay, caption=f"Results: {selected_viz} (Downsampled)"
                )

            # Roof type breakdown
            if "roof_type_mask" in predictions:
                st.subheader("🏠 Roof Type Classification")
                roof_mask = predictions["roof_type_mask"]
                cols = st.columns(5)
                for c, label in enumerate(ROOF_TYPES):
                    count = (roof_mask == c).sum()
                    with cols[c]:
                        st.metric(label, f"{count:,}")

            # Export
            st.subheader("💾 Export")
            if geo_meta:
                st.info(
                    "GeoTIFF detected — exports will include proper CRS coordinates"
                )

            if st.button("📥 Download Shapefiles (ZIP)"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    from inference.export import export_predictions

                    # Save temp tif for CRS
                    if geo_meta:
                        # Create temporary tif for CRS if needed
                        tif_source = Path(tmpdir) / "input.tif"
                        with rasterio.open(
                            tif_source,
                            "w",
                            driver="GTiff",
                            height=image_np.shape[0],
                            width=image_np.shape[1],
                            count=3,
                            dtype="uint8",
                            crs=geo_meta["crs"],
                            transform=geo_meta["transform"],
                        ) as dst:
                            for c in range(3):
                                dst.write(image_np[:, :, c], c + 1)

                        export_predictions(
                            predictions,
                            tif_source,
                            Path(tmpdir) / "output",
                            threshold=threshold,
                            roof_type_mask=predictions.get("roof_type_mask"),
                        )
                    else:
                        st.warning("Export requires GeoTIFF input for CRS coordinates")
                        return

                    # Zip
                    zip_buf = io.BytesIO()
                    out_dir = Path(tmpdir) / "output"
                    if out_dir.exists():
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for f in out_dir.rglob("*"):
                                if f.is_file():
                                    zf.write(f, f.name)

                        st.download_button(
                            "📥 Download ZIP",
                            data=zip_buf.getvalue(),
                            file_name="svamitva_features.zip",
                            mime="application/zip",
                        )


if __name__ == "__main__":
    main()
