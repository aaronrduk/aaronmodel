"""
Convert SVAMITVA shapefile point annotations to YOLO bounding-box format.

Reads orthophoto .tif files + associated point shapefiles (utility_point,
waterbody_point) from each MAP directory.  Produces a YOLO-ready dataset
with tiled images and per-tile label files.

Usage:
    python scripts/prepare_yolo_dataset.py \
        --map_dirs ../DATA/MAP4 ../DATA/MAP5 \
        --output yolo_dataset \
        --tile_size 1024 \
        --val_split 0.15
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box as shapely_box

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Class mapping ───────────────────────────────────────────────
# Matches the YOLO_CLASS_TO_MASK in inference/predict.py
#   0 → well  (waterbody_point)
#   1 → transformer (utility_point)
#   2 → tank / overhead tank (utility_point)

UTILITY_CLASS_MAP: Dict[str, int] = {
    "transformer": 1,
    "Transformer": 1,
    "TRANSFORMER": 1,
    "Trans": 1,
    "trans": 1,
    "Electric Transformer": 1,
    "electric transformer": 1,
    "E.T.": 1,
    "ET": 1,
    "overhead tank": 2,
    "Overhead Tank": 2,
    "OVERHEAD TANK": 2,
    "OHT": 2,
    "oht": 2,
    "tank": 2,
    "Tank": 2,
    "TANK": 2,
    "Water Tank": 2,
    "water tank": 2,
    "post": 1,
    "Post": 1,
    "POST": 1,
    "Pole": 1,
    "pole": 1,
    "Electric Pole": 1,
    "electric pole": 1,
}

WATERBODY_CLASS_MAP: Dict[str, int] = {
    # All waterbody points → class 0 (well)
}
WATERBODY_DEFAULT_CLASS = 0

CLASS_NAMES = {0: "well", 1: "transformer", 2: "tank"}

# Bounding box size in pixels around each point
BBOX_SIZE_PX = 32


def find_orthophoto(map_dir: Path) -> Optional[Path]:
    """Find the orthophoto .tif in a MAP directory."""
    for pattern in [
        "*.tif",
        "*.TIF",
        "*.tiff",
        "*.TIFF",
        "**/*.tif",
        "**/*.TIF",
    ]:
        matches = sorted(map_dir.glob(pattern))
        for m in matches:
            if m.stem.lower().startswith("ortho") or True:
                return m
    return None


def find_shapefiles(
    map_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Find utility-point and waterbody-point shapefiles."""
    utility_shps: List[Path] = []
    waterbody_shps: List[Path] = []

    for shp in map_dir.rglob("*.shp"):
        name_lower = shp.stem.lower()
        if any(
            k in name_lower
            for k in [
                "utility_poly",
                "utility_area",
                "utility_point",
                "transformer",
            ]
        ):
            utility_shps.append(shp)
        elif any(
            k in name_lower
            for k in [
                "waterbody_point",
                "water_body_point",
                "well",
            ]
        ):
            waterbody_shps.append(shp)

    return utility_shps, waterbody_shps


def detect_label_column(
    gdf: gpd.GeoDataFrame,
) -> Optional[str]:
    """Auto-detect the attribute column with feature names."""
    candidates = [
        "Name",
        "name",
        "NAME",
        "Type",
        "type",
        "TYPE",
        "Feature",
        "feature",
        "FEATURE",
        "Label",
        "label",
        "LABEL",
        "Category",
        "category",
        "CATEGORY",
        "Descriptio",
        "descriptio",
        "DESCRIPTIO",
        "Class",
        "class",
        "CLASS",
        "Sl_Typ",
        "SL_TYP",
        "sl_typ",
        "Structure",
        "STRUCTURE",
    ]
    for col in candidates:
        if col in gdf.columns:
            return col
    # Fallback: first text column that isn't geometry
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype == object:
            return col
    return None


def geo_to_pixel(
    x: float,
    y: float,
    transform: rasterio.Affine,
) -> Tuple[float, float]:
    """Convert geographic coordinates to pixel coordinates."""
    inv = ~transform
    px, py = inv * (x, y)
    return float(px), float(py)


def process_map_directory(
    map_dir: Path,
    out_images: Path,
    out_labels: Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> int:
    """Process one MAP directory into tiled images + YOLO labels."""
    ortho_path = find_orthophoto(map_dir)
    if ortho_path is None:
        logger.warning(f"No orthophoto found in {map_dir}")
        return 0

    utility_shps, waterbody_shps = find_shapefiles(map_dir)
    logger.info(
        f"[{map_dir.name}] Ortho: {ortho_path.name}, "
        f"Utility SHPs: {len(utility_shps)}, "
        f"Waterbody SHPs: {len(waterbody_shps)}"
    )

    if not utility_shps and not waterbody_shps:
        logger.warning(f"No point shapefiles found in {map_dir}")
        return 0

    # Load all point features
    all_points: List[Dict] = []  # {x, y, class_id}

    for shp_path in utility_shps:
        gdf = gpd.read_file(shp_path)
        label_col = detect_label_column(gdf)
        logger.info(
            f"  Utility SHP: {shp_path.name}, "
            f"{len(gdf)} features, "
            f"label_col={label_col}"
        )
        if label_col:
            unique_vals = gdf[label_col].unique()
            logger.info(f"    Unique values: {unique_vals}")

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            # Get centroid for any geometry type
            pt = geom.centroid
            label = ""
            if label_col and label_col in row.index:
                label = str(row[label_col]).strip()
            cls_id = UTILITY_CLASS_MAP.get(label, 1)
            all_points.append(
                {
                    "x": pt.x,
                    "y": pt.y,
                    "class_id": cls_id,
                }
            )

    for shp_path in waterbody_shps:
        gdf = gpd.read_file(shp_path)
        logger.info(f"  Waterbody SHP: {shp_path.name}, " f"{len(gdf)} features")
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            pt = geom.centroid
            all_points.append(
                {
                    "x": pt.x,
                    "y": pt.y,
                    "class_id": WATERBODY_DEFAULT_CLASS,
                }
            )

    logger.info(f"  Total point annotations: {len(all_points)}")
    if not all_points:
        return 0

    # Tile the orthophoto and create labels
    n_tiles = 0
    with rasterio.open(ortho_path) as src:
        img_h, img_w = src.height, src.width
        transform = src.transform
        crs = src.crs

        # Reproject points to raster CRS if needed
        pts_gdf = gpd.GeoDataFrame(
            all_points,
            geometry=gpd.points_from_xy(
                [p["x"] for p in all_points],
                [p["y"] for p in all_points],
            ),
        )
        # Try to match CRS
        if utility_shps:
            src_gdf = gpd.read_file(utility_shps[0])
            if src_gdf.crs and crs:
                pts_gdf = pts_gdf.set_crs(src_gdf.crs)
                if str(src_gdf.crs) != str(crs):
                    pts_gdf = pts_gdf.to_crs(crs)

        step = tile_size - overlap
        for y0 in range(0, img_h, step):
            for x0 in range(0, img_w, step):
                # Clamp tile to image bounds
                x_end = min(x0 + tile_size, img_w)
                y_end = min(y0 + tile_size, img_h)
                tw = x_end - x0
                th = y_end - y0

                if tw < tile_size // 2 or th < tile_size // 2:
                    continue

                # Tile window
                win = Window(x0, y0, tw, th)
                tile_transform = src.window_transform(win)

                # Find points in this tile
                tile_labels = []
                for _, pt_row in pts_gdf.iterrows():
                    px, py = geo_to_pixel(
                        pt_row.geometry.x,
                        pt_row.geometry.y,
                        tile_transform,
                    )
                    # Check if point is inside tile
                    if 0 <= px < tw and 0 <= py < th:
                        cls_id = pt_row["class_id"]
                        # YOLO format: class x_center y_center w h
                        # (normalized 0-1)
                        x_c = px / tw
                        y_c = py / th
                        bw = min(BBOX_SIZE_PX / tw, 1.0)
                        bh = min(BBOX_SIZE_PX / th, 1.0)
                        # Clamp to [0, 1]
                        x_c = max(bw / 2, min(x_c, 1 - bw / 2))
                        y_c = max(bh / 2, min(y_c, 1 - bh / 2))
                        tile_labels.append(
                            f"{cls_id} {x_c:.6f} " f"{y_c:.6f} {bw:.6f} {bh:.6f}"
                        )

                # Only save tiles that have annotations
                if not tile_labels:
                    continue

                # Read and save tile image
                tile_data = src.read(window=win)
                # Keep only 3 bands
                if tile_data.shape[0] > 3:
                    tile_data = tile_data[:3]

                tile_name = f"{map_dir.name}_{x0}_{y0}"

                # Save image as PNG (YOLO-friendly)
                img_out = out_images / f"{tile_name}.png"
                # Convert to uint8 RGB
                if tile_data.dtype == np.uint16:
                    tile_data = (tile_data.astype(np.float32) / 65535.0 * 255).astype(
                        np.uint8
                    )
                elif tile_data.dtype != np.uint8:
                    tile_data = np.clip(tile_data, 0, 255).astype(np.uint8)

                # Pad to tile_size if needed
                if tw < tile_size or th < tile_size:
                    padded = np.zeros(
                        (tile_data.shape[0], tile_size, tile_size),
                        dtype=np.uint8,
                    )
                    padded[:, :th, :tw] = tile_data
                    tile_data = padded

                # Save as PNG via rasterio
                with rasterio.open(
                    img_out,
                    "w",
                    driver="PNG",
                    height=tile_data.shape[1],
                    width=tile_data.shape[2],
                    count=tile_data.shape[0],
                    dtype=np.uint8,
                ) as dst:
                    dst.write(tile_data)

                # Save labels
                lbl_out = out_labels / f"{tile_name}.txt"
                with open(lbl_out, "w") as f:
                    f.write("\n".join(tile_labels) + "\n")

                n_tiles += 1

    logger.info(f"  Generated {n_tiles} annotated tiles")
    return n_tiles


def main():
    parser = argparse.ArgumentParser(
        description="Convert SVAMITVA shapefiles to YOLO dataset",
    )
    parser.add_argument(
        "--map_dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to MAP directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_dataset"),
        help="Output directory",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)

    out = args.output
    train_img = out / "images" / "train"
    train_lbl = out / "labels" / "train"
    val_img = out / "images" / "val"
    val_lbl = out / "labels" / "val"

    # Clean output
    if out.exists():
        shutil.rmtree(out)
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # First pass: generate all tiles into a temp dir
    tmp_img = out / "_tmp_images"
    tmp_lbl = out / "_tmp_labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    for map_dir in args.map_dirs:
        map_dir = map_dir.resolve()
        if not map_dir.exists():
            logger.warning(f"Directory not found: {map_dir}")
            continue
        n = process_map_directory(
            map_dir,
            tmp_img,
            tmp_lbl,
            tile_size=args.tile_size,
        )
        total += n

    if total == 0:
        logger.error("No annotated tiles generated!")
        return

    # Split into train/val
    all_labels = sorted(tmp_lbl.glob("*.txt"))
    random.shuffle(all_labels)
    n_val = max(1, int(len(all_labels) * args.val_split))
    val_set = set(f.stem for f in all_labels[:n_val])

    for lbl_file in all_labels:
        stem = lbl_file.stem
        img_file = tmp_img / f"{stem}.png"
        if not img_file.exists():
            continue

        if stem in val_set:
            shutil.move(str(img_file), str(val_img / img_file.name))
            shutil.move(str(lbl_file), str(val_lbl / lbl_file.name))
        else:
            shutil.move(str(img_file), str(train_img / img_file.name))
            shutil.move(str(lbl_file), str(train_lbl / lbl_file.name))

    # Cleanup temp
    shutil.rmtree(tmp_img, ignore_errors=True)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    # Create dataset YAML
    yaml_path = out / "svamitva_points.yaml"
    n_train = len(list(train_img.glob("*.png")))
    n_val_final = len(list(val_img.glob("*.png")))

    yaml_content = (
        f"# SVAMITVA Point Feature Detection Dataset\n"
        f"# Auto-generated by prepare_yolo_dataset.py\n"
        f"\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
    )
    for cid, cname in sorted(CLASS_NAMES.items()):
        yaml_content += f"  {cid}: {cname}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"\n{'='*50}")
    logger.info(f"YOLO dataset ready at: {out.resolve()}")
    logger.info(f"  Train: {n_train} tiles")
    logger.info(f"  Val:   {n_val_final} tiles")
    logger.info(f"  Classes: {CLASS_NAMES}")
    logger.info(f"  Config: {yaml_path}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
