"""
Mask → Shapefile/GeoJSON export module.

Converts model prediction masks into vectorized geographic features
with proper CRS, attributes, and geometry types.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Roof-type class labels
ROOF_LABELS = ["Background", "RCC", "Tiled", "Tin", "Others"]

# Feature metadata
FEATURE_INFO = {
    "building_mask": {
        "name": "Buildings",
        "geom_type": "Polygon",
        "min_area": 25,
        "simplify": 1.5,
    },
    "road_mask": {
        "name": "Roads",
        "geom_type": "Polygon",
        "min_area": 50,
        "simplify": 2.0,
    },
    "road_centerline_mask": {
        "name": "Road_Centrelines",
        "geom_type": "LineString",
        "min_area": 10,
        "simplify": 1.0,
    },
    "waterbody_mask": {
        "name": "Waterbodies",
        "geom_type": "Polygon",
        "min_area": 30,
        "simplify": 1.5,
    },
    "waterbody_line_mask": {
        "name": "Waterbody_Lines",
        "geom_type": "LineString",
        "min_area": 10,
        "simplify": 1.0,
    },
    "waterbody_point_mask": {
        "name": "Waterbody_Points",
        "geom_type": "Point",
        "min_area": 5,
        "simplify": 0,
    },
    "utility_line_mask": {
        "name": "Utility_Lines",
        "geom_type": "LineString",
        "min_area": 10,
        "simplify": 1.0,
    },
    "utility_point_mask": {
        "name": "Utility_Points",
        "geom_type": "Point",
        "min_area": 5,
        "simplify": 0,
    },
    "bridge_mask": {
        "name": "Bridges",
        "geom_type": "Polygon",
        "min_area": 20,
        "simplify": 1.5,
    },
    "railway_mask": {
        "name": "Railways",
        "geom_type": "LineString",
        "min_area": 10,
        "simplify": 1.0,
    },
}


def _mask_to_polygons(
    mask: np.ndarray,
    transform,
    threshold: float = 0.5,
    min_area: float = 25,
    simplify_tol: float = 1.5,
) -> List[Polygon]:
    """Convert a probability mask to simplified polygons."""
    binary = (mask > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return []

    polygons = []
    for geom, value in rasterio_shapes(binary, mask=binary > 0, transform=transform):
        if value == 0:
            continue
        poly = shape(geom)
        if poly.area < min_area:
            continue
        if simplify_tol > 0:
            poly = poly.simplify(simplify_tol, preserve_topology=True)
        if not poly.is_empty and poly.is_valid:
            polygons.append(poly)

    return polygons


def _mask_to_skeleton_lines(
    mask: np.ndarray, transform, threshold: float = 0.5
) -> List[LineString]:
    """Extract thin lines using skeletonization."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        logger.warning("skimage not installed, falling back to polygon boundaries")
        return []

    binary = (mask > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    skeleton = skeletonize(binary)
    lines = []
    for geom, value in rasterio_shapes(
        skeleton.astype(np.uint8),
        mask=skeleton > 0,
        transform=transform,
    ):
        if value > 0:
            ls = shape(geom)
            if not ls.is_empty:
                lines.append(ls)
    return lines


def _convert_to_target_geom(
    mask: np.ndarray,
    polygons: List[Polygon],
    target_type: str,
    transform,
) -> list:
    """Convert polygons to target geometry type."""
    if target_type == "Polygon":
        return polygons
    elif target_type == "LineString":
        return _mask_to_skeleton_lines(mask, transform)
    elif target_type == "Point":
        return [p.centroid for p in polygons if not p.is_empty]
    return polygons


# ── Main Export Functions ─────────────────────────────────────────────────────


def export_predictions(
    predictions: Dict[str, np.ndarray],
    tif_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    roof_type_mask: Optional[np.ndarray] = None,
    format: str = "both",
) -> Dict[str, Path]:
    """
    Export model predictions as Shapefiles and/or GeoJSON.

    Args:
        predictions: dict of mask_key -> (H, W) float32 maps
        tif_path: source orthophoto for CRS and transform
        output_dir: directory to write outputs
        threshold: binarization threshold
        roof_type_mask: optional (H, W) uint8 roof class indices
        format: 'shapefile', 'geojson', or 'both'

    Returns:
        Dict of mask_key -> output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs

    exported = {}

    for mask_key, mask in predictions.items():
        if mask_key not in FEATURE_INFO:
            continue

        info = FEATURE_INFO[mask_key]
        logger.info(f"Exporting {info['name']}...")

        polygons = _mask_to_polygons(
            mask,
            transform,
            threshold,
            info["min_area"],
            info["simplify"],
        )

        if not polygons:
            logger.info(f"  {info['name']}: no features detected")
            continue

        geometries = _convert_to_target_geom(
            mask, polygons, info["geom_type"], transform
        )

        if not geometries:
            continue

        records = []
        for i, geom in enumerate(geometries):
            record = {
                "geometry": geom,
                "FID": i + 1,
                "Feature": info["name"],
                "Area_sqm": geom.area if hasattr(geom, "area") else 0,
            }

            if mask_key == "building_mask" and roof_type_mask is not None:
                # Use representative_point() instead of centroid to guarantee the point
                # is INSIDE the building (crucial for L-shaped/concave buildings)
                point = geom.representative_point()
                cx, cy = point.coords[0]
                col, row = ~transform * (cx, cy)
                row, col = int(row), int(col)

                if (
                    0 <= row < roof_type_mask.shape[0]
                    and 0 <= col < roof_type_mask.shape[1]
                ):
                    rt_idx = int(roof_type_mask[row, col])
                    record["Roof_Type"] = ROOF_LABELS[min(rt_idx, 4)]
                else:
                    record["Roof_Type"] = "Unknown"

            # Finite boundaries: clean topology
            if not geom.is_valid:
                geom = geom.buffer(0)
                record["geometry"] = geom

            records.append(record)

        gdf = gpd.GeoDataFrame(records, crs=crs)

        base_name = info["name"]
        if format in ("shapefile", "both"):
            shp_path = output_dir / f"{base_name}.shp"
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            exported[mask_key] = shp_path
            logger.info(f"  -> {shp_path.name} ({len(gdf)} features)")

        if format in ("geojson", "both"):
            json_path = output_dir / f"{base_name}.geojson"
            gdf.to_file(json_path, driver="GeoJSON")
            if mask_key not in exported:
                exported[mask_key] = json_path
            logger.info(f"  -> {json_path.name} ({len(gdf)} features)")

    return exported


def create_overlay_image(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    threshold: float = 0.5,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a visualization overlay of predictions on the image."""
    COLORS = {
        "building_mask": (255, 100, 50),
        "road_mask": (255, 255, 100),
        "road_centerline_mask": (200, 200, 60),
        "waterbody_mask": (50, 150, 255),
        "waterbody_line_mask": (80, 180, 255),
        "waterbody_point_mask": (100, 200, 255),
        "utility_line_mask": (50, 220, 100),
        "utility_point_mask": (100, 255, 150),
        "bridge_mask": (220, 130, 50),
        "railway_mask": (180, 80, 255),
    }

    composite = image.copy()

    for mask_key, mask in predictions.items():
        if mask_key not in COLORS:
            continue
        color = COLORS[mask_key]
        binary = mask > threshold
        if binary.sum() == 0:
            continue

        overlay = np.zeros_like(composite)
        for c in range(3):
            overlay[:, :, c] = color[c]

        mask_3d = np.stack([binary] * 3, axis=-1)
        composite = np.where(
            mask_3d,
            (composite * (1 - alpha) + overlay * alpha).astype(np.uint8),
            composite,
        )

    return composite
