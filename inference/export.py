"""
Mask & Detection → GIS Vector (GeoPackage) export module.

V3 Improvements:
- Integrated YOLOv8 detection box to GIS point conversion.
- Consolidated feature mapping for ensemble outputs.
- Robust geometry cleaning and attribute assignment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import Point, Polygon, MultiPolygon, shape

logger = logging.getLogger(__name__)

# GIS Attribute Labels
ROOF_LABELS = ["Background", "RCC", "Tiled", "Tin", "Others"]

# Feature Optimization Settings
FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "building_mask": {
        "name": "Buildings",
        "type": "Polygon",
        "min_area": 15,
        "simplify": 0.5,
    },
    "roof_type_mask": {
        "name": "Roof_Types",
        "type": "Polygon",
        "min_area": 10,
        "simplify": 0.5,
    },
    "road_mask": {"name": "Roads", "type": "Polygon", "min_area": 30, "simplify": 0.8},
    "road_centerline_mask": {
        "name": "Road_Centerlines",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "waterbody_mask": {
        "name": "Waterbodies",
        "type": "Polygon",
        "min_area": 25,
        "simplify": 1.0,
    },
    "waterbody_line_mask": {
        "name": "Waterbody_Lines",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "waterbody_point_mask": {"name": "Wells", "type": "Point"},
    "utility_line_mask": {
        "name": "Utility_Lines",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "utility_point_mask": {"name": "Utility_Points", "type": "Point"},
    "bridge_mask": {
        "name": "Bridges",
        "type": "Polygon",
        "min_area": 20,
        "simplify": 0.5,
    },
    "railway_mask": {
        "name": "Railways",
        "type": "LineString",
        "min_length": 10,
        "simplify": 1.0,
    },
}

# YOLO Class Mapping
YOLO_CLASS_INFO = {
    0: {"name": "Wells", "key": "waterbody_point_mask"},
    1: {"name": "Transformers", "key": "utility_point_mask"},
    2: {"name": "Tanks", "key": "utility_point_mask"},
}


def _mask_to_geometries(
    mask: np.ndarray,
    transform: rasterio.Affine,
    threshold: float = 0.5,
    geom_type: str = "Polygon",
    min_val: float = 20.0,
    simplify_tol: float = 0.5,
) -> list:
    """Convert mask to cleaned shapely geometries."""
    binary = (mask > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    if geom_type == "LineString":
        from skimage.morphology import skeletonize

        skeleton = skeletonize(binary)
        shapes = list(
            rasterio_shapes(
                skeleton.astype(np.uint8), mask=skeleton > 0, transform=transform
            )
        )
        geoms = [shape(g) for g, v in shapes if v > 0]
        return [g for g in geoms if g.length > min_val]

    elif geom_type == "Polygon":
        shapes = list(rasterio_shapes(binary, mask=binary > 0, transform=transform))
        geoms = []
        for g, v in shapes:
            poly = shape(g)
            if poly.area < min_val:
                continue
            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            if not poly.is_empty and poly.is_valid:
                geoms.append(poly)
        return geoms

    elif geom_type == "Point":
        from skimage.measure import label, regionprops

        labels = label(binary)
        return [
            Point(transform * (prop.centroid[1], prop.centroid[0]))
            for prop in regionprops(labels)
        ]

    return []


def _roof_mask_to_records(
    roof_mask: np.ndarray,
    transform: rasterio.Affine,
    min_area: float = 10.0,
    simplify_tol: float = 0.5,
) -> List[Dict[str, Any]]:
    """Convert class-index roof mask to polygon records with roof_type labels."""
    records: List[Dict[str, Any]] = []
    for class_id in range(1, min(len(ROOF_LABELS), 5)):
        binary = (roof_mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        shapes = list(rasterio_shapes(binary, mask=binary > 0, transform=transform))
        for geom_raw, val in shapes:
            if val <= 0:
                continue
            poly = shape(geom_raw)
            if poly.area < min_area:
                continue
            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            if poly.is_empty or not poly.is_valid:
                continue
            records.append(
                {
                    "geometry": poly,
                    "class": "Roof_Types",
                    "roof_type": ROOF_LABELS[class_id],
                }
            )
    return records


class GISExporter:
    """Production exporter for ensemble V3 predictions."""

    def __init__(self, output_dir: Path, crs: Any):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crs = crs

    def export(
        self,
        results: Dict[str, Any],
        roof_mask: Optional[np.ndarray] = None,
        transform: Optional[rasterio.Affine] = None,
    ) -> Dict[str, Path]:
        """Main export loop handling both segmentation and detection."""
        exported_paths = {}
        if transform is None:
            logger.error("Transform is required for export.")
            return {}

        # 1. Process Segmentation Masks
        for key, config in FEATURE_CONFIG.items():
            if key not in results or not isinstance(results[key], np.ndarray):
                continue

            logger.info(f"Vectorizing {config['name']}...")
            mask = results[key]

            if key == "roof_type_mask":
                records = _roof_mask_to_records(
                    mask,
                    transform,
                    min_area=float(config.get("min_area", 10)),
                    simplify_tol=float(config.get("simplify", 0.5)),
                )
                if records:
                    exported_paths[key] = self._write_records(records, config["name"])
                continue

            geoms = _mask_to_geometries(
                mask,
                transform,
                geom_type=str(config["type"]),
                min_val=float(config.get("min_area", config.get("min_length", 0))),
                simplify_tol=float(config.get("simplify", 0)),
            )

            if geoms:
                exported_paths[key] = self._write_gpkg(
                    geoms, config["name"], key, transform, roof_mask
                )

        # 2. Process YOLO Detections
        if "detections" in results:
            logger.info("Processing YOLOv8 detections...")
            det_by_key: Dict[str, List[Point]] = {}
            for det in results["detections"]:
                info = YOLO_CLASS_INFO.get(det["class"])
                if info:
                    key = info["key"]
                    if key not in det_by_key:
                        det_by_key[key] = []

                    x1, y1, x2, y2 = det["box"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    world_pt = Point(transform * (cx, cy))
                    det_by_key[key].append(world_pt)

            for key, points in det_by_key.items():
                name = FEATURE_CONFIG[key]["name"]
                exported_paths[f"det_{key}"] = self._write_gpkg(
                    points, name, key, transform
                )

        return exported_paths

    def _write_records(self, records: List[Dict[str, Any]], layer_name: str) -> Path:
        """Write pre-built records to a GeoPackage layer."""
        for i, row in enumerate(records):
            if "id" not in row:
                row["id"] = i
        gdf = gpd.GeoDataFrame(records, crs=self.crs)
        out_path = self.output_dir / f"{layer_name}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        logger.info(f"  Saved {len(gdf)} features to {out_path.name}")
        return out_path

    def _write_gpkg(
        self,
        geoms: list,
        name: str,
        key: str,
        transform: rasterio.Affine,
        roof_mask: Optional[np.ndarray] = None,
    ) -> Path:
        """Helper to write GDF to GeoPackage."""
        records = []
        for i, g in enumerate(geoms):
            # Polygon orientation/topology cleaning
            if isinstance(g, (Polygon, MultiPolygon)) and not g.is_valid:
                g = g.buffer(0)

            data = {"geometry": g, "id": i, "class": name}

            # Roof Attribute logic (Specific to buildings)
            if key == "building_mask" and roof_mask is not None:
                cp = g.representative_point()
                inv_transform = ~transform
                px, py = inv_transform * (cp.x, cp.y)
                r, c = int(py), int(px)
                if 0 <= r < roof_mask.shape[0] and 0 <= c < roof_mask.shape[1]:
                    idx = int(roof_mask[r, c])
                    data["roof_type"] = ROOF_LABELS[min(idx, 4)]

            records.append(data)

        gdf = gpd.GeoDataFrame(records, crs=self.crs)
        out_path = self.output_dir / f"{name}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        logger.info(f"  Saved {len(gdf)} features to {out_path.name}")
        return out_path


def export_predictions(
    results: Dict[str, Any],
    tif_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    roof_type_mask: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Compatibility wrapper."""
    with rasterio.open(tif_path) as src:
        exporter = GISExporter(output_dir, src.crs)
        return exporter.export(
            results, roof_mask=roof_type_mask, transform=src.transform
        )
