"""
PyTorch Dataset for SVAMITVA orthophotos and annotations.

Design decisions:
- Each large MAP TIF is tiled into overlapping 512×512 patches so
  DataLoader gets many samples per MAP instead of one giant image.
- Shapefiles are matched first by explicit glob patterns, then by
  keyword search across all .shp files in the folder (handles
  non-standard naming like 'Abadi.shp', 'built_up.shp', etc.)
- GeoDataFrame is always reprojected to the TIF's CRS before
  rasterization so masks are never empty due to CRS mismatch.
- Bridge and railway shapefiles are optional — if absent, the mask
  stays all-zero (correct: "no such feature here").
- GeoDataFrames are cached per (map_name, task_key) so shapefiles
  are only read and reprojected once per dataset lifetime.

Dataset structure expected:
    DATA/
      MAP1/
        anything.tif
        Road.shp
        Built_Up_Area.shp
        ...
      MAP2/ ... MAP5/
"""

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset, Subset

from .augmentation import get_train_transforms, get_val_transforms
from .preprocessing import OrthophotoPreprocessor, ShapefileAnnotationParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shapefile task definitions ─────────────────────────────────────────────────
# Each entry: (explicit_glob_patterns, keyword_fallback_list, mask_key)
# Keywords matched case-insensitively against all .shp filenames.
# Bridge and Railway are included but gracefully tolerated when absent.
SHAPEFILE_TASKS = [
    (
        ["Build_up*.shp", "Built_Up_Area*.shp", "Abadi*.shp"],
        ["build_up", "built_up", "built", "abadi", "building", "structure"],
        "building",
    ),
    (["Road.shp"], ["^road$"], "road"),
    (
        ["Road_centre_line*.shp", "Road_Centre_Line*.shp"],
        ["centre_line", "centerline", "centreline", "center_line"],
        "road_centerline",
    ),
    (
        ["Waterbody_1*.shp", "Water_Body.shp", "Waterbody.shp", "Water_Body_1.shp"],
        ["^waterbody$", "^waterbody_1$", "^water_body$", "pond", "lake"],
        "waterbody",
    ),
    (
        ["Waterbody_line_1*.shp", "Water_Body_Line*.shp", "Waterbody_Line*.shp"],
        ["waterbody_line", "water_body_line", "canal", "drain"],
        "waterbody_line",
    ),
    (
        ["Waterbody_point_1*.shp", "Waterbody_Point*.shp", "Water_Body_Point*.shp"],
        ["waterbody_point", "water_body_point", "well"],
        "waterbody_point",
    ),
    # utility_point (previously utility_poly)
    (
        ["Utility_poly_1*.shp", "Utility_Poly*.shp"],
        ["utility_poly", "utility_area", "utility_point", "transformer", "tank"],
        "utility_point",
    ),
    (
        ["Utility_1*.shp", "Utility.shp", "Utility_Line*.shp"],
        ["^utility$", "^utility_1$", "utility_line", "pipeline", "wire"],
        "utility_line",
    ),
    # Optional — all-zero mask when absent
    (["Bridge*.shp", "Bridge.shp"], ["bridge"], "bridge"),
    (
        ["Railway*.shp", "Rail*.shp", "Railway.shp"],
        ["railway", "railroad", "rail"],
        "railway",
    ),
]

ALL_MASK_KEYS = [task for _, _, task in SHAPEFILE_TASKS] + ["roof_type"]

# Tasks where missing shapefiles produce a debug log, not a warning
OPTIONAL_TASKS = {"bridge", "railway"}

TILE_SIZE = 512
TILE_OVERLAP = 96  # Increased overlap for better boundary coverage


def _group_sample_indices_by_map(samples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        map_name = str(sample.get("map_name", "unknown"))
        grouped.setdefault(map_name, []).append(idx)
    return grouped


def split_indices_mapwise(
    samples: List[Dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Split indices by map_name to avoid train/val leakage across tiles
    from the same village map.
    """
    grouped = _group_sample_indices_by_map(samples)
    map_names = sorted(grouped.keys())

    if not map_names:
        return [], [], []

    if len(map_names) == 1:
        # No map-wise split possible with a single map; caller may fallback.
        only = map_names[0]
        return [], grouped[only], [only]

    rng = np.random.default_rng(seed)
    shuffled = list(map_names)
    rng.shuffle(shuffled)

    n_val_maps = int(round(len(shuffled) * float(val_split)))
    n_val_maps = max(1, n_val_maps)
    n_val_maps = min(len(shuffled) - 1, n_val_maps)

    val_maps = set(shuffled[:n_val_maps])
    train_idx: List[int] = []
    val_idx: List[int] = []

    for map_name, idxs in grouped.items():
        if map_name in val_maps:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return train_idx, val_idx, sorted(val_maps)


def create_map_kfold_splits(
    samples: List[Dict[str, Any]],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int], List[str]]]:
    """
    Build map-level K-fold splits.
    Returns a list of (train_indices, val_indices, val_map_names).
    """
    grouped = _group_sample_indices_by_map(samples)
    map_names = sorted(grouped.keys())
    if not map_names:
        return []

    effective_splits = max(1, min(int(n_splits), len(map_names)))
    rng = np.random.default_rng(seed)
    shuffled = list(map_names)
    rng.shuffle(shuffled)
    folds = np.array_split(np.array(shuffled, dtype=object), effective_splits)

    all_splits: List[Tuple[List[int], List[int], List[str]]] = []
    for fold_maps_arr in folds:
        fold_maps = set([str(m) for m in fold_maps_arr.tolist()])
        val_idx: List[int] = []
        train_idx: List[int] = []
        for map_name, idxs in grouped.items():
            if map_name in fold_maps:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)
        all_splits.append((train_idx, val_idx, sorted(fold_maps)))
    return all_splits


class SvamitvaDataset(Dataset):
    """
    PyTorch Dataset for SVAMITVA drone imagery.

    Returns 512×512 tile-level samples (not MAP-level).
    Each large orthophoto is split into overlapping tiles during __init__.
    """

    def __init__(
        self,
        root_dirs: Union[Path, List[Path]],
        image_size: int = TILE_SIZE,
        tile_overlap: Optional[int] = None,
        transform: Optional[Callable] = None,
        mode: str = "train",
        tasks: Optional[List[str]] = None,
    ):
        if isinstance(root_dirs, (str, Path)):
            self.root_dirs = [Path(root_dirs)]
        else:
            self.root_dirs = [Path(d) for d in root_dirs]

        self.image_size = int(image_size)
        if self.image_size < 128:
            raise ValueError("image_size must be >= 128")

        if tile_overlap is None:
            # Default overlap ~= 20% tile with floor/ceiling guards.
            self.tile_overlap = int(np.clip(round(self.image_size * 0.2), 32, 256))
        else:
            self.tile_overlap = int(tile_overlap)
        if self.tile_overlap >= self.image_size:
            raise ValueError("tile_overlap must be smaller than image_size")

        self.transform = transform
        self.mode = mode
        self.tasks = tasks  # e.g., ["building", "road"]
        self.ortho_preprocessor = OrthophotoPreprocessor()
        self.anno_parser = ShapefileAnnotationParser()
        self._gdf_cache: Dict[Tuple, gpd.GeoDataFrame] = {}  # keyed (map_name, task)
        self._supervised_mask_keys = self._build_supervised_mask_keys()

        self.samples = self._scan_dataset()
        logger.info(
            f"[{mode}] dataset ready: {len(self.samples)} tile samples from {len(self.root_dirs)} roots"
        )

    def _build_supervised_mask_keys(self) -> List[str]:
        if self.tasks is None:
            keys = [f"{task}_mask" for _, _, task in SHAPEFILE_TASKS]
            keys.append("roof_type_mask")
            return keys

        keys = [f"{task}_mask" for task in self.tasks]
        if "building" in self.tasks:
            keys.append("roof_type_mask")
        return keys

    # ── Shapefile finder ───────────────────────────────────────────────────────

    def _find_shapefile(
        self,
        folder: Path,
        patterns: List[str],
        keywords: List[str],
        taken: set,
    ) -> Optional[Path]:
        """
        Find a shapefile by explicit glob patterns first,
        then keyword fallback across all .shp in folder.
        Skips files already claimed by another task.
        """
        # 1. Explicit glob
        for pat in patterns:
            for hit in sorted(folder.glob(pat)):
                if hit not in taken:
                    return hit

        # 2. Keyword fallback — case-insensitive substring / anchored match
        for shp in sorted(folder.glob("*.shp")):
            stem = shp.stem.lower()
            for kw in keywords:
                if kw.startswith("^") and kw.endswith("$"):
                    if stem == kw[1:-1] and shp not in taken:
                        return shp
                elif kw in stem and shp not in taken:
                    return shp

        return None

    # ── Tile calculator ────────────────────────────────────────────────────────

    def _compute_tiles(self, tif_path: Path):
        """
        Open the TIF, apply K-Means clustering to skip NoData, compute tile windows.

        For MAPC sub-maps (already 512×512), this returns a single tile covering
        the entire image — no tiling overhead needed.
        """
        try:
            with rasterio.open(tif_path) as src:
                H, W = src.height, src.width
                tif_crs = src.crs
                tif_tf = src.transform

                # ── Fast path for pre-clipped MAPC tiles ─────────────────────
                # If the image is already ≤ TILE_SIZE in both dimensions,
                # return a single tile — no K-Means, no overlap logic.
                if H <= self.image_size and W <= self.image_size:
                    return [(0, 0)], H, W, tif_crs, tif_tf

                # Fetch a thumbnail to perform K-Means quickly without OOM
                try:
                    thumb_size = 1024
                    scale = min(thumb_size / max(H, W), 1.0)
                    out_shape = (src.count, int(H * scale), int(W * scale))
                    thumb = src.read(
                        out_shape=out_shape,
                        resampling=rasterio.enums.Resampling.bilinear,
                    )
                    thumb_h, thumb_w = out_shape[1], out_shape[2]
                except Exception as e:
                    logger.warning(f"Could not read thumbnail for KMeans: {e}")
                    thumb = None
                    scale = 1.0

        except Exception as e:
            logger.error(f"Cannot open {tif_path}: {e}")
            return None

        # Apply K-Means
        valid_mask_thumb = None
        if thumb is not None and thumb.shape[0] >= 3:
            try:
                import cv2

                img_rgb = np.transpose(thumb[:3, :, :], (1, 2, 0))
                pixels = img_rgb.reshape(-1, 3).astype(np.float32)

                # K-Means clustering (k=3 or 4 usually separates background, vegetation, built-up)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = 3
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS  # type: ignore
                )

                # The corners of the orthophoto are almost always NoData background
                corner_indices = [
                    0,
                    thumb_w - 1,
                    (thumb_h - 1) * thumb_w,
                    (thumb_h - 1) * thumb_w + thumb_w - 1,
                ]
                corner_labels = labels.flatten()[corner_indices]

                from collections import Counter

                bg_label = Counter(corner_labels).most_common(1)[0][0]

                valid_mask_thumb = (labels.flatten() != bg_label).reshape(
                    thumb_h, thumb_w
                )
            except Exception as e:
                logger.warning(
                    f"K-Means clustering failed, falling back to full map: {e}"
                )
                valid_mask_thumb = None

        stride = self.image_size - self.tile_overlap
        ys_all = list(range(0, max(H - self.image_size, 0) + 1, stride)) or [0]
        xs_all = list(range(0, max(W - self.image_size, 0) + 1, stride)) or [0]

        valid_tiles = []
        for y0 in ys_all:
            for x0 in xs_all:
                if valid_mask_thumb is not None:
                    # Project tile box to thumbnail space
                    tx0 = int(x0 * scale)
                    ty0 = int(y0 * scale)
                    tx1 = int(min(x0 + self.image_size, W) * scale)
                    ty1 = int(min(y0 + self.image_size, H) * scale)

                    # If this tile region in the thumbnail is 100% background, skip it
                    tile_valid = valid_mask_thumb[ty0:ty1, tx0:tx1]
                    # We require at least 5% of the tile to roughly contain some valid pixels
                    if tile_valid.size > 0 and tile_valid.mean() < 0.05:
                        continue

                valid_tiles.append((y0, x0))

        return valid_tiles, H, W, tif_crs, tif_tf

    # ── Dataset scanner ────────────────────────────────────────────────────────

    def _scan_dataset(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        # Collect all map directories (either children of roots or roots themselves)
        candidate_dirs: List[Path] = []
        for root in self.root_dirs:
            if not root.is_dir():
                logger.warning(f"Root directory not found: {root}")
                continue

            # If root contains a .tif, it's a map directory itself
            has_tif = any(root.glob("*.tif")) or any(root.glob("*.TIF"))
            if has_tif:
                candidate_dirs.append(root)
            else:
                # Otherwise, check children
                candidate_dirs.extend(sorted([d for d in root.iterdir() if d.is_dir()]))

        # Stable deterministic ordering and de-duplication.
        candidate_dirs = sorted(list(dict.fromkeys(candidate_dirs)))

        if not candidate_dirs:
            raise ValueError(f"No map directories found in {self.root_dirs}")

        for map_dir in candidate_dirs:
            # ── Find orthophoto ──────────────────────────────────────────────
            ortho = None
            for ext in [".tif", ".tiff", ".TIF", ".TIFF", ".ecw", ".ECW"]:
                hits = list(map_dir.glob(f"*{ext}"))
                if hits:
                    ortho = hits[0]
                    break

            if ortho is None:
                logger.warning(f"No orthophoto in {map_dir}, skipping")
                continue

            # ── Find shapefiles ──────────────────────────────────────────────
            annotations: Dict[str, Path] = {}
            taken: set = set()
            for patterns, keywords, task_key in SHAPEFILE_TASKS:
                # 🎯 Filter: Only process if task is in requested self.tasks
                if self.tasks is not None and task_key not in self.tasks:
                    continue

                shp = self._find_shapefile(map_dir, patterns, keywords, taken)
                if shp:
                    annotations[task_key] = shp
                    taken.add(shp)
                else:
                    if task_key in OPTIONAL_TASKS:
                        logger.debug(
                            f"[{map_dir.name}] Optional '{task_key}' not found — all-zero mask"
                        )
                    else:
                        logger.warning(
                            f"[{map_dir.name}] No shapefile for '{task_key}'"
                        )

            if not annotations:
                logger.warning(f"No shapefiles in {map_dir}, skipping")
                continue

            logger.info(
                f"[{map_dir.name}] {ortho.name} | tasks: {', '.join(annotations.keys())}"
            )

            # ── Compute tiles ────────────────────────────────────────────────
            result = self._compute_tiles(ortho)
            if result is None:
                continue
            tiles, H, W, tif_crs, tif_tf = result

            n_before = len(samples)
            for y0, x0 in tiles:
                y1 = min(y0 + self.image_size, H)
                x1 = min(x0 + self.image_size, W)
                win = Window(x0, y0, x1 - x0, y1 - y0)
                samples.append(
                    {
                        "map_name": map_dir.name,
                        "tif_path": ortho,
                        "annotations": annotations,
                        "window": win,
                        "tif_crs": tif_crs,
                        "tif_tf": tif_tf,
                        "H": H,
                        "W": W,
                    }
                )
            logger.info(
                f"  → {len(samples) - n_before} tiles from {map_dir.name} ({H}×{W}px)"
            )

        if not samples:
            raise ValueError(
                f"No tiles found in {self.root_dirs}. "
                "Each MAP folder needs a .tif + at least one .shp."
            )

        return samples

    # ── GeoDataFrame cache ─────────────────────────────────────────────────────

    def _get_cached_gdf(
        self,
        map_name: str,
        task_key: str,
        path: Path,
        target_crs,
    ) -> Optional[gpd.GeoDataFrame]:
        cache_key = (map_name, task_key)
        if cache_key in self._gdf_cache:
            return self._gdf_cache[cache_key]

        try:
            gdf = gpd.read_file(path)
            # FORCE strict projection alignment
            if target_crs is not None:
                if gdf.crs is None:
                    gdf.set_crs(target_crs, inplace=True)
                elif gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)

            # Drop null / invalid geometries up front (saves repeated checks)
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(
                drop=True
            )
            self._gdf_cache[cache_key] = gdf
            return gdf
        except Exception as e:
            logger.error(f"Error loading GDF {path}: {e}")
            return None

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._load_tile(idx, retries=0)

    def _load_tile(self, idx: int, retries: int = 0) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        win = sample["window"]

        # ── Load tile pixels ─────────────────────────────────────────────────
        try:
            with rasterio.open(sample["tif_path"]) as src:
                tile_data = src.read(
                    window=win,
                    out_shape=(src.count, self.image_size, self.image_size),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                tile_tf = src.window_transform(win)
                tif_crs = src.crs
        except Exception as e:
            logger.error(f"Error reading tile {sample['tif_path']}: {e}")
            tile_data = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
            tile_tf = sample["tif_tf"]
            tif_crs = sample["tif_crs"]

        # Build (H,W,C) float32 in [0,1]
        if tile_data.shape[0] >= 3:
            image = np.stack([tile_data[0], tile_data[1], tile_data[2]], axis=-1)
        else:
            image = np.stack([tile_data[0]] * 3, axis=-1)

        # Apply robust percentile stretch for stable remote-sensing normalization
        vmin, vmax = np.percentile(image, (2, 98))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin)
        image = image.astype(np.float32)

        # Numerical stability: replace NaN/Inf from NoData regions, clamp to [0,1]
        image = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # ── Valid Data Mask (NoData skipping) ─────────────────────────────────
        # User requested skipping non-value pixels. We define 'valid' as pixels
        # where at least one channel has a non-zero value.
        # This mask is used to zero-out the loss in black/empty areas.
        valid_mask = (image.sum(axis=-1) > 0.01).astype(np.uint8)

        # ── Build masks ───────────────────────────────────────────────────────
        output_shape = (self.image_size, self.image_size)
        masks: Dict[str, np.ndarray] = {"valid_mask": valid_mask}

        for _, _, task_key in SHAPEFILE_TASKS:
            # 🎯 Filter: Only generate mask if task is in requested self.tasks
            if self.tasks is not None and task_key not in self.tasks:
                continue

            mask_key = f"{task_key}_mask"
            if task_key in sample["annotations"]:
                gdf = self._get_cached_gdf(
                    sample["map_name"],
                    task_key,
                    sample["annotations"][task_key],
                    tif_crs,
                )
                if gdf is not None and len(gdf) > 0:
                    masks[mask_key] = self.anno_parser.rasterize_annotations(
                        gdf, tile_tf, output_shape, task_key
                    )
                else:
                    masks[mask_key] = np.zeros(output_shape, dtype=np.uint8)
            else:
                masks[mask_key] = np.zeros(output_shape, dtype=np.uint8)

        # ── Roof type mask ────────────────────────────────────────────────────
        if "building" in sample["annotations"]:
            bgdf = self._get_cached_gdf(
                sample["map_name"],
                "building",
                sample["annotations"]["building"],
                tif_crs,
            )
            if bgdf is not None and len(bgdf) > 0:
                masks["roof_type_mask"] = self.anno_parser.extract_roof_types(
                    bgdf, tile_tf, output_shape
                )
            else:
                masks["roof_type_mask"] = np.zeros(output_shape, dtype=np.uint8)
        else:
            masks["roof_type_mask"] = np.zeros(output_shape, dtype=np.uint8)

        # ── Augmentation / tensor conversion ──────────────────────────────────
        # Skip pure-background tiles in training mode — they carry no
        # supervision signal and waste GPU cycles.  Resample to a random
        # other tile instead (up to 3 retries to avoid infinite loops).
        if self.mode == "train" and retries < 3:
            any_positive = any(
                masks[k].sum() > 0 for k in self._supervised_mask_keys if k in masks
            )
            if not any_positive and len(self.samples) > 1:
                new_idx = np.random.randint(0, len(self.samples))
                return self._load_tile(new_idx, retries=retries + 1)

        if self.transform:
            transformed = self.transform(image=image, **masks)
            result: Dict[str, Any] = {"image": transformed["image"]}
            for k in masks:
                result[k] = transformed[k].long()
        else:
            result = {"image": torch.from_numpy(image).permute(2, 0, 1).float()}  # type: ignore
            for k, v in masks.items():
                result[k] = torch.from_numpy(v).long()

        # result["valid_mask"] comes from transformed successfully if self.transform exists
        if not self.transform:
            result["valid_mask"] = torch.from_numpy(valid_mask).float()

        result["metadata"] = {
            "map_name": sample["map_name"],
            "idx": idx,
            "tile_x": win.col_off,
            "tile_y": win.row_off,
        }
        return result


# ── DataLoader factory ─────────────────────────────────────────────────────────


def create_dataloaders(
    train_dirs: List[Path],
    val_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: int = TILE_SIZE,
    tile_overlap: Optional[int] = None,
    val_split: float = 0.2,
    split_mode: str = "map",
    seed: int = 42,
    max_train_tiles: Optional[int] = None,
    max_val_tiles: Optional[int] = None,
    distributed: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation DataLoaders.

    If val_dir is provided, creates a separate validation dataset.
    Otherwise, splits training data by val_split fraction.
    """
    train_ds = SvamitvaDataset(
        train_dirs,
        image_size=image_size,
        tile_overlap=tile_overlap,
        transform=get_train_transforms(image_size),
        mode="train",
    )
    tr_ds_final: Dataset[Any]
    val_ds_final: Dataset[Any]

    if val_dir is not None:
        val_ds_full = SvamitvaDataset(
            val_dir,
            image_size=image_size,
            tile_overlap=tile_overlap,
            transform=get_val_transforms(image_size),
            mode="val",
        )
        tr_ds_final = train_ds
        val_ds_final = val_ds_full
        logger.info(
            f"Separate val dir: {len(tr_ds_final)} train / {len(val_ds_final)} val tiles"
        )
    else:
        val_ds = SvamitvaDataset(
            train_dirs,
            image_size=image_size,
            tile_overlap=tile_overlap,
            transform=get_val_transforms(image_size),
            mode="val",
        )
        total = len(train_ds)
        if len(val_ds) != total:
            raise RuntimeError(
                "Train/val dataset scans differ in tile count for the same train_dirs. "
                "Check deterministic dataset scanning."
            )

        split_mode_norm = str(split_mode).strip().lower()
        if split_mode_norm not in {"map", "tile"}:
            logger.warning(
                "Unknown split_mode='%s'; falling back to map split.", split_mode
            )
            split_mode_norm = "map"

        if split_mode_norm == "map":
            train_idx, val_idx, val_maps = split_indices_mapwise(
                train_ds.samples, val_split=val_split, seed=seed
            )
            if not train_idx:
                logger.warning(
                    "Map-wise split not possible (likely one map). "
                    "Falling back to tile-wise split."
                )
                split_mode_norm = "tile"
            else:
                logger.info(
                    "Auto-split (map-wise): %d train / %d val tiles | val maps: %s",
                    len(train_idx),
                    len(val_idx),
                    ", ".join(val_maps),
                )

        if split_mode_norm == "tile":
            val_size = max(1, int(math.ceil(total * val_split)))
            val_size = min(total - 1, val_size) if total > 1 else 1
            rng = np.random.default_rng(seed)
            perm = rng.permutation(total).tolist()
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
            logger.info(
                "Auto-split (tile-wise): %d train / %d val tiles",
                len(train_idx),
                len(val_idx),
            )

        tr_ds_final = Subset(train_ds, train_idx)
        val_ds_final = Subset(val_ds, val_idx)

    # Optional caps for fast smoke runs on very large maps.
    if (
        max_train_tiles is not None
        and max_train_tiles > 0
        and len(tr_ds_final) > max_train_tiles
    ):
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(tr_ds_final), size=int(max_train_tiles), replace=False
        ).tolist()
        tr_ds_final = Subset(tr_ds_final, idx)
        logger.info("Applied max_train_tiles=%d", len(tr_ds_final))

    if (
        max_val_tiles is not None
        and max_val_tiles > 0
        and len(val_ds_final) > max_val_tiles
    ):
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(
            len(val_ds_final), size=int(max_val_tiles), replace=False
        ).tolist()
        val_ds_final = Subset(val_ds_final, idx)
        logger.info("Applied max_val_tiles=%d", len(val_ds_final))

    # Distributed Samplers
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_ds_final)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds_final, shuffle=False
        )

    train_loader = torch.utils.data.DataLoader(
        tr_ds_final,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=len(tr_ds_final) > batch_size,
        persistent_workers=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds_final,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    logger.info(
        f"DataLoaders ready (Distributed={distributed}): "
        f"{len(train_loader)} train batches, {len(val_loader)} val batches"
    )
    return train_loader, val_loader


def create_kfold_dataloaders(
    train_dirs: List[Path],
    n_splits: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = TILE_SIZE,
    tile_overlap: Optional[int] = None,
    seed: int = 42,
    distributed: bool = False,
) -> List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]]:
    """
    Create map-level K-fold train/val DataLoaders.
    Returns list of (train_loader, val_loader, val_map_names).
    """
    train_ds = SvamitvaDataset(
        train_dirs,
        image_size=image_size,
        tile_overlap=tile_overlap,
        transform=get_train_transforms(image_size),
        mode="train",
    )
    val_ds = SvamitvaDataset(
        train_dirs,
        image_size=image_size,
        tile_overlap=tile_overlap,
        transform=get_val_transforms(image_size),
        mode="val",
    )
    if len(train_ds) != len(val_ds):
        raise RuntimeError(
            "Train/val dataset size mismatch while building k-fold loaders."
        )

    splits = create_map_kfold_splits(train_ds.samples, n_splits=n_splits, seed=seed)
    from torch.utils.data import Subset

    fold_loaders: List[
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]
    ] = []
    for train_idx, val_idx, val_maps in splits:
        tr_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)

        # Distributed Samplers
        tr_sampler = None
        val_sampler = None
        if distributed:
            tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_subset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_subset, shuffle=False
            )

        tr_loader = torch.utils.data.DataLoader(
            tr_subset,
            batch_size=batch_size,
            shuffle=(tr_sampler is None),
            sampler=tr_sampler,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            drop_last=len(tr_subset) > batch_size,
            persistent_workers=(num_workers > 0),
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            persistent_workers=(num_workers > 0),
        )
        fold_loaders.append((tr_loader, val_loader, val_maps))

    logger.info(
        "Built %d map-level folds from %d tiles (%d maps).",
        len(fold_loaders),
        len(train_ds),
        len(_group_sample_indices_by_map(train_ds.samples)),
    )
    return fold_loaders
