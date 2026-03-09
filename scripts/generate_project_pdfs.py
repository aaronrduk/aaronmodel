#!/usr/bin/env python3
"""
Generate detailed PDF documentation for the SVAMITVA PS-1 project.

Outputs:
  - reports/SVAMITVA_Model_Documentation.pdf
  - reports/SVAMITVA_Final_Report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch
from fpdf import FPDF


BINARY_TASKS = [
    "building",
    "road",
    "road_centerline",
    "waterbody",
    "waterbody_line",
    "waterbody_point",
    "utility_line",
    "utility_point",
    "bridge",
    "railway",
]

OUTPUT_FEATURES: List[Tuple[str, str, str]] = [
    ("building_mask", "Built-up area", "Polygon"),
    ("roof_type_mask", "Roof classification (RCC/Tiled/Tin/Others)", "Polygon"),
    ("road_mask", "Road", "Polygon"),
    ("road_centerline_mask", "Road centre line", "Line"),
    ("waterbody_mask", "Water body", "Polygon"),
    ("waterbody_line_mask", "Water body line", "Line"),
    ("waterbody_point_mask", "Waterbody point (wells)", "Point"),
    ("utility_line_mask", "Utility line (pipeline/wires)", "Line"),
    ("utility_point_mask", "Utility point (transformers/tanks)", "Point"),
    ("bridge_mask", "Bridge", "Polygon"),
    ("railway_mask", "Railway", "Line"),
]


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def _run_cmd(cmd: List[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def load_best_checkpoint_metrics(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "exists": False,
            "path": str(checkpoint_path),
            "epoch": None,
            "best_score": None,
            "metrics": {},
        }

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    return {
        "exists": True,
        "path": str(checkpoint_path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_score": ckpt.get("best_score") if isinstance(ckpt, dict) else None,
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def gather_project_context(repo_root: Path, checkpoint_path: Path) -> Dict[str, Any]:
    metrics_blob = load_best_checkpoint_metrics(checkpoint_path)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _safe_version("torch"),
        "numpy": _safe_version("numpy"),
        "rasterio": _safe_version("rasterio"),
        "geopandas": _safe_version("geopandas"),
        "streamlit": _safe_version("streamlit"),
        "albumentations": _safe_version("albumentations"),
        "ultralytics": _safe_version("ultralytics"),
    }

    git = {
        "branch": _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "commit": _run_cmd(["git", "rev-parse", "HEAD"], repo_root),
        "status_short": _run_cmd(["git", "status", "--short"], repo_root),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "checkpoint": metrics_blob,
        "env": env,
        "git": git,
    }


class ReportPDF(FPDF):
    def __init__(self, title: str, subtitle: str = "") -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = title
        self.report_subtitle = subtitle
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, self.report_title, new_x="LMARGIN", new_y="NEXT", align="L")
        if self.report_subtitle:
            self.set_font("Helvetica", "", 8)
            self.cell(0, 4, self.report_subtitle, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(1)
        self.set_draw_color(180, 180, 180)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def add_cover(self, heading: str, subheading: str, meta_lines: Iterable[str]) -> None:
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_x(self.l_margin)
        self.multi_cell(0, 10, heading)
        self.ln(2)
        self.set_font("Helvetica", "", 12)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, subheading)
        self.ln(4)
        self.set_font("Helvetica", "", 10)
        for line in meta_lines:
            self.set_x(self.l_margin)
            self.multi_cell(0, 6, line)
        self.ln(3)

    def section(self, title: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(15, 15, 15)
        self.set_x(self.l_margin)
        self.multi_cell(0, 7, title)
        self.ln(1)

    def subsection(self, title: str) -> None:
        self.ln(1)
        self.set_font("Helvetica", "B", 11)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, title)

    def paragraph(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def bullet_list(self, items: Iterable[str]) -> None:
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.5, f"- {item}")
        self.ln(0.5)

    def simple_table(self, headers: List[str], rows: List[List[str]], col_widths: List[float]) -> None:
        self.set_font("Helvetica", "B", 9)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, align="L")
        self.ln()
        self.set_font("Helvetica", "", 8.8)
        for row in rows:
            x_start = self.get_x()
            y_start = self.get_y()
            max_height = 0.0
            for i, val in enumerate(row):
                x = self.get_x()
                y = self.get_y()
                self.multi_cell(col_widths[i], 5, str(val), border=1)
                cell_height = self.get_y() - y
                max_height = max(max_height, cell_height)
                self.set_xy(x + col_widths[i], y)
            self.set_xy(x_start, y_start + max_height)
        self.ln(1.5)


def _pct(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.2f}%"


def _fmt(v: float | int | None, nd: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{nd}f}"


def build_model_documentation_pdf(out_path: Path, context: Mapping[str, Any]) -> None:
    ck_raw = context.get("checkpoint", {})
    ck: Dict[str, Any] = ck_raw if isinstance(ck_raw, dict) else {}
    metrics_raw = ck.get("metrics", {})
    metrics: Dict[str, float] = metrics_raw if isinstance(metrics_raw, dict) else {}
    env_raw = context.get("env", {})
    env: Dict[str, Any] = env_raw if isinstance(env_raw, dict) else {}
    git_raw = context.get("git", {})
    git: Dict[str, Any] = git_raw if isinstance(git_raw, dict) else {}

    pdf = ReportPDF(
        title="SVAMITVA PS-1 Model Documentation",
        subtitle="Architecture, Training Process, and Deployment Guidelines",
    )
    pdf.add_cover(
        heading="SVAMITVA Feature Extraction System Documentation",
        subheading=(
            "Problem Statement 1 only: orthophoto feature extraction for building footprints, roof classes, roads, "
            "waterbodies, utilities, bridges, and railway using a SAM2-based multi-task architecture."
        ),
        meta_lines=[
            f"Generated (UTC): {context.get('generated_at', 'unknown')}",
            f"Repository: {context.get('repo_root', 'unknown')}",
            f"Git branch: {git.get('branch', 'unknown')}",
            f"Git commit: {git.get('commit', 'unknown')}",
            f"Primary checkpoint: {ck.get('path', 'unknown')}",
        ],
    )

    pdf.section("1. Scope and Objectives")
    pdf.paragraph(
        "This document defines the current production-oriented design for the SVAMITVA drone orthophoto extraction system. "
        "The codebase intentionally excludes point-cloud DTM/drainage modeling (Problem Statement 2) and focuses only on "
        "Problem Statement 1 deliverables."
    )
    pdf.bullet_list(
        [
            "Extract geospatial features from large orthophotos with tiled inference and GIS export.",
            "Support 11 output layers covering polygon, line, and point geometries.",
            "Maintain train/infer consistency in normalization, class schema, and thresholding.",
            "Provide deployable interfaces via Python APIs and Streamlit UI.",
        ]
    )

    pdf.section("2. Feature Catalog and Output Contract")
    pdf.paragraph("The model outputs follow a fixed key contract used during training, inference, and GIS vectorization.")
    rows = [[k, f, g] for (k, f, g) in OUTPUT_FEATURES]
    pdf.simple_table(
        headers=["Output key", "Target feature", "Geometry"],
        rows=rows,
        col_widths=[48, 94, 38],
    )

    pdf.section("3. System Architecture")
    pdf.subsection("3.1 High-level data flow")
    pdf.bullet_list(
        [
            "Input raster map is split into overlapping tiles.",
            "Each tile goes through percentile stretch + ImageNet normalization.",
            "SAM2 encoder extracts hierarchical multi-scale features.",
            "FPN + CBAM decoder fuses features into a shared task tensor.",
            "Task-specific heads produce 10 binary masks + 1 roof multi-class mask.",
            "YOLOv8 detects sparse point objects and fuses with point masks.",
            "Tile outputs are blended with Gaussian weights and stitched globally.",
            "Final masks are vectorized to GeoPackage layers with geometry cleaning.",
        ]
    )

    pdf.subsection("3.2 Backbone")
    pdf.paragraph(
        "Primary backbone is SAM2 Hiera B+ (4-scale feature hierarchy). If SAM2 cannot be imported, the code falls back "
        "to a ResNet50 encoder for continuity of execution, but this fallback should be treated as degraded mode for "
        "accuracy-critical production runs."
    )
    pdf.bullet_list(
        [
            "Expected checkpoint: checkpoints/sam2.1_hiera_base_plus.pt",
            "Model config reference: configs/sam2.1/sam2.1_hiera_b+.yaml",
            "Feature levels: feat_s4, feat_s8, feat_s16, feat_s32",
        ]
    )

    pdf.subsection("3.3 Decoder and Attention")
    pdf.paragraph(
        "The decoder is an FPN-style top-down path with lateral projections, smoothing blocks, and CBAM attention "
        "(channel + spatial). All scale features are upsampled to the finest resolution, concatenated, and fused to a "
        "256-channel shared feature map consumed by all task heads."
    )

    pdf.subsection("3.4 Prediction Heads")
    pdf.bullet_list(
        [
            "BuildingHead: dual-output for building mask and roof-type classes.",
            "BinaryHead: polygon-type tasks such as road, waterbody, bridge.",
            "LineHead (D-Link style): line continuity tasks (road centerline, utility line, railway, waterbody line).",
            "DetectionHead: sparse point-like masks (waterbody_point, utility_point).",
        ]
    )

    pdf.section("4. Data Engineering and Annotation Processing")
    pdf.subsection("4.1 Dataset scanning and shapefile matching")
    pdf.paragraph(
        "The dataset loader supports explicit filename globs and keyword fallback matching for shapefiles. This allows "
        "operation on non-uniform map folder naming conventions. Geometry files are reprojected to raster CRS before "
        "rasterization to avoid empty masks from CRS mismatch."
    )
    pdf.subsection("4.2 Tiling strategy for large maps")
    pdf.bullet_list(
        [
            "Tile size is configurable (default 512; large-map usage typically 1024+).",
            "Overlap is configurable (default 96 in training, higher in inference for seam quality).",
            "A valid-data mask skips near-empty no-data tiles to reduce wasted compute.",
            "Training mode resamples pure-background tiles to improve label signal density.",
        ]
    )
    pdf.subsection("4.3 Mask rasterization specifics")
    pdf.bullet_list(
        [
            "Line and point annotations are buffered before rasterization to generate learnable supervision.",
            "Optional local majority refinement (KNN-style) denoises sparse binary masks.",
            "Roof class extraction maps category strings to class ids: 0 background, 1 RCC, 2 Tiled, 3 Tin, 4 Others.",
        ]
    )

    pdf.section("5. Augmentation and Preprocessing")
    pdf.paragraph(
        "Training augmentations include safe geometric transforms (rotate/flip/affine), color perturbations, CLAHE, "
        "limited blur/noise, coarse dropout, and tensor conversion. Validation path uses deterministic resize + normalize. "
        "Input normalization is aligned between training and inference."
    )

    pdf.section("6. Training Process")
    pdf.subsection("6.1 Optimization stack")
    pdf.bullet_list(
        [
            "Optimizer: AdamW (default) or SGD.",
            "Scheduler: OneCycleLR for AdamW, warmup-cosine for SGD.",
            "Mixed precision enabled on CUDA.",
            "Gradient clipping and NaN/Inf batch skipping.",
            "Early stopping on average IoU.",
        ]
    )
    pdf.subsection("6.2 Loss design")
    pdf.paragraph(
        "Binary tasks use a composite loss combining BCE, Dice, Focal, Lovasz-Hinge, and Boundary loss, with valid-mask "
        "gating for no-data exclusion and OHEM in BCE component. Roof classification uses CE(ignore background) + "
        "multi-class Dice. Final loss is weighted by task-specific coefficients."
    )
    pdf.subsection("6.3 Train/val splitting policy")
    pdf.bullet_list(
        [
            "Preferred split mode: map-wise to prevent tile leakage across same village map.",
            "Automatic fallback to tile-wise split for single-map scenarios.",
            "K-fold map-level training utility available for low-map datasets.",
        ]
    )

    pdf.section("7. Inference and Post-processing")
    pdf.subsection("7.1 Tiled inference")
    pdf.bullet_list(
        [
            "Sliding-window inference with Gaussian blending to suppress seam artifacts.",
            "Optional test-time augmentation by flips.",
            "Thresholded binary masks and argmax roof mask generation.",
            "YOLO detections merged via class-wise NMS and fused into point masks.",
        ]
    )
    pdf.subsection("7.2 Vector export")
    pdf.paragraph(
        "Mask-to-vector conversion supports Polygon, LineString (with skeletonization), and Point layers. Exporter applies "
        "minimum area/length filters, geometry simplification, topology cleaning, and roof attribute transfer into "
        "building polygons. Output format is GeoPackage."
    )

    pdf.section("8. Deployment Guidelines")
    pdf.subsection("8.1 Runtime environment baseline")
    env_rows = [[k, str(v)] for k, v in env.items()]
    pdf.simple_table(headers=["Component", "Version"], rows=env_rows, col_widths=[60, 120])

    pdf.subsection("8.2 Recommended hardware profile")
    pdf.bullet_list(
        [
            "Training (recommended): NVIDIA GPU >= 16 GB VRAM; CPU >= 8 cores; RAM >= 64 GB for 2+ GB maps.",
            "Inference (recommended): GPU >= 8 GB VRAM for responsive UX on large orthophotos.",
            "Storage: fast SSD with at least 200 GB free for map cache, checkpoints, and GIS outputs.",
        ]
    )

    pdf.subsection("8.3 Production runbook")
    pdf.bullet_list(
        [
            "Pin environment and dependencies in a dedicated virtual environment.",
            "Store only approved model artifacts under checkpoints/ and keep data external to git.",
            "Use map-wise validation in all benchmark runs intended for reporting.",
            "Calibrate task thresholds before final export to maximize IoU/F1 per class.",
            "Perform smoke inference on at least one full-size map before release.",
            "Track model hash, config, and metrics in release metadata for auditability.",
        ]
    )
    pdf.subsection("8.4 Security and data governance")
    pdf.bullet_list(
        [
            "Do not embed personal paths, credentials, or private records in source or logs.",
            "Strip debug artifacts and local machine metadata before repository publication.",
            "Apply access control for raw village imagery and generated geospatial outputs.",
        ]
    )

    pdf.section("9. Current Checkpoint Snapshot")
    if ck.get("exists"):
        pdf.paragraph(
            f"Checkpoint loaded: {ck.get('path')} | epoch={ck.get('epoch')} | "
            f"best_score(avg_iou)={_fmt(ck.get('best_score'), 6)}"
        )
        summary_rows = [
            ["avg_iou", _fmt(metrics.get("avg_iou"), 6), _pct(metrics.get("avg_iou"))],
            ["avg_dice", _fmt(metrics.get("avg_dice"), 6), _pct(metrics.get("avg_dice"))],
            ["val_loss", _fmt(metrics.get("val_loss"), 6), "NA"],
        ]
        pdf.simple_table(
            headers=["Metric", "Value", "Percentage view"],
            rows=summary_rows,
            col_widths=[45, 55, 80],
        )
    else:
        pdf.paragraph("No checkpoint metrics were found. Train and evaluate to populate this section.")

    pdf.section("10. Known Limitations")
    pdf.bullet_list(
        [
            "Current measured metrics (from available checkpoint) are below the 95% target; additional training/data curation is required.",
            "Single-map self-validation can overestimate or underestimate generalization depending on class sparsity.",
            "Point-feature performance is sensitive to YOLO class calibration and object density.",
            "Sparse classes (bridge/railway/utilities) need stronger class balancing or targeted sampling.",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))


def build_final_report_pdf(out_path: Path, context: Mapping[str, Any]) -> None:
    ck_raw = context.get("checkpoint", {})
    ck: Dict[str, Any] = ck_raw if isinstance(ck_raw, dict) else {}
    metrics_raw = ck.get("metrics", {})
    metrics: Dict[str, float] = metrics_raw if isinstance(metrics_raw, dict) else {}
    git_raw = context.get("git", {})
    git: Dict[str, Any] = git_raw if isinstance(git_raw, dict) else {}

    avg_iou = float(metrics.get("avg_iou", 0.0))
    avg_dice = float(metrics.get("avg_dice", 0.0))
    target = 0.95
    gap_iou = target - avg_iou
    gap_dice = target - avg_dice

    pdf = ReportPDF(
        title="SVAMITVA PS-1 Final Project Report",
        subtitle="Outcomes, Metrics, Risks, and Improvement Plan",
    )
    pdf.add_cover(
        heading="Final Report: SVAMITVA Orthophoto Feature Extraction",
        subheading=(
            "This report summarizes observed outcomes from the current repository checkpoint and maps them to the "
            "hackathon target requirements."
        ),
        meta_lines=[
            f"Generated (UTC): {context.get('generated_at', 'unknown')}",
            f"Git branch: {git.get('branch', 'unknown')}",
            f"Git commit: {git.get('commit', 'unknown')}",
            f"Checkpoint assessed: {ck.get('path', 'unknown')}",
        ],
    )

    pdf.section("1. Executive Summary")
    pdf.paragraph(
        "A unified SAM2-based multi-task model and deployment pipeline has been implemented for Problem Statement 1. "
        "The repository includes training, map-wise split logic, tiled inference, point-feature fusion with YOLOv8, "
        "and GIS export. A validated checkpoint exists and can run end-to-end inference. However, the current measured "
        "accuracy is materially below the 95% target, so this should be considered a functional baseline, not final "
        "competition-grade accuracy."
    )

    pdf.section("2. Project Scope Achieved")
    pdf.bullet_list(
        [
            "Problem Statement 1 coverage only; Problem Statement 2 excluded as requested.",
            "11 output layers implemented with consistent output key contract.",
            "Training and evaluation pipelines runnable in py314 environment.",
            "Large-map tiling and overlap controls integrated for training and inference.",
            "GIS export to GeoPackage supported for all feature categories.",
        ]
    )

    pdf.section("3. Evaluation Basis")
    pdf.paragraph(
        "Metrics in this report are taken from the repository checkpoint metadata (best.pt). "
        "This corresponds to a short verification run and should be interpreted as a baseline indicator."
    )
    if ck.get("exists"):
        table_rows = [
            ["Epoch", str(ck.get("epoch", "NA"))],
            ["Best score (avg_iou)", _fmt(ck.get("best_score"), 6)],
            ["avg_iou", _fmt(avg_iou, 6)],
            ["avg_dice", _fmt(avg_dice, 6)],
            ["val_loss", _fmt(metrics.get("val_loss"), 6)],
        ]
        pdf.simple_table(headers=["Item", "Value"], rows=table_rows, col_widths=[70, 110])
    else:
        pdf.paragraph("No checkpoint file found; quantitative sections are unavailable.")

    pdf.section("4. Per-feature Metrics Snapshot")
    metric_rows: List[List[str]] = []
    for task in BINARY_TASKS:
        iou = metrics.get(f"{task}_iou")
        dice = metrics.get(f"{task}_dice")
        prec = metrics.get(f"{task}_precision")
        rec = metrics.get(f"{task}_recall")
        metric_rows.append(
            [
                task,
                _fmt(iou, 6),
                _fmt(dice, 6),
                _fmt(prec, 6),
                _fmt(rec, 6),
            ]
        )
    metric_rows.append(
        [
            "roof_type_accuracy",
            _fmt(metrics.get("roof_type_accuracy"), 6),
            "NA",
            "NA",
            "NA",
        ]
    )
    pdf.simple_table(
        headers=["Task", "IoU/Acc", "Dice", "Precision", "Recall"],
        rows=metric_rows,
        col_widths=[42, 34, 34, 34, 36],
    )

    pdf.section("5. Target Compliance Assessment")
    compliance_rows = [
        ["Target metric", "95.00%"],
        ["Current avg_iou", _pct(avg_iou)],
        ["Current avg_dice", _pct(avg_dice)],
        ["Gap vs target (IoU)", _pct(gap_iou)],
        ["Gap vs target (Dice)", _pct(gap_dice)],
    ]
    pdf.simple_table(headers=["Criterion", "Value"], rows=compliance_rows, col_widths=[90, 90])
    pdf.paragraph(
        "Conclusion: the present checkpoint does not satisfy the 95% feature identification objective. "
        "The pipeline is operational, but significant model/data optimization is required for competition-ready performance."
    )

    pdf.section("6. Root-cause Analysis (Current Baseline)")
    pdf.bullet_list(
        [
            "Limited effective training signal for sparse classes (bridge, railway, utility points/lines).",
            "Short verification training horizon is insufficient for convergence on multi-task setup.",
            "Single-map self-validation limits representativeness and increases variance.",
            "Potential class imbalance and annotation sparsity reduce IoU stability across tasks.",
            "Thresholds likely uncalibrated for per-task optimal operating points.",
        ]
    )

    pdf.section("7. Recommendations for Future Improvements")
    pdf.subsection("7.1 Data and labeling")
    pdf.bullet_list(
        [
            "Expand supervised map count from 5 to 10+ with strict map-wise split.",
            "Audit shapefile quality for topology errors, missing classes, and CRS consistency.",
            "Generate class-balanced tile sampling pools, especially for sparse/line/point classes.",
            "Use targeted hard-example mining for rare features and boundary-heavy regions.",
        ]
    )
    pdf.subsection("7.2 Training strategy")
    pdf.bullet_list(
        [
            "Run staged head-wise training followed by joint fine-tuning (already scaffolded in notebook).",
            "Increase epochs substantially (for example 50-150) with early stopping and checkpoint sweeps.",
            "Apply k-fold map-level cross-validation and aggregate fold metrics for robust reporting.",
            "Calibrate task-specific thresholds via validation search before final export.",
            "Tune per-task loss weights and sampling ratios to raise low-performing classes.",
        ]
    )
    pdf.subsection("7.3 Inference and deployment")
    pdf.bullet_list(
        [
            "Use larger inference tiles (1024-1536) with tuned overlap for seam reduction on 2 GB maps.",
            "Enable TTA only for final benchmark runs due to latency cost.",
            "Version model + thresholds together to ensure reproducible exports.",
            "Add automated regression checks on representative maps before each release.",
        ]
    )

    pdf.section("8. Suggested Improvement Roadmap")
    roadmap_rows = [
        ["Phase 1 (1-2 weeks)", "Data QA + class balancing + threshold calibration", "High"],
        ["Phase 2 (2-3 weeks)", "Map-wise k-fold training and hyperparameter sweep", "High"],
        ["Phase 3 (1-2 weeks)", "Sparse class specialization and YOLO label refinement", "Medium-High"],
        ["Phase 4 (1 week)", "Deployment hardening, profiling, and GIS QA", "Medium"],
    ]
    pdf.simple_table(
        headers=["Phase", "Key activities", "Expected impact"],
        rows=roadmap_rows,
        col_widths=[45, 105, 30],
    )

    pdf.section("9. Deployment-readiness Checklist")
    checklist = [
        "Environment pinned and reproducible.",
        "Model checkpoint validated on representative maps.",
        "Per-task thresholds calibrated and versioned.",
        "No personal paths, credentials, or local artifacts in tracked files.",
        "GIS output QA done for geometry validity and CRS correctness.",
        "Monitoring hooks in place for failure rates and inference latency.",
    ]
    pdf.bullet_list(checklist)

    pdf.section("10. Final Conclusion")
    pdf.paragraph(
        "The project currently provides a complete PS-1 technical foundation: architecture, training loop, inference, and export. "
        "Operational functionality is verified, but target-level accuracy is not yet reached by the assessed checkpoint. "
        "Following the staged optimization roadmap and expanding high-quality labeled data should materially improve results and "
        "move the system toward the 95% objective."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVAMITVA detailed PDF documentation and final report.")
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Checkpoint used for metric extraction.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write generated PDFs.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    checkpoint = (repo_root / args.checkpoint).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    context = gather_project_context(repo_root=repo_root, checkpoint_path=checkpoint)
    (output_dir / "report_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")

    doc_pdf = output_dir / "SVAMITVA_Model_Documentation.pdf"
    final_pdf = output_dir / "SVAMITVA_Final_Report.pdf"

    build_model_documentation_pdf(doc_pdf, context)
    build_final_report_pdf(final_pdf, context)

    print(f"Generated: {doc_pdf}")
    print(f"Generated: {final_pdf}")
    print(f"Context:   {output_dir / 'report_context.json'}")


if __name__ == "__main__":
    main()
