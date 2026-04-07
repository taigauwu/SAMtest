"""
Orthophoto Hole Detector & Extent Inferencer
=============================================
Stage 1: Detect NoData holes in orthophoto GeoTIFFs
Stage 2: Infer expected extent, classify interior vs edge gaps
Stage 2b: SAM 2-powered context classification around each hole

Dependencies:
    pip install rasterio numpy shapely scipy geopandas fiona
    pip install sam-2 torch torchvision

Usage:
    from ortho_hole_detector import OrthoHoleDetector
    detector = OrthoHoleDetector("path/to/orthophoto.tif")
    holes = detector.run()
    detector.export_holes("holes.geojson")
"""

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.windows import from_bounds
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Delaunay
import geopandas as gpd
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM 2 context classifier — uses automatic mask generation to understand
# what kind of terrain surrounds each hole (vegetation, road, mixed, etc.)
# ---------------------------------------------------------------------------

class SAM2ContextClassifier:
    """
    Classify the terrain context around a hole using SAM 2
    automatic mask generation + color statistics per mask.
    """

    def __init__(self, checkpoint: str = "sam2.1_hiera_large.pt",
                 config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "auto"):
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._build_model(checkpoint, config)

    def _build_model(self, checkpoint, config):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generation import SAM2AutomaticMaskGenerator

        sam2 = build_sam2(config, checkpoint, device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=200,
        )
        log.info(f"SAM 2 context classifier loaded on {self.device}")

    def classify_region(self, rgb: np.ndarray) -> dict:
        """
        Run SAM 2 on an RGB crop and classify each mask.

        Returns dict with:
          - masks: list of SAM 2 mask dicts enriched with 'label'
          - dominant_context: overall classification for the region
          - label_areas: {label: total_pixel_area}
        """
        masks = self.mask_generator.generate(rgb)

        label_areas = {"vegetation": 0, "road": 0, "bare_earth": 0,
                       "structure": 0, "water": 0, "unknown": 0}

        for m in masks:
            seg = m["segmentation"]  # bool array (H, W)
            label = self._classify_single_mask(rgb, seg)
            m["label"] = label
            label_areas[label] += int(seg.sum())

        # Dominant context = label with largest area (ignoring unknown)
        known = {k: v for k, v in label_areas.items() if k != "unknown"}
        dominant = max(known, key=known.get) if any(known.values()) else "unknown"

        return {
            "masks": masks,
            "dominant_context": dominant,
            "label_areas": label_areas,
        }

    @staticmethod
    def _classify_single_mask(rgb: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify a single SAM 2 mask using color statistics.

        Much more reliable than classifying raw pixels because SAM gives us
        coherent object-level regions instead of noisy pixel neighborhoods.
        """
        if mask.sum() < 50:
            return "unknown"

        r = rgb[:, :, 0][mask].astype(float)
        g = rgb[:, :, 1][mask].astype(float)
        b = rgb[:, :, 2][mask].astype(float)

        mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
        total = mean_r + mean_g + mean_b + 1e-6
        green_ratio = mean_g / total
        brightness = total / 3
        std_all = np.std(r) + np.std(g) + np.std(b)

        # Excess green index (common in remote sensing)
        egi = 2 * mean_g - mean_r - mean_b

        # Classification rules (tuned for aerial/drone orthophotos)
        if egi > 30 and green_ratio > 0.37:
            return "vegetation"
        if brightness > 160 and std_all < 60 and green_ratio < 0.38:
            return "road"
        if brightness < 80:
            return "water"
        if std_all < 40 and brightness > 100:
            return "bare_earth"
        if std_all > 100:
            return "structure"
        return "unknown"


# ---------------------------------------------------------------------------
# Hole metadata
# ---------------------------------------------------------------------------

@dataclass
class HoleInfo:
    """Metadata for a single detected hole."""
    id: int
    geometry: Polygon
    area_m2: float
    centroid: tuple
    is_interior: bool
    surrounding_context: str = "unknown"
    sam_masks: list = field(default_factory=list)   # SAM 2 masks for this hole's ROI
    label_areas: dict = field(default_factory=dict)  # {label: pixel_area}
    bbox: tuple = None

    def __post_init__(self):
        b = self.geometry.bounds
        self.bbox = (b[0], b[1], b[2], b[3])


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class OrthoHoleDetector:
    """
    Detects holes in orthophotos and classifies their surroundings via SAM 2.

    Pipeline:
    1. Build binary valid/nodata mask from GeoTIFF
    2. Vectorize nodata regions into polygons
    3. Compute concave hull of valid pixels → expected extent
    4. Classify each nodata polygon as interior hole vs edge gap
    5. Run SAM 2 on the region around each interior hole to classify context
    """

    def __init__(self, ortho_path: str,
                 nodata_val: Optional[float] = None,
                 min_hole_area_m2: float = 1.0,
                 alpha: float = 50.0,
                 sam2_checkpoint: str = "sam2.1_hiera_large.pt",
                 sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "auto"):
        self.ortho_path = Path(ortho_path)
        self.nodata_val = nodata_val
        self.min_hole_area_m2 = min_hole_area_m2
        self.alpha = alpha

        self.holes: list[HoleInfo] = []
        self.valid_extent: Optional[Polygon] = None
        self.crs = None
        self.transform = None
        self.shape = None

        self._load_metadata()

        # Initialize SAM 2 context classifier
        log.info("Loading SAM 2 for context classification...")
        self.sam_classifier = SAM2ContextClassifier(
            checkpoint=sam2_checkpoint, config=sam2_config, device=device
        )

    def _load_metadata(self):
        with rasterio.open(self.ortho_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.shape = (src.height, src.width)
            if self.nodata_val is None:
                self.nodata_val = src.nodata
            log.info(f"Loaded: {src.width}x{src.height}, CRS={self.crs}, nodata={self.nodata_val}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> list[HoleInfo]:
        log.info("=== Starting hole detection pipeline ===")

        mask = self._build_mask()
        nodata_polys = self._vectorize_nodata(mask)
        log.info(f"Found {len(nodata_polys)} nodata regions")

        self.valid_extent = self._compute_valid_extent(mask)
        log.info(f"Valid extent area: {self.valid_extent.area:.1f} sq units")

        self.holes = self._classify_holes(nodata_polys)
        interior = [h for h in self.holes if h.is_interior]
        log.info(f"Interior holes: {len(interior)} / {len(self.holes)} total")

        # SAM 2 context analysis for each interior hole
        self._sam2_context_analysis()

        return self.holes

    # ------------------------------------------------------------------
    # Stage 1: Mask building
    # ------------------------------------------------------------------

    def _build_mask(self) -> np.ndarray:
        """
        Build binary mask: 1 = valid, 0 = nodata.
        Handles alpha channel, nodata value, or all-zero fallback.
        """
        with rasterio.open(self.ortho_path) as src:
            if src.count >= 4:
                alpha = src.read(4)
                mask = (alpha > 0).astype(np.uint8)
                log.info("Using alpha channel for mask")
            elif self.nodata_val is not None:
                bands = src.read()
                mask = np.any(bands != self.nodata_val, axis=0).astype(np.uint8)
                log.info(f"Using nodata={self.nodata_val} for mask")
            else:
                bands = src.read()
                mask = np.any(bands != 0, axis=0).astype(np.uint8)
                log.info("Using all-zero detection for mask")

        valid_pct = mask.sum() / mask.size * 100
        log.info(f"Valid pixels: {valid_pct:.1f}%")
        return mask

    def _vectorize_nodata(self, mask: np.ndarray) -> list[Polygon]:
        nodata_mask = (mask == 0).astype(np.uint8)
        polys = []
        for geom, val in rio_shapes(nodata_mask, mask=nodata_mask, transform=self.transform):
            if val == 1:
                p = shape(geom)
                if p.is_valid and p.area >= self._area_threshold():
                    polys.append(p)
        return polys

    def _area_threshold(self) -> float:
        px_w = abs(self.transform.a)
        px_h = abs(self.transform.e)
        px_area = px_w * px_h
        return self.min_hole_area_m2 / (px_area if px_area > 0.001 else 1.0) * px_area

    # ------------------------------------------------------------------
    # Stage 2: Extent inference
    # ------------------------------------------------------------------

    def _compute_valid_extent(self, mask: np.ndarray) -> Polygon:
        """Concave hull of valid pixels via filtered Delaunay triangulation."""
        ys, xs = np.where(mask == 1)
        step = max(1, len(ys) // 10000)
        xs_s, ys_s = xs[::step], ys[::step]

        map_xs = self.transform.c + xs_s * self.transform.a
        map_ys = self.transform.f + ys_s * self.transform.e
        points = np.column_stack([map_xs, map_ys])

        log.info(f"Computing concave hull from {len(points)} points (alpha={self.alpha})")

        tri = Delaunay(points)
        triangles = []
        for simplex in tri.simplices:
            pts = points[simplex]
            edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
            if max(edges) < self.alpha:
                triangles.append(Polygon(pts))

        if not triangles:
            log.warning("Concave hull empty — falling back to convex hull")
            from shapely import MultiPoint
            return MultiPoint(points).convex_hull

        extent = unary_union(triangles)
        if isinstance(extent, MultiPolygon):
            extent = max(extent.geoms, key=lambda g: g.area)
        return extent

    def _classify_holes(self, nodata_polys: list[Polygon]) -> list[HoleInfo]:
        holes = []
        for i, poly in enumerate(nodata_polys):
            is_interior = self.valid_extent.buffer(-2).contains(poly.centroid)
            holes.append(HoleInfo(
                id=i, geometry=poly, area_m2=poly.area,
                centroid=(poly.centroid.x, poly.centroid.y),
                is_interior=is_interior,
            ))
        return holes

    # ------------------------------------------------------------------
    # Stage 2b: SAM 2 context analysis
    # ------------------------------------------------------------------

    def _sam2_context_analysis(self):
        """Run SAM 2 on the ROI around each interior hole to classify context."""
        interior = [h for h in self.holes if h.is_interior]
        if not interior:
            return

        log.info(f"Running SAM 2 context analysis on {len(interior)} holes...")

        for hole in interior:
            rgb_crop = self._extract_hole_roi(hole, padding_factor=2.0)
            if rgb_crop is None:
                continue

            result = self.sam_classifier.classify_region(rgb_crop)
            hole.surrounding_context = result["dominant_context"]
            hole.sam_masks = result["masks"]
            hole.label_areas = result["label_areas"]

            log.info(
                f"Hole {hole.id}: context={hole.surrounding_context} "
                f"areas={hole.label_areas}"
            )

    def _extract_hole_roi(self, hole: HoleInfo, padding_factor: float = 2.0):
        """Extract an RGB crop around a hole for SAM 2 processing."""
        minx, miny, maxx, maxy = hole.geometry.bounds
        pad = max(maxx - minx, maxy - miny) * padding_factor
        roi_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        try:
            with rasterio.open(self.ortho_path) as src:
                window = from_bounds(*roi_bounds, transform=src.transform)
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height)
                )
                if window.width < 20 or window.height < 20:
                    return None
                rgb = src.read([1, 2, 3], window=window)
            return np.moveaxis(rgb, 0, -1)  # (H, W, 3) for SAM 2
        except Exception as e:
            log.warning(f"Failed to extract ROI for hole {hole.id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_holes(self, output_path: str, interior_only: bool = True):
        holes = [h for h in self.holes if h.is_interior] if interior_only else self.holes
        if not holes:
            log.warning("No holes to export")
            return

        gdf = gpd.GeoDataFrame(
            [{
                "hole_id": h.id,
                "area_m2": h.area_m2,
                "is_interior": h.is_interior,
                "context": h.surrounding_context,
                "geometry": h.geometry,
            } for h in holes],
            crs=self.crs,
        )
        out = Path(output_path)
        driver = "GeoJSON" if out.suffix == ".geojson" else "ESRI Shapefile"
        gdf.to_file(out, driver=driver)
        log.info(f"Exported {len(holes)} holes to {out}")

    def export_extent(self, output_path: str):
        if self.valid_extent is None:
            return
        gdf = gpd.GeoDataFrame(
            [{"type": "valid_extent", "geometry": self.valid_extent}], crs=self.crs
        )
        gdf.to_file(output_path)

    def get_summary(self) -> dict:
        interior = [h for h in self.holes if h.is_interior]
        return {
            "total_holes": len(self.holes),
            "interior_holes": len(interior),
            "contexts": {h.id: h.surrounding_context for h in interior},
            "total_hole_area_m2": sum(h.area_m2 for h in interior),
            "crs": str(self.crs),
        }


# === CLI ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect holes in orthophotos (SAM 2)")
    parser.add_argument("ortho", help="Path to orthophoto GeoTIFF")
    parser.add_argument("-o", "--output", default="holes.geojson")
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--sam2-checkpoint", default="sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    det = OrthoHoleDetector(
        args.ortho, min_hole_area_m2=args.min_area, alpha=args.alpha,
        sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
        device=args.device,
    )
    det.run()
    print(json.dumps(det.get_summary(), indent=2))
    det.export_holes(args.output)