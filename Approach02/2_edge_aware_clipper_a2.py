"""
SAM 2 Edge-Aware Clip Polygon Generator
=========================================
Stage 3: Generate intelligent clip shapefiles using SAM 2 segmentation
to follow natural feature boundaries around detected holes.

Instead of classical CV (Canny, Hough), this uses SAM 2 to:
  1. Segment the region around each hole into coherent objects
  2. Classify each segment (vegetation, road, bare earth, etc.)
  3. Select segments that border the hole
  4. Build the clip polygon from segment boundaries

Dependencies:
    pip install sam-2 torch torchvision
    pip install opencv-python-headless shapely geopandas rasterio numpy

Usage:
    from ortho_hole_detector import OrthoHoleDetector
    from edge_aware_clipper import SAM2Clipper

    detector = OrthoHoleDetector("orthophoto.tif")
    holes = detector.run()

    clipper = SAM2Clipper("orthophoto.tif", detector.transform, detector.crs)
    clips = clipper.generate_clips(holes)
    clipper.export_clips("clip_polygons.shp")
"""

import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of clip polygon generation for one hole."""
    hole_id: int
    clip_polygon: Polygon
    strategy: str                # "sam2_boundary", "sam2_prompted", "buffered_fallback"
    confidence: float
    n_masks_used: int            # How many SAM 2 masks contributed to the polygon
    context: str                 # Terrain context from detection stage
    mask_labels: dict            # {label: count} of masks in the clip region


class SAM2Clipper:
    """
    Generate clip polygons using SAM 2 segmentation.

    Two SAM 2 modes depending on hole context:

    1. AUTOMATIC MASK mode (default):
       Run SAM 2 auto-mask on the hole's ROI. Identify which masks touch
       or surround the hole. Union those masks → clip polygon that naturally
       follows object boundaries (canopy edges, road edges, structures).

    2. PROMPTED mode (for complex or large holes):
       Place prompt points along the hole boundary. SAM 2 segments outward
       from each point, capturing the feature that the hole cuts into.
       Union the resulting masks → clip polygon.

    Both modes produce polygons that follow real feature boundaries instead
    of arbitrary buffers — the key improvement over classical CV.
    """

    def __init__(self, ortho_path: str, transform, crs,
                 sam2_checkpoint: str = "sam2.1_hiera_large.pt",
                 sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "auto",
                 buffer_m: float = 5.0,
                 roi_padding_factor: float = 3.0):
        """
        Args:
            ortho_path: Path to orthophoto GeoTIFF
            transform: Affine transform from rasterio
            crs: CRS from rasterio
            sam2_checkpoint: Path to SAM 2 model checkpoint
            sam2_config: Path to SAM 2 config YAML
            device: "auto", "cuda", or "cpu"
            buffer_m: Minimum buffer around holes (map units)
            roi_padding_factor: How much to pad the ROI beyond the hole bbox
        """
        self.ortho_path = Path(ortho_path)
        self.transform = transform
        self.crs = crs
        self.buffer_m = buffer_m
        self.roi_padding = roi_padding_factor
        self.px_size = abs(transform.a)

        self.clip_results: list[ClipResult] = []

        # Initialize SAM 2
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._build_sam2(sam2_checkpoint, sam2_config)

    def _build_sam2(self, checkpoint: str, config: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generation import SAM2AutomaticMaskGenerator

        sam2_model = build_sam2(config, checkpoint, device=self.device)

        # Auto-mask generator for region-wide segmentation
        self.auto_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=48,           # Dense sampling for fine detail
            pred_iou_thresh=0.72,
            stability_score_thresh=0.82,
            crop_n_layers=1,              # Multi-scale for varied feature sizes
            min_mask_region_area=100,
        )

        # Prompted predictor for targeted segmentation around hole edges
        self.predictor = SAM2ImagePredictor(sam2_model)

        log.info(f"SAM 2 clipper loaded on {self.device}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_clips(self, holes: list, interior_only: bool = True) -> list[ClipResult]:
        """Generate SAM 2-based clip polygons for all detected holes."""
        targets = [h for h in holes if h.is_interior] if interior_only else holes
        log.info(f"Generating SAM 2 clip polygons for {len(targets)} holes")

        self.clip_results = []
        for hole in targets:
            result = self._process_hole(hole)
            self.clip_results.append(result)
            log.info(
                f"Hole {hole.id}: strategy={result.strategy}, "
                f"confidence={result.confidence:.2f}, "
                f"masks_used={result.n_masks_used}, "
                f"labels={result.mask_labels}"
            )
        return self.clip_results

    def _process_hole(self, hole) -> ClipResult:
        """Process a single hole through SAM 2 segmentation."""
        # Extract ROI
        rgb, roi_tf, roi_bounds = self._extract_roi(hole)
        if rgb is None:
            return self._fallback_buffer(hole)

        # Convert hole geometry to pixel coordinates within the ROI
        hole_mask_px = self._hole_to_pixel_mask(hole, rgb.shape, roi_tf)

        # Strategy selection based on hole size relative to ROI
        hole_px_area = hole_mask_px.sum()
        roi_px_area = rgb.shape[0] * rgb.shape[1]
        hole_ratio = hole_px_area / roi_px_area

        if hole_ratio < 0.3:
            # Small-to-medium hole: auto-mask the whole ROI, pick bordering masks
            result = self._auto_mask_strategy(hole, rgb, hole_mask_px, roi_tf)
        else:
            # Large hole: use prompted segmentation along the hole boundary
            result = self._prompted_strategy(hole, rgb, hole_mask_px, roi_tf)

        return result if result else self._fallback_buffer(hole)

    # ------------------------------------------------------------------
    # Strategy 1: Automatic mask generation
    # ------------------------------------------------------------------

    def _auto_mask_strategy(self, hole, rgb, hole_mask_px, roi_tf) -> Optional[ClipResult]:
        """
        Run SAM 2 auto-mask on the ROI. Find masks that border the hole.
        Union those masks to form a clip polygon that follows natural edges.
        """
        try:
            masks = self.auto_generator.generate(rgb)
        except Exception as e:
            log.warning(f"SAM 2 auto-mask failed for hole {hole.id}: {e}")
            return None

        if not masks:
            return None

        # Dilate the hole mask slightly to find neighboring segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hole_border = cv2.dilate(hole_mask_px.astype(np.uint8), kernel) - hole_mask_px
        hole_border = hole_border.astype(bool)

        # Find masks that overlap with the hole border region
        bordering_masks = []
        mask_labels = {}
        for m in masks:
            seg = m["segmentation"]
            overlap = np.logical_and(seg, hole_border).sum()
            if overlap > 20:  # Minimum overlap threshold
                label = self._classify_mask(rgb, seg)
                m["label"] = label
                mask_labels[label] = mask_labels.get(label, 0) + 1
                bordering_masks.append(m)

        if not bordering_masks:
            return None

        # Build the clip polygon from bordering masks
        clip_poly = self._masks_to_clip_polygon(
            bordering_masks, hole, hole_mask_px, roi_tf
        )

        confidence = min(1.0, len(bordering_masks) / 8) * 0.85
        return ClipResult(
            hole_id=hole.id,
            clip_polygon=clip_poly,
            strategy="sam2_boundary",
            confidence=confidence,
            n_masks_used=len(bordering_masks),
            context=hole.surrounding_context,
            mask_labels=mask_labels,
        )

    # ------------------------------------------------------------------
    # Strategy 2: Prompted segmentation along hole boundary
    # ------------------------------------------------------------------

    def _prompted_strategy(self, hole, rgb, hole_mask_px, roi_tf) -> Optional[ClipResult]:
        """
        Place prompt points along the hole boundary edge. For each point,
        SAM 2 segments the feature it sits on. Union all results.

        Better for large holes where auto-mask might not capture the full
        bordering context.
        """
        try:
            self.predictor.set_image(rgb)
        except Exception as e:
            log.warning(f"SAM 2 predictor failed for hole {hole.id}: {e}")
            return None

        # Sample points along the hole boundary (in pixel coords)
        boundary_points = self._sample_boundary_points(hole_mask_px, n_points=16)
        if len(boundary_points) < 4:
            return None

        # For each boundary point, segment outward (away from hole)
        all_masks = []
        mask_labels = {}

        for pt in boundary_points:
            # Place a positive point just outside the hole, negative inside
            outside_pt = self._nudge_point_outward(pt, hole_mask_px, distance=5)
            inside_pt = self._nudge_point_inward(pt, hole_mask_px, distance=5)

            point_coords = np.array([[outside_pt[1], outside_pt[0]],   # x, y format
                                     [inside_pt[1], inside_pt[0]]])
            point_labels = np.array([1, 0])  # 1=foreground, 0=background

            try:
                pred_masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                # Pick highest-scoring mask
                best_idx = np.argmax(scores)
                seg = pred_masks[best_idx]

                label = self._classify_mask(rgb, seg)
                mask_labels[label] = mask_labels.get(label, 0) + 1
                all_masks.append({"segmentation": seg, "label": label})

            except Exception:
                continue

        if not all_masks:
            return None

        clip_poly = self._masks_to_clip_polygon(
            all_masks, hole, hole_mask_px, roi_tf
        )

        confidence = min(1.0, len(all_masks) / 12) * 0.80
        return ClipResult(
            hole_id=hole.id,
            clip_polygon=clip_poly,
            strategy="sam2_prompted",
            confidence=confidence,
            n_masks_used=len(all_masks),
            context=hole.surrounding_context,
            mask_labels=mask_labels,
        )

    # ------------------------------------------------------------------
    # Mask → polygon conversion
    # ------------------------------------------------------------------

    def _masks_to_clip_polygon(self, masks, hole, hole_mask_px, roi_tf) -> Polygon:
        """
        Convert a set of SAM 2 masks into a single georeferenced clip polygon.

        Steps:
        1. Union all mask pixels into a single binary mask
        2. Include the hole itself (clip must cover the hole)
        3. Find the largest contour
        4. Simplify and convert to map coordinates
        """
        h, w = hole_mask_px.shape
        combined = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            seg = m["segmentation"]
            if seg.shape == (h, w):
                combined = np.logical_or(combined, seg).astype(np.uint8)

        # Ensure the hole area is included
        combined = np.logical_or(combined, hole_mask_px).astype(np.uint8)

        # Morphological closing to fill small gaps between masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined * 255, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return hole.geometry.buffer(self.buffer_m * 2)

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)

        # Simplify the contour (reduce vertices while keeping shape)
        epsilon = self.px_size * 2
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)

        # Convert pixel contour to map coordinates
        map_coords = []
        for pt in approx.reshape(-1, 2):
            mx = roi_tf.c + pt[0] * roi_tf.a
            my = roi_tf.f + pt[1] * roi_tf.e
            map_coords.append((mx, my))

        if len(map_coords) < 3:
            return hole.geometry.buffer(self.buffer_m * 2)

        poly = Polygon(map_coords)
        poly = self._validate_clip(poly, hole)
        return poly

    # ------------------------------------------------------------------
    # Helper: ROI extraction
    # ------------------------------------------------------------------

    def _extract_roi(self, hole, padding_factor=None):
        """Extract a padded RGB region around the hole."""
        pad_f = padding_factor or self.roi_padding
        minx, miny, maxx, maxy = hole.geometry.bounds
        pad = max(maxx - minx, maxy - miny) * pad_f
        roi_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        try:
            with rasterio.open(self.ortho_path) as src:
                window = from_bounds(*roi_bounds, transform=src.transform)
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height)
                )
                if window.width < 30 or window.height < 30:
                    return None, None, None
                rgb = src.read([1, 2, 3], window=window)
                roi_tf = src.window_transform(window)

            return np.moveaxis(rgb, 0, -1), roi_tf, roi_bounds
        except Exception as e:
            log.warning(f"ROI extraction failed for hole {hole.id}: {e}")
            return None, None, None

    def _hole_to_pixel_mask(self, hole, img_shape, roi_tf) -> np.ndarray:
        """Rasterize the hole polygon into a pixel mask within the ROI."""
        from rasterio.features import rasterize

        h, w = img_shape[:2]
        mask = rasterize(
            [(hole.geometry, 1)],
            out_shape=(h, w),
            transform=roi_tf,
            fill=0,
            dtype=np.uint8,
        )
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Helper: boundary point sampling
    # ------------------------------------------------------------------

    def _sample_boundary_points(self, hole_mask, n_points=16) -> list:
        """Sample evenly spaced points along the hole boundary."""
        contours, _ = cv2.findContours(
            hole_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return []

        boundary = max(contours, key=len).reshape(-1, 2)
        if len(boundary) < n_points:
            return [(pt[1], pt[0]) for pt in boundary]  # (row, col) format

        indices = np.linspace(0, len(boundary) - 1, n_points, dtype=int)
        return [(boundary[i][1], boundary[i][0]) for i in indices]

    def _nudge_point_outward(self, pt, hole_mask, distance=5):
        """Move a boundary point outward (away from hole center)."""
        r, c = pt
        h, w = hole_mask.shape

        # Compute local gradient of the mask (points away from hole)
        patch_r = max(0, r - 10), min(h, r + 10)
        patch_c = max(0, c - 10), min(w, c + 10)
        patch = hole_mask[patch_r[0]:patch_r[1], patch_c[0]:patch_c[1]].astype(float)

        if patch.size == 0:
            return (r - distance, c)

        gy, gx = np.gradient(patch)
        cy, cx = patch.shape[0] // 2, patch.shape[1] // 2

        # Direction away from hole interior
        dr = -gy[cy, cx] if abs(gy[cy, cx]) > 0.01 else -1
        dc = -gx[cy, cx] if abs(gx[cy, cx]) > 0.01 else 0
        norm = max(np.sqrt(dr ** 2 + dc ** 2), 0.01)

        new_r = int(np.clip(r + dr / norm * distance, 0, h - 1))
        new_c = int(np.clip(c + dc / norm * distance, 0, w - 1))
        return (new_r, new_c)

    def _nudge_point_inward(self, pt, hole_mask, distance=5):
        """Move a boundary point inward (into the hole)."""
        r, c = pt
        h, w = hole_mask.shape
        out_r, out_c = self._nudge_point_outward(pt, hole_mask, distance)
        # Inward = opposite direction of outward
        dr = r - out_r
        dc = c - out_c
        norm = max(np.sqrt(dr ** 2 + dc ** 2), 0.01)
        new_r = int(np.clip(r + dr / norm * distance, 0, h - 1))
        new_c = int(np.clip(c + dc / norm * distance, 0, w - 1))
        return (new_r, new_c)

    # ------------------------------------------------------------------
    # Helper: mask classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_mask(rgb: np.ndarray, mask: np.ndarray) -> str:
        """Classify a SAM 2 mask by color statistics."""
        if mask.sum() < 50:
            return "unknown"

        r = rgb[:, :, 0][mask].astype(float)
        g = rgb[:, :, 1][mask].astype(float)
        b = rgb[:, :, 2][mask].astype(float)

        mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
        total = mean_r + mean_g + mean_b + 1e-6
        green_ratio = mean_g / total
        brightness = total / 3
        egi = 2 * mean_g - mean_r - mean_b
        std_all = np.std(r) + np.std(g) + np.std(b)

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_clip(self, polygon: Polygon, hole) -> Polygon:
        """Ensure the clip polygon is valid and fully contains the hole."""
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda g: g.area)

        # Must contain the hole
        if not polygon.contains(hole.geometry):
            polygon = unary_union([polygon, hole.geometry.buffer(self.buffer_m)])
            if isinstance(polygon, MultiPolygon):
                polygon = max(polygon.geoms, key=lambda g: g.area)

        # Minimum area sanity check
        if polygon.area < hole.geometry.area * 1.1:
            polygon = hole.geometry.buffer(self.buffer_m * 3)

        return polygon

    def _fallback_buffer(self, hole) -> ClipResult:
        """Simple buffer fallback when SAM 2 fails."""
        return ClipResult(
            hole_id=hole.id,
            clip_polygon=hole.geometry.buffer(self.buffer_m * 3),
            strategy="buffered_fallback",
            confidence=0.3,
            n_masks_used=0,
            context=hole.surrounding_context,
            mask_labels={},
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_clips(self, output_path: str):
        if not self.clip_results:
            log.warning("No clip results — run generate_clips() first")
            return

        gdf = gpd.GeoDataFrame(
            [{
                "hole_id": r.hole_id,
                "strategy": r.strategy,
                "confidence": r.confidence,
                "n_masks": r.n_masks_used,
                "context": r.context,
                "geometry": r.clip_polygon,
            } for r in self.clip_results],
            crs=self.crs,
        )
        out = Path(output_path)
        driver = "GeoJSON" if out.suffix == ".geojson" else "ESRI Shapefile"
        gdf.to_file(out, driver=driver)
        log.info(f"Exported {len(self.clip_results)} clip polygons to {out}")

    def export_individual_clips(self, output_dir: str):
        """Export each clip as a separate shapefile for WBT processing."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in self.clip_results:
            gdf = gpd.GeoDataFrame(
                [{"hole_id": r.hole_id, "geometry": r.clip_polygon}],
                crs=self.crs,
            )
            gdf.to_file(out_dir / f"clip_hole_{r.hole_id}.shp", driver="ESRI Shapefile")

        log.info(f"Exported {len(self.clip_results)} shapefiles to {out_dir}")