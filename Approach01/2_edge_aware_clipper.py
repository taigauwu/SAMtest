"""
Edge-Aware Clip Polygon Generator
===================================
Stage 3: Generate intelligent clip shapefiles that follow natural features
(road edges, vegetation boundaries) around detected holes.

This replaces the manual QGIS polygon-drawing step from the WBT workaround.

Dependencies:
    pip install opencv-python-headless scikit-image shapely geopandas rasterio numpy

Usage:
    from ortho_hole_detector import OrthoHoleDetector, HoleInfo
    from edge_aware_clipper import EdgeAwareClipper

    detector = OrthoHoleDetector("orthophoto.tif")
    holes = detector.run()

    clipper = EdgeAwareClipper("orthophoto.tif", detector.transform, detector.crs)
    clip_polygons = clipper.generate_clips(holes)
    clipper.export_clips("clip_polygons.shp")
"""

import numpy as np
import cv2
from skimage import morphology, segmentation, filters
from skimage.measure import find_contours
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, snap
from shapely.validation import make_valid
import geopandas as gpd
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of clip polygon generation for a single hole."""
    hole_id: int
    clip_polygon: Polygon        # The final clip polygon for WBT
    strategy: str                # "vegetation", "road_snap", "buffered"
    confidence: float            # 0-1 confidence in the polygon quality
    edge_points_used: int        # How many edge feature points were used


class EdgeAwareClipper:
    """
    Generates clip polygons that follow natural features around holes.

    Strategy per context type:
    - "vegetation": Expand clip boundary outward from hole, following canopy edges
                    detected via green-channel gradients and texture segmentation.
    - "road":       Detect road edges using Canny + Hough lines, snap polygon
                    vertices to the nearest detected road edge.
    - "mixed":      Combine both strategies, using road edges where detected
                    and vegetation edges elsewhere.
    """

    def __init__(self, ortho_path: str, transform, crs,
                 buffer_m: float = 5.0, edge_search_radius_m: float = 20.0):
        """
        Args:
            ortho_path: Path to the orthophoto GeoTIFF
            transform: Affine transform from rasterio
            crs: CRS object from rasterio
            buffer_m: Base buffer distance around holes (map units)
            edge_search_radius_m: How far from the hole boundary to search for edges
        """
        self.ortho_path = Path(ortho_path)
        self.transform = transform
        self.crs = crs
        self.buffer_m = buffer_m
        self.edge_search_radius = edge_search_radius_m
        self.px_size = abs(transform.a)

        self.clip_results: list[ClipResult] = []

    def generate_clips(self, holes: list, interior_only: bool = True) -> list[ClipResult]:
        """
        Generate edge-aware clip polygons for all detected holes.

        Args:
            holes: List of HoleInfo objects from OrthoHoleDetector
            interior_only: Only process interior holes (recommended)

        Returns:
            List of ClipResult with the final clip polygons
        """
        target_holes = [h for h in holes if h.is_interior] if interior_only else holes
        log.info(f"Generating clip polygons for {len(target_holes)} holes")

        self.clip_results = []
        for hole in target_holes:
            result = self._generate_single_clip(hole)
            self.clip_results.append(result)
            log.info(
                f"Hole {hole.id}: strategy={result.strategy}, "
                f"confidence={result.confidence:.2f}, edge_pts={result.edge_points_used}"
            )

        return self.clip_results

    def _generate_single_clip(self, hole) -> ClipResult:
        """Generate a clip polygon for a single hole."""
        context = hole.surrounding_context

        # Step 1: Extract the region of interest (hole + surrounding buffer)
        roi_rgb, roi_transform, roi_bounds = self._extract_roi(hole)

        if roi_rgb is None:
            # Fallback: simple buffer
            return ClipResult(
                hole_id=hole.id,
                clip_polygon=hole.geometry.buffer(self.buffer_m),
                strategy="buffered",
                confidence=0.3,
                edge_points_used=0,
            )

        # Step 2: Detect edges in the ROI
        edges, edge_lines = self._detect_edges(roi_rgb, context)

        # Step 3: Generate the clip polygon based on context
        if context == "vegetation":
            clip_poly, confidence, n_pts = self._clip_vegetation(
                hole, roi_rgb, edges, roi_transform, roi_bounds
            )
            strategy = "vegetation"
        elif context == "road":
            clip_poly, confidence, n_pts = self._clip_road(
                hole, roi_rgb, edges, edge_lines, roi_transform, roi_bounds
            )
            strategy = "road_snap"
        else:
            clip_poly, confidence, n_pts = self._clip_mixed(
                hole, roi_rgb, edges, edge_lines, roi_transform, roi_bounds
            )
            strategy = "mixed"

        # Validate and clean the polygon
        clip_poly = self._validate_polygon(clip_poly, hole)

        return ClipResult(
            hole_id=hole.id,
            clip_polygon=clip_poly,
            strategy=strategy,
            confidence=confidence,
            edge_points_used=n_pts,
        )

    def _extract_roi(self, hole, padding_factor: float = 3.0):
        """Extract a padded region of interest around the hole."""
        minx, miny, maxx, maxy = hole.geometry.bounds
        pad = max(maxx - minx, maxy - miny) * padding_factor + self.edge_search_radius

        roi_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        try:
            with rasterio.open(self.ortho_path) as src:
                window = from_bounds(*roi_bounds, transform=src.transform)
                # Clamp window to image bounds
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                if window.width < 10 or window.height < 10:
                    return None, None, None

                rgb = src.read([1, 2, 3], window=window)
                roi_transform = src.window_transform(window)

            # Transpose to (H, W, 3) for OpenCV
            rgb = np.moveaxis(rgb, 0, -1)
            return rgb, roi_transform, roi_bounds

        except Exception as e:
            log.warning(f"Failed to extract ROI for hole {hole.id}: {e}")
            return None, None, None

    def _detect_edges(self, rgb: np.ndarray, context: str):
        """
        Detect edges in the ROI using context-appropriate methods.

        Returns:
            edges: Binary edge map (uint8)
            edge_lines: List of detected line segments (for road context)
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if context == "vegetation":
            # For vegetation: use green channel gradient (canopy edges are sharp in green)
            green = rgb[:, :, 1].astype(np.float32)
            # Bilateral filter to preserve edges while smoothing texture
            green_filtered = cv2.bilateralFilter(green, 9, 75, 75)
            edges = cv2.Canny(green_filtered.astype(np.uint8), 30, 80)
            # Dilate slightly to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            edge_lines = []

        elif context == "road":
            # For roads: stronger edge detection + Hough line detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Hough line detection for straight road edges
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=50,
                minLineLength=30, maxLineGap=10
            )
            edge_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    edge_lines.append(((x1, y1), (x2, y2)))

        else:
            # Mixed: combine both approaches
            green = rgb[:, :, 1].astype(np.float32)
            green_edges = cv2.Canny(
                cv2.bilateralFilter(green, 9, 75, 75).astype(np.uint8), 30, 80
            )
            gray_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
            edges = cv2.bitwise_or(green_edges, gray_edges)

            lines = cv2.HoughLinesP(
                gray_edges, rho=1, theta=np.pi / 180, threshold=50,
                minLineLength=30, maxLineGap=10
            )
            edge_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    edge_lines.append(((x1, y1), (x2, y2)))

        return edges, edge_lines

    def _clip_vegetation(self, hole, roi_rgb, edges, roi_transform, roi_bounds):
        """
        Generate clip polygon following vegetation canopy edges.

        Strategy:
        1. Buffer the hole boundary outward
        2. Find strong edge contours in the buffer zone
        3. Snap buffered polygon vertices to nearest canopy edges
        4. Smooth the result
        """
        # Start with a buffered hole polygon
        buffered = hole.geometry.buffer(self.buffer_m * 2)

        # Find contours of strong edges
        contours = find_contours(edges, level=128)
        edge_points_map = []

        for contour in contours:
            for row, col in contour:
                # Convert pixel to map coordinates
                mx = roi_transform.c + col * roi_transform.a
                my = roi_transform.f + row * roi_transform.e
                edge_points_map.append((mx, my))

        if len(edge_points_map) < 10:
            return buffered, 0.4, 0

        # Snap buffered polygon vertices to nearby edge points
        snapped = self._snap_to_edges(buffered, edge_points_map, max_dist=self.edge_search_radius)

        # Smooth the polygon to avoid jagged edges
        snapped = snapped.simplify(self.px_size * 3, preserve_topology=True)

        # Ensure the clip polygon fully contains the hole
        if not snapped.contains(hole.geometry):
            snapped = unary_union([snapped, hole.geometry.buffer(self.buffer_m)])

        confidence = min(1.0, len(edge_points_map) / 100) * 0.8
        return snapped, confidence, len(edge_points_map)

    def _clip_road(self, hole, roi_rgb, edges, edge_lines, roi_transform, roi_bounds):
        """
        Generate clip polygon snapping to road edges.

        Strategy:
        1. Convert detected Hough lines to map coordinates
        2. Buffer the hole
        3. Where the buffer intersects a road line, snap to it
        4. Use the road edge as the clip boundary on that side
        """
        buffered = hole.geometry.buffer(self.buffer_m * 2)

        if not edge_lines:
            return buffered, 0.3, 0

        # Convert edge lines to map coordinates
        map_lines = []
        for (x1, y1), (x2, y2) in edge_lines:
            mx1 = roi_transform.c + x1 * roi_transform.a
            my1 = roi_transform.f + y1 * roi_transform.e
            mx2 = roi_transform.c + x2 * roi_transform.a
            my2 = roi_transform.f + y2 * roi_transform.e
            map_lines.append(LineString([(mx1, my1), (mx2, my2)]))

        # Merge nearby lines
        merged_lines = unary_union(map_lines)

        # Find lines that are near the hole boundary
        hole_boundary = hole.geometry.boundary
        nearby_lines = []
        for line in (merged_lines.geoms if hasattr(merged_lines, 'geoms') else [merged_lines]):
            dist = hole_boundary.distance(line)
            if dist < self.edge_search_radius:
                nearby_lines.append(line)

        if not nearby_lines:
            return buffered, 0.35, 0

        # Snap the buffered polygon to road lines
        all_edge_pts = []
        for line in nearby_lines:
            coords = list(line.coords)
            all_edge_pts.extend(coords)

        snapped = self._snap_to_edges(buffered, all_edge_pts, max_dist=self.edge_search_radius)
        snapped = snapped.simplify(self.px_size * 2, preserve_topology=True)

        if not snapped.contains(hole.geometry):
            snapped = unary_union([snapped, hole.geometry.buffer(self.buffer_m)])

        confidence = min(1.0, len(nearby_lines) / 5) * 0.85
        return snapped, confidence, len(all_edge_pts)

    def _clip_mixed(self, hole, roi_rgb, edges, edge_lines, roi_transform, roi_bounds):
        """Combine vegetation and road strategies."""
        veg_poly, veg_conf, veg_pts = self._clip_vegetation(
            hole, roi_rgb, edges, roi_transform, roi_bounds
        )
        road_poly, road_conf, road_pts = self._clip_road(
            hole, roi_rgb, edges, edge_lines, roi_transform, roi_bounds
        )

        # Use whichever has higher confidence, or union them
        if road_conf > veg_conf and road_pts > 5:
            return road_poly, road_conf, road_pts
        elif veg_pts > road_pts:
            return veg_poly, veg_conf, veg_pts
        else:
            merged = unary_union([veg_poly, road_poly]).convex_hull
            return merged, max(veg_conf, road_conf) * 0.9, veg_pts + road_pts

    def _snap_to_edges(self, polygon: Polygon, edge_points: list,
                       max_dist: float) -> Polygon:
        """
        Snap polygon vertices to nearby detected edge points.

        For each vertex of the buffered polygon, if there's an edge point
        within max_dist, move the vertex to that edge point.
        """
        if not edge_points:
            return polygon

        edge_arr = np.array(edge_points)
        coords = list(polygon.exterior.coords)
        snapped_coords = []

        for x, y in coords:
            # Find nearest edge point
            dists = np.sqrt((edge_arr[:, 0] - x) ** 2 + (edge_arr[:, 1] - y) ** 2)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            if min_dist < max_dist:
                snapped_coords.append(edge_points[min_idx])
            else:
                snapped_coords.append((x, y))

        try:
            result = Polygon(snapped_coords)
            if result.is_valid:
                return result
            return make_valid(result)
        except Exception:
            return polygon

    def _validate_polygon(self, polygon: Polygon, hole) -> Polygon:
        """Ensure the clip polygon is valid and usable."""
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda g: g.area)

        # Must contain the hole
        if not polygon.contains(hole.geometry):
            polygon = unary_union([polygon, hole.geometry.buffer(self.buffer_m)])

        # Minimum area check
        if polygon.area < hole.geometry.area * 1.1:
            polygon = hole.geometry.buffer(self.buffer_m * 3)

        return polygon

    # === Export ===

    def export_clips(self, output_path: str):
        """Export all clip polygons as a shapefile or GeoJSON."""
        if not self.clip_results:
            log.warning("No clip results to export! Run generate_clips() first.")
            return

        gdf = gpd.GeoDataFrame(
            [{
                "hole_id": r.hole_id,
                "strategy": r.strategy,
                "confidence": r.confidence,
                "edge_pts": r.edge_points_used,
                "geometry": r.clip_polygon,
            } for r in self.clip_results],
            crs=self.crs,
        )

        out = Path(output_path)
        if out.suffix == ".geojson":
            gdf.to_file(out, driver="GeoJSON")
        elif out.suffix == ".shp":
            gdf.to_file(out, driver="ESRI Shapefile")
        else:
            gdf.to_file(out)

        log.info(f"Exported {len(self.clip_results)} clip polygons to {out}")

    def export_individual_clips(self, output_dir: str):
        """Export each clip polygon as a separate shapefile (for WBT processing)."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for result in self.clip_results:
            gdf = gpd.GeoDataFrame(
                [{"hole_id": result.hole_id, "geometry": result.clip_polygon}],
                crs=self.crs,
            )
            path = out_dir / f"clip_hole_{result.hole_id}.shp"
            gdf.to_file(path, driver="ESRI Shapefile")

        log.info(f"Exported {len(self.clip_results)} individual shapefiles to {out_dir}")