"""
Orthophoto Hole Detector & Extent Inferencer
=============================================
Stage 1: Detect NoData holes in orthophoto GeoTIFFs
Stage 2: Infer expected extent from valid pixels, classify interior vs edge gaps

Dependencies:
    pip install rasterio numpy shapely scipy geopandas fiona matplotlib

Usage:
    from ortho_hole_detector import OrthoHoleDetector
    detector = OrthoHoleDetector("path/to/orthophoto.tif")
    holes = detector.run()
    detector.export_holes("holes.geojson")
    detector.visualize()
"""

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.transform import rowcol
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
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


@dataclass
class HoleInfo:
    """Metadata for a single detected hole."""
    id: int
    geometry: Polygon
    area_m2: float
    centroid: tuple
    is_interior: bool  # True = hole inside valid data; False = edge gap
    surrounding_context: str  # "vegetation", "road", "mixed", "unknown"
    bbox: tuple = field(default=None)

    def __post_init__(self):
        b = self.geometry.bounds
        self.bbox = (b[0], b[1], b[2], b[3])


class OrthoHoleDetector:
    """
    Detects holes in orthophotos and infers expected extent.
    
    The pipeline:
    1. Load orthophoto, build a binary valid/nodata mask
    2. Vectorize nodata regions into polygons
    3. Compute concave hull of valid pixels → defines expected extent
    4. Classify each nodata polygon as interior hole vs edge gap
    5. Analyze surrounding pixels to tag context (vegetation/road/mixed)
    """

    def __init__(self, ortho_path: str, nodata_val: Optional[float] = None,
                 min_hole_area_m2: float = 1.0, alpha: float = 50.0):
        """
        Args:
            ortho_path: Path to orthophoto GeoTIFF
            nodata_val: Override nodata value (auto-detected if None)
            min_hole_area_m2: Ignore holes smaller than this (sq meters)
            alpha: Concave hull tightness (smaller = tighter fit to data boundary)
        """
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

    def _load_metadata(self):
        """Read CRS, transform, and shape from the orthophoto."""
        with rasterio.open(self.ortho_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.shape = (src.height, src.width)
            if self.nodata_val is None:
                self.nodata_val = src.nodata
            log.info(f"Loaded ortho: {src.width}x{src.height}, CRS={self.crs}, nodata={self.nodata_val}")

    def run(self) -> list[HoleInfo]:
        """Execute the full detection pipeline."""
        log.info("=== Starting hole detection pipeline ===")

        # Stage 1: Build valid/nodata mask
        mask = self._build_mask()

        # Stage 1b: Vectorize nodata regions
        nodata_polys = self._vectorize_nodata(mask)
        log.info(f"Found {len(nodata_polys)} nodata regions total")

        # Stage 2: Infer expected extent via concave hull
        self.valid_extent = self._compute_valid_extent(mask)
        log.info(f"Valid extent area: {self.valid_extent.area:.1f} sq units")

        # Stage 2b: Classify holes (interior vs edge)
        self.holes = self._classify_holes(nodata_polys)
        interior = [h for h in self.holes if h.is_interior]
        log.info(f"Interior holes: {len(interior)} / {len(self.holes)} total")

        # Stage 1c: Analyze surrounding context for each interior hole
        self._analyze_context(mask)

        return self.holes

    def _build_mask(self) -> np.ndarray:
        """
        Build a binary mask: 1 = valid pixel, 0 = nodata/hole.
        
        Handles multiple detection strategies:
        - Alpha channel (band 4) if present
        - Explicit nodata value
        - All-zero pixels (common in DJI Terra output)
        """
        with rasterio.open(self.ortho_path) as src:
            if src.count >= 4:
                # Use alpha channel
                alpha = src.read(4)
                mask = (alpha > 0).astype(np.uint8)
                log.info("Using alpha channel for mask")
            elif self.nodata_val is not None:
                # Use nodata value across all bands
                bands = src.read()
                mask = np.any(bands != self.nodata_val, axis=0).astype(np.uint8)
                log.info(f"Using nodata value ({self.nodata_val}) for mask")
            else:
                # Fallback: pixels where ALL bands are 0
                bands = src.read()
                mask = np.any(bands != 0, axis=0).astype(np.uint8)
                log.info("Fallback: using all-zero detection for mask")

        valid_pct = mask.sum() / mask.size * 100
        log.info(f"Valid pixels: {valid_pct:.1f}%")
        return mask

    def _vectorize_nodata(self, mask: np.ndarray) -> list[Polygon]:
        """Convert nodata regions in the mask to georeferenced polygons."""
        # Invert mask: we want nodata=1 for vectorization
        nodata_mask = (mask == 0).astype(np.uint8)

        polys = []
        for geom, val in rio_shapes(nodata_mask, mask=nodata_mask, transform=self.transform):
            if val == 1:
                p = shape(geom)
                if p.is_valid and p.area >= self._area_threshold():
                    polys.append(p)

        return polys

    def _area_threshold(self) -> float:
        """Convert min_hole_area_m2 to map units based on pixel size."""
        px_w = abs(self.transform.a)
        px_h = abs(self.transform.e)
        px_area = px_w * px_h
        # Rough threshold: min_hole_area_m2 in map units
        # For projected CRS this is already in meters; for geographic, approximate
        return self.min_hole_area_m2 / (px_area if px_area > 0.001 else 1.0) * px_area

    def _compute_valid_extent(self, mask: np.ndarray) -> Polygon:
        """
        Compute a concave hull of valid pixels to define expected extent.
        
        Strategy: Sample valid pixel coordinates, compute Delaunay triangulation,
        remove triangles with edges longer than alpha threshold, union the rest.
        """
        # Downsample for performance — take every Nth valid pixel
        ys, xs = np.where(mask == 1)
        n_pts = len(ys)
        step = max(1, n_pts // 10000)  # Cap at ~10k sample points
        xs_s, ys_s = xs[::step], ys[::step]

        # Convert pixel coords to map coords
        map_xs = self.transform.c + xs_s * self.transform.a
        map_ys = self.transform.f + ys_s * self.transform.e
        points = np.column_stack([map_xs, map_ys])

        log.info(f"Computing concave hull from {len(points)} sample points (alpha={self.alpha})")

        # Delaunay triangulation
        tri = Delaunay(points)

        # Filter triangles by edge length
        triangles = []
        for simplex in tri.simplices:
            pts = points[simplex]
            # Compute edge lengths
            edges = [
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[2] - pts[0]),
            ]
            if max(edges) < self.alpha:
                triangles.append(Polygon(pts))

        if not triangles:
            # Fallback: convex hull
            log.warning("Concave hull empty — falling back to convex hull")
            from shapely import MultiPoint
            return MultiPoint(points).convex_hull

        extent = unary_union(triangles)
        if isinstance(extent, MultiPolygon):
            extent = max(extent.geoms, key=lambda g: g.area)

        return extent

    def _classify_holes(self, nodata_polys: list[Polygon]) -> list[HoleInfo]:
        """Classify each nodata polygon as interior hole or edge gap."""
        holes = []
        for i, poly in enumerate(nodata_polys):
            # A hole is "interior" if it's fully contained within the valid extent
            # (with a small buffer tolerance for edge effects)
            is_interior = self.valid_extent.buffer(-2).contains(poly.centroid)

            holes.append(HoleInfo(
                id=i,
                geometry=poly,
                area_m2=poly.area,  # Approximate if CRS is projected
                centroid=(poly.centroid.x, poly.centroid.y),
                is_interior=is_interior,
                surrounding_context="unknown",
            ))

        return holes

    def _analyze_context(self, mask: np.ndarray):
        """
        Analyze pixels surrounding each interior hole to determine context.
        
        Uses color statistics of the border region:
        - High green ratio + low variance → vegetation
        - High gray/brightness + linear features → road
        - Mixed → mixed
        """
        with rasterio.open(self.ortho_path) as src:
            rgb = src.read([1, 2, 3])  # Shape: (3, H, W)

        for hole in self.holes:
            if not hole.is_interior:
                continue

            # Get bounding box in pixel coordinates
            minx, miny, maxx, maxy = hole.geometry.bounds
            col_min, row_max = ~self.transform * (minx, miny)
            col_max, row_min = ~self.transform * (maxx, maxy)

            # Clamp to image bounds
            row_min = max(0, int(row_min) - 10)
            row_max = min(self.shape[0], int(row_max) + 10)
            col_min = max(0, int(col_min) - 10)
            col_max = min(self.shape[1], int(col_max) + 10)

            # Extract the surrounding region's RGB values (valid pixels only)
            region_mask = mask[row_min:row_max, col_min:col_max]
            region_rgb = rgb[:, row_min:row_max, col_min:col_max]

            valid_px = region_mask == 1
            if valid_px.sum() < 10:
                continue

            r = region_rgb[0][valid_px].astype(float)
            g = region_rgb[1][valid_px].astype(float)
            b = region_rgb[2][valid_px].astype(float)

            # Green ratio: vegetation tends to have high G relative to R and B
            total = r + g + b + 1e-6
            green_ratio = (g / total).mean()
            brightness = total.mean() / 3

            # Variance: vegetation has lower variance than mixed terrain
            variance = np.std(r) + np.std(g) + np.std(b)

            if green_ratio > 0.38 and variance < 120:
                hole.surrounding_context = "vegetation"
            elif brightness > 150 and variance < 80:
                hole.surrounding_context = "road"
            else:
                hole.surrounding_context = "mixed"

            log.info(
                f"Hole {hole.id}: context={hole.surrounding_context} "
                f"(green_ratio={green_ratio:.2f}, brightness={brightness:.0f}, var={variance:.0f})"
            )

    # === Export Methods ===

    def export_holes(self, output_path: str, interior_only: bool = True):
        """Export detected holes as GeoJSON or Shapefile."""
        holes = [h for h in self.holes if h.is_interior] if interior_only else self.holes

        if not holes:
            log.warning("No holes to export!")
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
        if out.suffix == ".geojson":
            gdf.to_file(out, driver="GeoJSON")
        elif out.suffix == ".shp":
            gdf.to_file(out, driver="ESRI Shapefile")
        else:
            gdf.to_file(out)

        log.info(f"Exported {len(holes)} holes to {out}")

    def export_extent(self, output_path: str):
        """Export the inferred valid extent polygon."""
        if self.valid_extent is None:
            log.warning("Run detection first!")
            return

        gdf = gpd.GeoDataFrame(
            [{"type": "valid_extent", "geometry": self.valid_extent}],
            crs=self.crs,
        )
        gdf.to_file(output_path)
        log.info(f"Exported valid extent to {output_path}")

    def get_summary(self) -> dict:
        """Return a summary dict for downstream stages."""
        interior = [h for h in self.holes if h.is_interior]
        return {
            "total_holes": len(self.holes),
            "interior_holes": len(interior),
            "contexts": {h.id: h.surrounding_context for h in interior},
            "total_hole_area_m2": sum(h.area_m2 for h in interior),
            "crs": str(self.crs),
            "ortho_path": str(self.ortho_path),
        }

    def visualize(self, output_path: Optional[str] = None):
        """Quick matplotlib visualization of detected holes."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot valid extent
        if self.valid_extent:
            ext_x, ext_y = self.valid_extent.exterior.xy
            ax.plot(ext_x, ext_y, 'b-', linewidth=1, label="Valid extent", alpha=0.5)

        # Plot holes
        interior_patches, edge_patches = [], []
        for h in self.holes:
            coords = list(h.geometry.exterior.coords)
            patch = MplPoly(coords, closed=True)
            if h.is_interior:
                interior_patches.append(patch)
            else:
                edge_patches.append(patch)

        if interior_patches:
            pc = PatchCollection(interior_patches, facecolor='red', edgecolor='darkred',
                                 alpha=0.6, linewidth=1)
            ax.add_collection(pc)
        if edge_patches:
            pc = PatchCollection(edge_patches, facecolor='gray', edgecolor='darkgray',
                                 alpha=0.3, linewidth=0.5)
            ax.add_collection(pc)

        # Labels for interior holes
        for h in self.holes:
            if h.is_interior:
                ax.annotate(
                    f"#{h.id}\n{h.surrounding_context}\n{h.area_m2:.0f}m²",
                    xy=h.centroid, fontsize=7, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8),
                )

        ax.set_aspect('equal')
        ax.set_title(f"Detected holes ({len([h for h in self.holes if h.is_interior])} interior)")
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved visualization to {output_path}")
        else:
            plt.show()


# === CLI Entry Point ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect holes in orthophotos")
    parser.add_argument("ortho", help="Path to orthophoto GeoTIFF")
    parser.add_argument("-o", "--output", default="holes.geojson", help="Output path")
    parser.add_argument("--min-area", type=float, default=1.0, help="Min hole area (m²)")
    parser.add_argument("--alpha", type=float, default=50.0, help="Concave hull tightness")
    parser.add_argument("--viz", action="store_true", help="Show visualization")
    args = parser.parse_args()

    detector = OrthoHoleDetector(args.ortho, min_hole_area_m2=args.min_area, alpha=args.alpha)
    holes = detector.run()

    print(json.dumps(detector.get_summary(), indent=2))
    detector.export_holes(args.output)

    if args.viz:
        detector.visualize()