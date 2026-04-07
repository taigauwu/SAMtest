"""
WBT Orthophoto Patcher — PyQGIS Automation
============================================
Stage 4: Automate the WhiteboxTools patching pipeline via PyQGIS.

Mirrors the manual QGIS workflow:
  1. Clip point cloud with generated shapefile
  2. Run LidarNearestNeighbourGridding (WBT)
  3. Merge WBT output with original DJI Terra orthophoto

Dependencies:
    - QGIS 3.x with Python bindings (run inside QGIS Python console or standalone)
    - WhiteboxTools plugin installed in QGIS
    - pip install geopandas rasterio

Usage (standalone — requires QGIS env setup):
    python wbt_patcher.py \
        --ortho georeferenced_ortho.tif \
        --pointcloud georeferenced.laz \
        --clips clip_polygons/ \
        --output patched_ortho.tif

Usage (from QGIS Python console):
    from wbt_patcher import WBTPatcher
    patcher = WBTPatcher(
        ortho_path="ortho.tif",
        pointcloud_path="pointcloud.laz",
        clip_dir="clip_polygons/",
        output_dir="output/"
    )
    patcher.run()
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


def _detect_units_from_crs(filepath: str) -> str:
    """
    Detect linear units from a file's CRS header.

    Reads the CRS from GeoTIFF, LAZ/LAS, or any GDAL/rasterio-supported format
    and returns "meters" or "feet". Falls back to "meters" for geographic CRS
    or if detection fails.
    """
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            crs = src.crs
    except Exception:
        try:
            # Fallback for point clouds: read via laspy or pdal
            import laspy
            with laspy.read(filepath) as las:
                # laspy doesn't always expose CRS cleanly; try VLRs
                for vlr in las.vlrs:
                    if vlr.record_id in (2111, 2112, 34735):
                        # GeoTIFF keys — would need pyproj to parse
                        pass
            log.warning(f"Could not read CRS from {filepath} via laspy, defaulting to meters")
            return "meters"
        except Exception:
            log.warning(f"Could not read CRS from {filepath}, defaulting to meters")
            return "meters"

    if crs is None or crs.is_geographic:
        log.warning("CRS is geographic (lat/lon) or missing — defaulting to meters")
        return "meters"

    # pyproj / rasterio CRS exposes linear units
    try:
        unit_name = crs.linear_units.lower()
        log.info(f"Detected CRS linear units: {unit_name}")

        if any(k in unit_name for k in ["foot", "feet", "ft", "us_foot", "survey_foot"]):
            return "feet"
        return "meters"
    except AttributeError:
        # Fallback: check the EPSG/proj string for unit hints
        wkt = crs.to_wkt()
        wkt_lower = wkt.lower()
        if "foot" in wkt_lower or "feet" in wkt_lower:
            return "feet"
        return "meters"


@dataclass
class PatchConfig:
    """Configuration for the patching pipeline."""
    # Input paths
    ortho_path: str
    pointcloud_path: str
    clip_dir: str  # Directory containing individual clip shapefiles
    output_dir: str = "wbt_output"

    # WBT LidarNearestNeighbourGridding parameters
    interpolation_param: str = "rgb"
    grid_resolution: float = None        # Auto-set from CRS units
    search_radius: float = 0.3

    # Merge settings
    output_dtype: str = "Byte"
    merge_order: str = "wbt_below"      # "wbt_below" = WBT below DJI Terra (recommended)

    # Auto-detected
    units: str = field(default=None, init=False)

    def __post_init__(self):
        # Detect units from the orthophoto CRS header
        self.units = _detect_units_from_crs(self.ortho_path)
        log.info(f"Auto-detected units: {self.units}")

        # Set grid resolution based on detected units (per WBT doc)
        if self.grid_resolution is None:
            self.grid_resolution = 0.15 if self.units == "feet" else 0.05
            log.info(f"Grid resolution set to {self.grid_resolution} ({self.units})")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def _init_qgis():
    """Initialize QGIS application for standalone use."""
    try:
        from qgis.core import QgsApplication
        # Check if already running inside QGIS
        if QgsApplication.instance():
            return QgsApplication.instance(), False

        # Standalone initialization
        qgis_prefix = os.environ.get("QGIS_PREFIX_PATH", "/usr")
        app = QgsApplication([], False)
        app.setPrefixPath(qgis_prefix, True)
        app.initQgis()
        log.info("QGIS initialized (standalone mode)")
        return app, True
    except ImportError:
        log.error("QGIS Python bindings not found. Run from QGIS or set up the environment.")
        sys.exit(1)


class WBTPatcher:
    """
    Automates the WBT orthophoto patching pipeline.

    Pipeline per hole:
      1. Clip the georeferenced point cloud using the generated shapefile
      2. Run LidarNearestNeighbourGridding on the clipped cloud
      3. Merge the WBT raster patch with the original orthophoto
    """

    def __init__(self, ortho_path: str, pointcloud_path: str,
                 clip_dir: str, output_dir: str = "wbt_output",
                 grid_resolution: float = None):
        self.config = PatchConfig(
            ortho_path=ortho_path,
            pointcloud_path=pointcloud_path,
            clip_dir=clip_dir,
            output_dir=output_dir,
            grid_resolution=grid_resolution,
        )
        self.app = None
        self._standalone = False
        self.wbt_outputs: list[str] = []

    def run(self):
        """Execute the full patching pipeline."""
        self.app, self._standalone = _init_qgis()

        clip_files = sorted(Path(self.config.clip_dir).glob("*.shp"))
        if not clip_files:
            log.error(f"No shapefiles found in {self.config.clip_dir}")
            return

        log.info(f"Processing {len(clip_files)} clip regions")

        # Process each clip region
        for i, clip_shp in enumerate(clip_files):
            log.info(f"--- Processing clip {i+1}/{len(clip_files)}: {clip_shp.name} ---")

            # Step 1: Clip point cloud
            clipped_laz = self._clip_pointcloud(clip_shp, i)
            if clipped_laz is None:
                continue

            # Step 2: Run WBT LidarNearestNeighbourGridding
            wbt_raster = self._run_wbt_gridding(clipped_laz, i)
            if wbt_raster:
                self.wbt_outputs.append(wbt_raster)

        # Step 3: Merge all WBT patches with original ortho
        if self.wbt_outputs:
            final_output = self._merge_rasters()
            log.info(f"=== DONE. Final patched ortho: {final_output} ===")
        else:
            log.warning("No WBT outputs generated — nothing to merge")

        if self._standalone:
            self.app.exitQgis()

    def _clip_pointcloud(self, clip_shp: Path, idx: int) -> Optional[str]:
        """
        Clip the point cloud using the shapefile polygon.
        QGIS: Processing Toolbox → Point cloud data management → Clip
        """
        import processing

        output_path = str(Path(self.config.output_dir) / f"clipped_{idx}.laz")

        try:
            result = processing.run("pdal:clip", {
                'INPUT': self.config.pointcloud_path,
                'OVERLAY': str(clip_shp),
                'FILTER_EXPRESSION': '',
                'FILTER_EXTENT': None,
                'OUTPUT': output_path,
            })
            log.info(f"Clipped point cloud → {output_path}")
            return result.get('OUTPUT', output_path)

        except Exception as e:
            log.error(f"Point cloud clip failed: {e}")
            # Fallback: try the native QGIS clip
            try:
                result = processing.run("native:pointcloudclip", {
                    'INPUT': self.config.pointcloud_path,
                    'OVERLAY': str(clip_shp),
                    'OUTPUT': output_path,
                })
                return result.get('OUTPUT', output_path)
            except Exception as e2:
                log.error(f"Fallback clip also failed: {e2}")
                return None

    def _run_wbt_gridding(self, clipped_laz: str, idx: int) -> Optional[str]:
        """
        Run WhiteboxTools LidarNearestNeighbourGridding.
        QGIS: Processing Toolbox → WhiteboxTools → LiDAR Tools → LidarNearestNeighbourGridding

        Parameters (from the WBT workaround doc):
            - Input File: clipped LAZ
            - Interpolation Parameter: rgb
            - Grid Resolution: 0.05 (meters) or 0.15 (feet)
            - Search Radius: 0.3
        """
        import processing

        output_path = str(Path(self.config.output_dir) / f"wbt_patch_{idx}.tif")

        try:
            result = processing.run("wbt:LidarNearestNeighbourGridding", {
                'input': clipped_laz,
                'parameter': self.config.interpolation_param,
                'returns': 'all',
                'resolution': self.config.grid_resolution,
                'radius': self.config.search_radius,
                'exclude_cls': '',
                'minz': None,
                'maxz': None,
                'output': output_path,
            })
            log.info(f"WBT gridding complete → {output_path}")
            return result.get('output', output_path)

        except Exception as e:
            log.error(f"WBT LidarNearestNeighbourGridding failed: {e}")
            log.info("Ensure WhiteboxTools plugin is installed in QGIS")
            return None

    def _merge_rasters(self) -> str:
        """
        Merge all WBT patches with the original orthophoto.
        QGIS: Raster → Miscellaneous → Merge

        IMPORTANT: WBT patches go BELOW the DJI Terra ortho in the input order
        so the original ortho takes priority where data exists.
        Output Data Type must be set to Byte.
        """
        import processing

        final_output = str(Path(self.config.output_dir) / "patched_orthophoto.tif")

        # Build input layer list — order matters!
        # Per the doc: WBT clipped should be BELOW the DJI Terra output
        if self.config.merge_order == "wbt_below":
            input_layers = self.wbt_outputs + [self.config.ortho_path]
        else:
            input_layers = [self.config.ortho_path] + self.wbt_outputs

        try:
            result = processing.run("gdal:merge", {
                'INPUT': input_layers,
                'PCT': False,
                'SEPARATE': False,
                'NODATA_INPUT': None,
                'NODATA_OUTPUT': None,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 1,  # 1 = Byte
                'OUTPUT': final_output,
            })
            log.info(f"Merge complete → {final_output}")
            return result.get('OUTPUT', final_output)

        except Exception as e:
            log.error(f"Raster merge failed: {e}")
            return self._merge_rasters_fallback(input_layers, final_output)

    def _merge_rasters_fallback(self, input_layers: list, output_path: str) -> str:
        """Fallback merge using GDAL directly if processing framework fails."""
        try:
            from osgeo import gdal

            vrt_path = output_path.replace('.tif', '.vrt')
            vrt = gdal.BuildVRT(vrt_path, input_layers)
            gdal.Translate(
                output_path, vrt,
                outputType=gdal.GDT_Byte,
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            )
            vrt = None  # Close
            log.info(f"GDAL fallback merge complete → {output_path}")
            return output_path

        except Exception as e:
            log.error(f"GDAL fallback also failed: {e}")
            return ""


# === Orchestrator: Full Pipeline ===

def run_full_pipeline(ortho_path: str, pointcloud_path: str,
                      output_dir: str = "patched_output",
                      min_hole_area: float = 1.0,
                      alpha: float = 50.0):
    """
    Run the complete AI-powered orthophoto patching pipeline.

    This is the main entry point that chains all stages together:
      Stage 1+2: Detect holes + infer extent
      Stage 3: Generate edge-aware clip polygons
      Stage 4: Execute WBT patching via PyQGIS

    CRS units (meters/feet) are auto-detected from the orthophoto header,
    and WBT grid resolution is set accordingly.

    Args:
        ortho_path: Path to georeferenced orthophoto GeoTIFF
        pointcloud_path: Path to georeferenced point cloud (LAZ/LAS)
        output_dir: Directory for all outputs
        min_hole_area: Minimum hole area in m² to process
        alpha: Concave hull tightness for extent inference
    """
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 & 2: Detection ---
    log.info("=" * 60)
    log.info("STAGE 1 & 2: Hole Detection + Extent Inference")
    log.info("=" * 60)

    from ortho_hole_detector import OrthoHoleDetector
    detector = OrthoHoleDetector(ortho_path, min_hole_area_m2=min_hole_area, alpha=alpha)
    holes = detector.run()

    # Save detection results
    detector.export_holes(str(out / "detected_holes.geojson"))
    detector.export_extent(str(out / "valid_extent.geojson"))
    detector.visualize(str(out / "detection_map.png"))

    summary = detector.get_summary()
    with open(out / "detection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    interior_holes = [h for h in holes if h.is_interior]
    if not interior_holes:
        log.info("No interior holes detected — orthophoto looks clean!")
        return

    log.info(f"Found {len(interior_holes)} interior holes to patch")

    # --- Stage 3: Edge-Aware Clip Generation ---
    log.info("=" * 60)
    log.info("STAGE 3: Edge-Aware Clip Polygon Generation")
    log.info("=" * 60)

    from edge_aware_clipper import EdgeAwareClipper
    clipper = EdgeAwareClipper(ortho_path, detector.transform, detector.crs)
    clip_results = clipper.generate_clips(holes)

    clip_dir = out / "clip_shapefiles"
    clipper.export_individual_clips(str(clip_dir))
    clipper.export_clips(str(out / "all_clips.geojson"))

    # --- Stage 4: WBT Patching ---
    log.info("=" * 60)
    log.info("STAGE 4: Automated WBT Patching")
    log.info("=" * 60)

    patcher = WBTPatcher(
        ortho_path=ortho_path,
        pointcloud_path=pointcloud_path,
        clip_dir=str(clip_dir),
        output_dir=str(out / "wbt_processing"),
    )
    patcher.run()

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"Check output in: {out}")
    log.info("=" * 60)


# === CLI ===

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="AI-powered orthophoto hole patching")
    parser.add_argument("--ortho", required=True, help="Path to georeferenced orthophoto")
    parser.add_argument("--pointcloud", required=True, help="Path to georeferenced point cloud")
    parser.add_argument("--output", default="patched_output", help="Output directory")
    parser.add_argument("--min-area", type=float, default=1.0, help="Min hole area (m²)")
    parser.add_argument("--alpha", type=float, default=50.0, help="Concave hull tightness")

    # Allow running individual stages
    parser.add_argument("--stage", choices=["detect", "clip", "patch", "all"], default="all")
    args = parser.parse_args()

    if args.stage == "all":
        run_full_pipeline(
            ortho_path=args.ortho,
            pointcloud_path=args.pointcloud,
            output_dir=args.output,
            min_hole_area=args.min_area,
            alpha=args.alpha,
        )
    elif args.stage == "detect":
        from ortho_hole_detector import OrthoHoleDetector
        det = OrthoHoleDetector(args.ortho, min_hole_area_m2=args.min_area, alpha=args.alpha)
        det.run()
        det.export_holes(str(Path(args.output) / "holes.geojson"))
        det.visualize()
    else:
        log.info(f"Stage '{args.stage}' requires prior stage outputs. Use --stage all.")