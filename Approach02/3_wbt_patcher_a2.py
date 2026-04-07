"""
WBT Orthophoto Patcher — Full Orchestrator + PyQGIS Automation
===============================================================
Chains all stages together:
  Stage 1+2: Hole detection + extent inference (SAM 2 context)
  Stage 3:   SAM 2 edge-aware clip polygon generation
  Stage 4:   Point cloud clip → WBT gridding → raster merge

CRS units and grid resolution are auto-detected from the file header.

Dependencies:
    - QGIS 3.x with Python bindings
    - WhiteboxTools QGIS plugin
    - pip install sam-2 torch torchvision rasterio geopandas shapely

Usage (full pipeline):
    python wbt_patcher.py --ortho georef_ortho.tif --pointcloud georef.laz

Usage (from QGIS console):
    from wbt_patcher import run_full_pipeline
    run_full_pipeline("ortho.tif", "pointcloud.laz")
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detect CRS units from file header
# ---------------------------------------------------------------------------

def detect_units_from_crs(filepath: str) -> str:
    """
    Read linear units from a file's CRS header.
    Returns "meters" or "feet". Falls back to "meters" for
    geographic CRS or if detection fails.
    """
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            crs = src.crs
    except Exception:
        log.warning(f"Could not read CRS from {filepath}, defaulting to meters")
        return "meters"

    if crs is None or crs.is_geographic:
        log.warning("CRS is geographic or missing — defaulting to meters")
        return "meters"

    try:
        unit_name = crs.linear_units.lower()
        log.info(f"Detected CRS linear units: {unit_name}")
        if any(k in unit_name for k in ("foot", "feet", "ft", "survey")):
            return "feet"
        return "meters"
    except AttributeError:
        wkt = crs.to_wkt().lower()
        if "foot" in wkt or "feet" in wkt:
            return "feet"
        return "meters"


# ---------------------------------------------------------------------------
# Patch config
# ---------------------------------------------------------------------------

@dataclass
class PatchConfig:
    ortho_path: str
    pointcloud_path: str
    clip_dir: str
    output_dir: str = "wbt_output"

    # WBT parameters (auto-set from CRS)
    interpolation_param: str = "rgb"
    grid_resolution: float = None
    search_radius: float = 0.3

    # Merge settings
    output_dtype: str = "Byte"
    merge_order: str = "wbt_below"

    # Auto-detected
    units: str = field(default=None, init=False)

    def __post_init__(self):
        self.units = detect_units_from_crs(self.ortho_path)
        if self.grid_resolution is None:
            self.grid_resolution = 0.15 if self.units == "feet" else 0.05
        log.info(f"Units: {self.units}, grid resolution: {self.grid_resolution}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# QGIS initialization
# ---------------------------------------------------------------------------

def _init_qgis():
    try:
        from qgis.core import QgsApplication
        if QgsApplication.instance():
            return QgsApplication.instance(), False
        qgis_prefix = os.environ.get("QGIS_PREFIX_PATH", "/usr")
        app = QgsApplication([], False)
        app.setPrefixPath(qgis_prefix, True)
        app.initQgis()
        log.info("QGIS initialized (standalone)")
        return app, True
    except ImportError:
        log.error("QGIS Python bindings not found")
        sys.exit(1)


# ---------------------------------------------------------------------------
# WBT Patcher — Stage 4 execution
# ---------------------------------------------------------------------------

class WBTPatcher:
    """
    Automates the WBT patching pipeline per hole:
      1. Clip point cloud with generated shapefile
      2. Run LidarNearestNeighbourGridding
      3. Merge WBT raster patches with original orthophoto
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
        self.app, self._standalone = _init_qgis()

        clip_files = sorted(Path(self.config.clip_dir).glob("*.shp"))
        if not clip_files:
            log.error(f"No shapefiles in {self.config.clip_dir}")
            return

        log.info(f"Processing {len(clip_files)} clip regions")

        for i, clip_shp in enumerate(clip_files):
            log.info(f"--- Clip {i+1}/{len(clip_files)}: {clip_shp.name} ---")

            # Step 1: Clip the point cloud
            clipped_laz = self._clip_pointcloud(clip_shp, i)
            if clipped_laz is None:
                continue

            # Step 2: NEW - Drop Black Points
            clean_laz = self._drop_black_points(clipped_laz, i)

            # Step 3: Run WBT using the CLEANED laz
            wbt_raster = self._run_wbt_gridding(clean_laz, i)
            if wbt_raster:
                self.wbt_outputs.append(wbt_raster)

        if self.wbt_outputs:
            final = self._merge_rasters()
            log.info(f"=== DONE. Patched ortho: {final} ===")
        else:
            log.warning("No WBT outputs — nothing to merge")

        if self._standalone:
            self.app.exitQgis()

    def _clip_pointcloud(self, clip_shp: Path, idx: int):
        import processing
        output = str(Path(self.config.output_dir) / f"clipped_{idx}.laz")
        try:
            result = processing.run("pdal:clip", {
                'INPUT': self.config.pointcloud_path,
                'OVERLAY': str(clip_shp),
                'FILTER_EXPRESSION': '',
                'FILTER_EXTENT': None,
                'OUTPUT': output,
            })
            log.info(f"Clipped → {output}")
            return result.get('OUTPUT', output)
        except Exception as e:
            log.error(f"Clip failed: {e}")
            try:
                result = processing.run("native:pointcloudclip", {
                    'INPUT': self.config.pointcloud_path,
                    'OVERLAY': str(clip_shp),
                    'OUTPUT': output,
                })
                return result.get('OUTPUT', output)
            except Exception as e2:
                log.error(f"Fallback clip also failed: {e2}")
                return None

    def _run_wbt_gridding(self, clipped_laz: str, idx: int):
        import processing
        output = str(Path(self.config.output_dir) / f"wbt_patch_{idx}.tif")
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
                'output': output,
            })
            log.info(f"WBT gridding → {output}")
            return result.get('output', output)
        except Exception as e:
            log.error(f"WBT gridding failed: {e}")
            return None

    def _merge_rasters(self) -> str:
        import processing
        final = str(Path(self.config.output_dir) / "patched_orthophoto.tif")

        # WBT below DJI Terra so original takes priority where data exists
        if self.config.merge_order == "wbt_below":
            inputs = self.wbt_outputs + [self.config.ortho_path]
        else:
            inputs = [self.config.ortho_path] + self.wbt_outputs

        try:
            result = processing.run("gdal:merge", {
                'INPUT': inputs,
                'PCT': False,
                'SEPARATE': False,
                'NODATA_INPUT': None,
                'NODATA_OUTPUT': None,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 1,  # Byte
                'OUTPUT': final,
            })
            log.info(f"Merged → {final}")
            return result.get('OUTPUT', final)
        except Exception as e:
            log.error(f"Merge failed: {e}, trying GDAL fallback")
            return self._gdal_fallback(inputs, final)

    def _gdal_fallback(self, inputs, output):
        try:
            from osgeo import gdal
            vrt = gdal.BuildVRT(output.replace('.tif', '.vrt'), inputs)
            gdal.Translate(output, vrt,
                           outputType=gdal.GDT_Byte,
                           creationOptions=['COMPRESS=LZW', 'TILED=YES'])
            vrt = None
            log.info(f"GDAL fallback → {output}")
            return output
        except Exception as e:
            log.error(f"GDAL fallback also failed: {e}")
            return ""
    
    def _drop_black_points(self, input_laz: str, idx: int) -> str:
        """Uses las2las to drop RGB 0 0 0 points before WBT interpolation."""
        output_laz = str(Path(self.config.output_dir) / f"clipped_clean_{idx}.laz")
        
        command = [
            "las2las",
            "-i", input_laz,
            "-o", output_laz,
            "-drop_RGB_red", "0", "0",
            "-drop_RGB_green", "0", "0",
            "-drop_RGB_blue", "0", "0",
            "-filter_and"
        ]
        
        try:
            log.info(f"Running las2las to drop black points for patch {idx}...")
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log.info(f"Black points dropped → {output_laz}")
            return output_laz
        except subprocess.CalledProcessError as e:
            log.error(f"las2las failed: {e.stderr.decode()}")
            # Fallback to the original clipped laz if filtering fails
            return input_laz


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(ortho_path: str, pointcloud_path: str,
                      output_dir: str = "patched_output",
                      min_hole_area: float = 1.0,
                      alpha: float = 50.0,
                      sam2_checkpoint: str = "sam2.1_hiera_large.pt",
                      sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                      device: str = "auto"):
    """
    Run the complete SAM 2-powered orthophoto patching pipeline.

    CRS units and WBT grid resolution are auto-detected from the file header.
    SAM 2 is loaded once and shared across Stage 1 (context) and Stage 3 (clipping).

    Args:
        ortho_path: Georeferenced orthophoto GeoTIFF
        pointcloud_path: Georeferenced point cloud (LAZ/LAS)
        output_dir: Output directory for all results
        min_hole_area: Minimum hole area (m²) to process
        alpha: Concave hull tightness for extent inference
        sam2_checkpoint: Path to SAM 2 checkpoint file
        sam2_config: Path to SAM 2 config YAML
        device: "auto", "cuda", or "cpu"
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Stage 1 & 2: Hole Detection + SAM 2 Context ===
    log.info("=" * 60)
    log.info("STAGE 1 & 2: Hole Detection + SAM 2 Context Analysis")
    log.info("=" * 60)

    from ortho_hole_detector import OrthoHoleDetector
    detector = OrthoHoleDetector(
        ortho_path,
        min_hole_area_m2=min_hole_area,
        alpha=alpha,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        device=device,
    )
    holes = detector.run()

    detector.export_holes(str(out / "detected_holes.geojson"))
    detector.export_extent(str(out / "valid_extent.geojson"))

    summary = detector.get_summary()
    with open(out / "detection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    interior = [h for h in holes if h.is_interior]
    if not interior:
        log.info("No interior holes detected — orthophoto is clean!")
        return

    log.info(f"Found {len(interior)} interior holes to patch")

    # === Stage 3: SAM 2 Clip Polygon Generation ===
    log.info("=" * 60)
    log.info("STAGE 3: SAM 2 Edge-Aware Clip Generation")
    log.info("=" * 60)

    from edge_aware_clipper import SAM2Clipper
    clipper = SAM2Clipper(
        ortho_path, detector.transform, detector.crs,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        device=device,
    )
    clipper.generate_clips(holes)

    clip_dir = out / "clip_shapefiles"
    clipper.export_individual_clips(str(clip_dir))
    clipper.export_clips(str(out / "all_clips.geojson"))

    # === Stage 4: WBT Patching ===
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
    log.info(f"Output: {out}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SAM 2-powered orthophoto hole patching")
    parser.add_argument("--ortho", required=True, help="Georeferenced orthophoto GeoTIFF")
    parser.add_argument("--pointcloud", required=True, help="Georeferenced point cloud")
    parser.add_argument("--output", default="patched_output", help="Output directory")
    parser.add_argument("--min-area", type=float, default=1.0, help="Min hole area (m²)")
    parser.add_argument("--alpha", type=float, default=50.0, help="Concave hull tightness")
    parser.add_argument("--sam2-checkpoint", default="sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--stage", choices=["detect", "clip", "patch", "all"], default="all")
    args = parser.parse_args()

    if args.stage == "all":
        run_full_pipeline(
            ortho_path=args.ortho,
            pointcloud_path=args.pointcloud,
            output_dir=args.output,
            min_hole_area=args.min_area,
            alpha=args.alpha,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            device=args.device,
        )
    elif args.stage == "detect":
        from ortho_hole_detector import OrthoHoleDetector
        det = OrthoHoleDetector(
            args.ortho, min_hole_area_m2=args.min_area, alpha=args.alpha,
            sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
            device=args.device,
        )
        det.run()
        det.export_holes(str(Path(args.output) / "holes.geojson"))
    else:
        log.info(f"Stage '{args.stage}' requires prior outputs. Use --stage all.")