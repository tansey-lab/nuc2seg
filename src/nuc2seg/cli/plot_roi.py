import argparse
import logging
from pathlib import Path

import geopandas
import numpy as np
from shapely import box

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
from nuc2seg.plotting import (
    plot_model_class_predictions,
    plot_model_predictions,
    create_interactive_segmentation_comparison,
)
from nuc2seg.utils import (
    get_indexed_tiles,
    transform_bbox_to_slide_space,
    transform_bbox_to_raster_space,
)
from nuc2seg.xenium import (
    load_and_filter_transcripts_as_points,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Segment cell nuclei.")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Ouput dir for plots.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--transcripts",
        help="Xenium transcripts file in parquet format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segments",
        help="GeoParquet segments.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prior-segments",
        help="GeoParquet Prior segments.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        help="Model predictions in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--roi-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format. '
        "Will choose region of interest based on the dataset if not provided.",
    )
    return parser


def get_roi(resolution, labels, size=200):
    pixel_size = int(size / resolution)

    tiles = get_indexed_tiles(
        extent=labels.shape, tile_size=(pixel_size, pixel_size), overlap=0.0
    )

    n_nuclei = {}

    for tile_idx, bounds in tiles.items():
        x1, y1, x2, y2 = bounds
        tile = labels[y1:y2, x1:x2]
        n_nuclei[tile_idx] = len(np.unique(tile[tile > 0]))

    median_n_nuclei = sorted(list(n_nuclei.values()))[len(n_nuclei) // 2]

    for tile_idx, n in n_nuclei.items():
        if n == median_n_nuclei:
            x1, y1, x2, y2 = tiles[tile_idx]
            return x1, y1, x2, y2


def main():
    args = get_parser().parse_args()
    output_dir = Path(args.output_dir)

    log_config.configure_logging(args)

    logger.info("Loading data...")
    ds = Nuc2SegDataset.load_h5(args.dataset)
    predictions = ModelPredictions.load_h5(args.predictions)

    if args.roi_area:
        x1, y1, x2, y2 = [int(x) for x in args.roi_area.split(",")]
        slide_bbox = box(x1, y1, x2, y2)
        raster_bbox = transform_bbox_to_raster_space(slide_bbox, ds.resolution, ds.bbox)
    else:
        raster_bbox = get_roi(ds.resolution, ds.labels)
        slide_bbox = box(
            *transform_bbox_to_slide_space(box(*raster_bbox), ds.resolution, ds.bbox)
        )

    logger.info(f"ROI in slide coordinates: {slide_bbox.bounds}")
    logger.info(f"ROI in raster coordinates: {raster_bbox}")

    segments = geopandas.read_parquet(args.segments)
    prior_segments = geopandas.read_parquet(args.prior_segments)

    plot_model_class_predictions(
        dataset=ds,
        prior_segmentation_gdf=prior_segments,
        segmentation_gdf=segments,
        model_predictions=predictions,
        output_path=str(output_dir / "model_class_predictions.pdf"),
        bbox=slide_bbox,
    )

    plot_model_predictions(
        dataset=ds,
        prior_segmentation_gdf=prior_segments,
        segmentation_gdf=segments,
        model_predictions=predictions,
        output_path=str(output_dir / "model_predictions.pdf"),
        bbox=slide_bbox,
    )

    segments_clip = segments.clip(slide_bbox)
    prior_segments_clip = prior_segments.clip(slide_bbox)
    tx = load_and_filter_transcripts_as_points(args.transcripts, sample_area=slide_bbox)
    create_interactive_segmentation_comparison(
        polygon_gdfs=[prior_segments_clip, segments_clip],
        names=["labels", "nuc2seg"],
        points_gdf=tx,
        output_path=str(output_dir / "segmentation_comparison.html"),
    )
