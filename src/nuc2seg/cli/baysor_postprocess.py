import argparse
import logging
import pandas as pd
import geopandas as gpd
import pandas
import math
import os.path
import tqdm
import re

from nuc2seg import log_config
from nuc2seg.postprocess import (
    stitch_shapes,
    read_baysor_shapes_with_cluster_assignment,
    filter_baysor_shapes_to_most_significant_nucleus_overlap,
    read_baysor_shapefile,
)
from nuc2seg.plotting import plot_final_segmentation, plot_segmentation_class_assignment
from nuc2seg.xenium import load_nuclei, read_transcripts_into_points
from nuc2seg.segment import convert_transcripts_to_anndata

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Post process tiled baysor results.")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--baysor-shapefiles",
        help="One or more shapefiles output by baysor.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--nuclei-file",
        help="Path to the Xenium nuclei boundaries parquet file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--transcripts",
        help="Xenium transcripts in parquet format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output path.",
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--tile-height",
        help="Height of the tiles.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--tile-width",
        help="Width of the tiles.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--overlap-percentage",
        help="What percent of each tile dimension overlaps with the next tile.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-molecules-per-cell",
        help="Dont output cells with less than this many gene counts.",
        type=int,
        default=None,
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def get_tile_idx(fn):
    fn_clean = os.path.splitext(os.path.basename(fn))[0]
    # search for `tile_{number}` and extract number with regex
    return int(re.search(r"tile_(\d+)", fn_clean).group(1))


def main():
    args = get_args()

    log_config.configure_logging(args)

    transcript_df = read_transcripts_into_points(args.transcripts)

    x_extent = math.ceil(transcript_df["x_location"].astype(float).max())
    y_extent = math.ceil(transcript_df["y_location"].astype(float).max())

    shapefiles_fns = sorted(args.baysor_shapefiles, key=get_tile_idx)

    logger.info("Reading baysor results")
    shape_gdfs = []

    for tile_idx, shapefile_fn in tqdm.tqdm(enumerate(shapefiles_fns)):
        shape_gdf = read_baysor_shapefile(shapes_fn=shapefile_fn)
        shape_gdfs.append(shape_gdf)

    logger.info("Done loading baysor results.")

    logger.info("Stitching shapes")
    stitched_shapes = stitch_shapes(
        shapes=shape_gdfs,
        tile_size=(args.tile_width, args.tile_height),
        base_size=(x_extent, y_extent),
        overlap=args.overlap_percentage,
    )

    logger.info("Loading nuclei shapes")
    nuclei_geo_df = load_nuclei(
        nuclei_file=args.nuclei_file,
        sample_area=None,
    )

    stitched_shapes.to_parquet(args.output)

    logger.info("Filtering baysor shapes to most significant nucleus overlap")
    baysor_nucleus_intersection = (
        filter_baysor_shapes_to_most_significant_nucleus_overlap(
            baysor_shapes=stitched_shapes,
            nuclei_shapes=nuclei_geo_df,
        )
    )

    baysor_nucleus_intersection.to_parquet(
        os.path.join(
            os.path.dirname(args.output), "baysor_nucleus_intersecting_shapes.parquet"
        )
    )

    logger.info("Plotting final segmentation")
    plot_final_segmentation(
        nuclei_gdf=nuclei_geo_df,
        segmentation_gdf=baysor_nucleus_intersection,
        output_path=os.path.join(os.path.dirname(args.output), "segmentation.png"),
    )

    logger.info("Creating anndata")
    adata = convert_transcripts_to_anndata(
        transcript_gdf=transcript_df,
        segmentation_gdf=baysor_nucleus_intersection,
        min_molecules_per_cell=args.min_molecules_per_cell,
    )
    adata.write_h5ad(os.path.join(os.path.dirname(args.output), "anndata.h5ad"))
