import argparse
import logging
import pandas as pd
import geopandas as gpd
import pandas
import math
import os.path
import tqdm

from nuc2seg import log_config
from nuc2seg.postprocess import stitch_shapes, read_baysor_results
from nuc2seg.plotting import plot_final_segmentation, plot_segmentation_class_assignment
from nuc2seg.xenium import load_nuclei
from nuc2seg.segment import convert_transcripts_to_anndata

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark cell segmentation given post-Xenium IF data that includes an autofluorescence marker."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--baysor-shapefiles",
        help="One or more shapefiles output by baysor.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--baysor-transcript-assignments",
        help="One or more transcript assignment files output by baysor.",
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

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    transcript_df = pandas.read_parquet(args.transcripts)

    x_extent = math.ceil(transcript_df["x_location"].astype(float).max())
    y_extent = math.ceil(transcript_df["y_location"].astype(float).max())

    shapefiles = sorted(
        args.baysor_shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    transcript_assignment_files = sorted(
        args.baysor_transcript_assignments,
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    if len(shapefiles) != len(transcript_assignment_files):
        raise ValueError(
            "Number of shapefiles and transcript assignment files must be the same."
        )

    zipped_fns = list(enumerate(zip(shapefiles, transcript_assignment_files)))

    logger.info("Reading baysor results")
    all_shapes = None
    all_tx = None

    for tile_idx, (shapefile_fn, tx_fn) in tqdm.tqdm(zipped_fns):
        shape_gdf, tx_gdf = read_baysor_results(
            shapes_fn=shapefile_fn,
            transcripts_fn=tx_fn,
            tile_idx=tile_idx,
            tile_size=(args.tile_width, args.tile_height),
            base_size=(x_extent, y_extent),
            overlap=args.overlap_percentage,
        )

        if all_shapes is None:
            all_shapes = shape_gdf
        else:
            all_shapes = gpd.GeoDataFrame(
                pd.concat([all_shapes, shape_gdf], ignore_index=True)
            )

        if all_tx is None:
            all_tx = tx_gdf
        else:
            all_tx = pd.concat([all_tx, tx_gdf], ignore_index=True)

    logger.info("Done loading baysor results.")

    stitched_shapes = stitch_shapes(
        shapes=shape_gdfs,
        tile_size=(args.tile_width, args.tile_height),
        base_size=(x_extent, y_extent),
        overlap=args.overlap_percentage,
    )

    nuclei_geo_df = load_nuclei(
        nuclei_file=args.nuclei_file,
        sample_area=None,
    )

    plot_final_segmentation(
        nuclei_gdf=nuclei_geo_df,
        segmentation_gdf=stitched_shapes,
        output_path=os.path.join(os.path.dirname(args.output), "segmentation.png"),
    )
    plot_segmentation_class_assignment(
        segmentation_gdf=stitched_shapes,
        output_path=os.path.join(os.path.dirname(args.output), "class_assignment.png"),
        cat_column="cluster",
    )

    stitched_shapes.to_parquet(args.output)
