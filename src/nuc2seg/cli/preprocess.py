import argparse
import logging
import numpy as np
import pandas
import os.path

from nuc2seg import log_config
from nuc2seg.xenium import (
    load_nuclei,
    load_and_filter_transcripts,
    create_shapely_rectangle,
)
from nuc2seg.celltyping import combine_celltyping_chains
from nuc2seg.preprocessing import create_rasterized_dataset
from nuc2seg.data import CelltypingResults
from nuc2seg.plotting import plot_celltype_estimation_results

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for preprocessing Xenium data for the model."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--transcripts-file",
        help="Path to the Xenium transcripts parquet file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--nuclei-file",
        help="Path to the Xenium nuclei boundaries parquet file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output path.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--celltyping-results",
        help="Path to one or more celltyping results files.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--resolution",
        help="Size of a pixel in microns for rasterization.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-qv",
        help="Minimum quality value for a transcript to be included.",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--foreground-nucleus-distance",
        help="Distance from a nucleus to be considered foreground.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--background-nucleus-distance",
        help="Distance from a nucleus to be considered background.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--background-transcript-distance",
        help="Distance from a transcript to be considered background.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--background-pixel-transcripts",
        help="Number of transcripts in a pixel to be considered background.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )

    else:
        df = pandas.read_parquet(args.transcripts_file)
        y_max = df["y_location"].max()
        x_max = df["x_location"].max()

        sample_area = create_shapely_rectangle(0, 0, x_max, y_max)

    nuclei_geo_df = load_nuclei(
        nuclei_file=args.nuclei_file,
        sample_area=sample_area,
    )

    tx_geo_df = load_and_filter_transcripts(
        transcripts_file=args.transcripts_file,
        sample_area=sample_area,
        min_qv=args.min_qv,
    )

    rasterized_dataset = create_rasterized_dataset(
        nuclei_geo_df=nuclei_geo_df,
        tx_geo_df=tx_geo_df,
        sample_area=sample_area,
        resolution=args.resolution,
        foreground_nucleus_distance=args.foreground_nucleus_distance,
        background_nucleus_distance=args.background_nucleus_distance,
        background_pixel_transcripts=args.background_pixel_transcripts,
        background_transcript_distance=args.background_transcript_distance,
    )

    celltyping_chains = [CelltypingResults.load_h5(x) for x in args.celltyping_results]
    celltyping_results, aic_scores, bic_scores = combine_celltyping_chains(
        celltyping_chains
    )

    plot_celltype_estimation_results(
        aic_scores,
        bic_scores,
        celltyping_results.final_expression_profiles,
        celltyping_results.final_prior_probs,
        celltyping_results.final_cell_types,
        celltyping_results.relative_expression,
        os.path.join(args.output_dir, "cell_typing_plots"),
    )
