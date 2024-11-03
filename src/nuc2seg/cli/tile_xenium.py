import argparse
import logging
import pandas as pd
import math

from nuc2seg import log_config
from nuc2seg.xenium import (
    load_and_filter_transcripts_as_table,
)
from nuc2seg.preprocessing import (
    tile_transcripts_to_disk,
    tile_nuclei_to_disk,
)
from shapely import box

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Divide xenium files into overlapping tiles.", allow_abbrev=False
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--transcript-output-dir",
        help="Directory to save transcript tile files.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--nuclei-output-dir",
        help="Directory to save nuclei tile files.",
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
        "--nuclei-file",
        help="Path to the Xenium nuclei boundaries parquet file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--min-qv",
        help="Minimum quality value for a transcript to be included.",
        type=float,
        default=20.0,
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
        "--output-format",
        help="Output format for transcript files",
        type=str,
        default="csv",
        choices=("csv", "parquet"),
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    if args.sample_area is not None:
        bounds = list(map(int, args.sample_area.split(",")))
        sample_area = box(*bounds)
    else:
        sample_area = None

    transcripts = load_and_filter_transcripts_as_table(
        transcripts_file=args.transcripts,
        sample_area=sample_area,
        min_qv=args.min_qv,
    )

    if args.sample_area is None:
        bounds = [
            0,
            0,
            math.ceil(transcripts["x_location"].max()),
            math.ceil(transcripts["y_location"].max()),
        ]

    mask = (transcripts["cell_id"] > 0) & (transcripts["overlaps_nucleus"].astype(bool))

    transcripts["nucleus_id"] = 0
    transcripts.loc[mask, "nucleus_id"] = transcripts["cell_id"][mask]
    logger.info(f"Writing transcripts to {args.transcript_output_dir}")
    tile_transcripts_to_disk(
        transcripts=transcripts,
        bounds=bounds,
        tile_size=(args.tile_height, args.tile_width),
        overlap=args.overlap_percentage,
        output_dir=args.transcript_output_dir,
        output_format=args.output_format,
    )

    del transcripts

    logger.info("loading nuclei")
    nuclei_df = pd.read_parquet(args.nuclei_file)

    logger.info(f"Writing nuclei to {args.nuclei_output_dir}")
    tile_nuclei_to_disk(
        nuclei_df=nuclei_df,
        bounds=bounds,
        tile_size=(args.tile_height, args.tile_width),
        overlap=args.overlap_percentage,
        output_dir=args.nuclei_output_dir,
        output_format=args.output_format,
    )
