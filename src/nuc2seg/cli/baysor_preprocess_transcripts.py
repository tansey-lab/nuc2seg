import argparse
import logging

from nuc2seg import log_config
from nuc2seg.xenium import (
    load_and_filter_transcripts,
    create_shapely_rectangle,
)
from nuc2seg.preprocessing import tile_transcripts_to_csv

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark cell segmentation given post-Xenium IF data that includes an autofluorescence marker."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Directory to save baysor input CSVs.",
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

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    logger.info(f"Loading transcripts from {args.transcripts}")

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )
        transcripts = load_and_filter_transcripts(
            transcripts_file=args.transcripts,
            sample_area=sample_area,
            min_qv=args.min_qv,
        )
        transcripts["x_location"] = transcripts["x_location"] - sample_area.bounds[0]
        transcripts["y_location"] = transcripts["y_location"] - sample_area.bounds[1]
    else:
        sample_area = None
        transcripts = load_and_filter_transcripts(
            transcripts_file=args.transcripts,
            sample_area=sample_area,
            min_qv=args.min_qv,
        )

    mask = (transcripts["cell_id"] > 0) & (transcripts["overlaps_nucleus"].astype(bool))

    transcripts["nucleus_id"] = 0
    transcripts.loc[mask, "nucleus_id"] = transcripts["cell_id"][mask]

    logger.info(f"Writing CSVs to {args.output_dir}")
    tile_transcripts_to_csv(
        transcripts=transcripts,
        tile_size=(args.tile_height, args.tile_width),
        overlap=args.overlap_percentage,
        output_dir=args.output_dir,
    )
