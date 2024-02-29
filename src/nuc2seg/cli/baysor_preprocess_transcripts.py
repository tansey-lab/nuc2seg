import argparse
import logging
import pandas
import os.path

from nuc2seg import log_config
from nuc2seg.xenium import (
    load_and_filter_transcripts,
    create_shapely_rectangle,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark cell segmentation given post-Xenium IF data that includes an autofluorescence marker."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-path",
        help="Destination for transcript CSV.",
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

    else:
        df = pandas.read_parquet(args.transcripts_file)
        y_max = df["y_location"].max()
        x_max = df["x_location"].max()

        sample_area = create_shapely_rectangle(0, 0, x_max, y_max)

    transcripts = load_and_filter_transcripts(
        transcripts_file=args.transcripts,
        sample_area=sample_area,
        min_qv=args.min_qv,
    )
    mask = (transcripts["cell_id"] > 0) & (transcripts["overlaps_nucleus"].astype(bool))

    transcripts["nucleus_id"] = 0
    transcripts.loc[mask, "nucleus_id"] = transcripts["cell_id"][mask]

    logger.info(f"Writing CSV to {args.output_path}")
    transcripts.to_csv(os.path.join(args.output_path), index=False)
