import argparse
import logging
import numpy as np
import pandas

from nuc2seg import log_config
from nuc2seg.xenium import (
    load_nuclei,
    load_and_filter_transcripts,
    create_shapely_rectangle,
)
from nuc2seg.celltyping import run_cell_type_estimation

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run celltype estimation on nucelus-segmented reads."
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
        "--seed",
        help="Seed to use for PRNG.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--index",
        help="Chain index.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n-chains",
        help="Number of parallel chains with different random initialization for cell typing.",
        type=int,
        default=1,
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
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--max-n-celltypes",
        help="Maximum number of cell types to consider (inclusive).",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--min-n-celltypes",
        help="Minimum number of cell types to consider.",
        type=int,
        default=2,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    seeds = np.random.SeedSequence(args.seed).spawn(args.n_chains)
    rng = np.random.default_rng(seeds[args.index])

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

    celltype_results = run_cell_type_estimation(
        nuclei_geo_df=nuclei_geo_df,
        tx_geo_df=tx_geo_df,
        foreground_nucleus_distance=args.foreground_nucleus_distance,
        min_components=args.min_n_celltypes,
        max_components=args.max_n_celltypes,
        rng=rng,
    )

    celltype_results.save_h5(args.output)
