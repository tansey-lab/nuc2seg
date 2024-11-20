import argparse
import logging
import h5py

from nuc2seg import log_config
from nuc2seg.utils import get_indexed_tiles

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Get the number of tiles given the tiling parameters (used for pipelining)."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-file",
        help="Text file with number of tiles.",
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
        "--tile-height",
        help="Height of the tiles.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--tile-width",
        help="Width of the tiles.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--overlap-percentage",
        help="What percent of each tile dimension overlaps with the next tile.",
        type=float,
        default=0.25,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    logger.info(f"Loading dataset from {args.dataset}")

    with h5py.File(args.dataset, "r") as f:
        base_width = f["labels"].shape[0]
        base_height = f["labels"].shape[0]

    tile_idx_lookup = get_indexed_tiles(
        extent=(base_width, base_height),
        tile_size=(args.tile_width, args.tile_height),
        overlap=args.overlap_percentage,
    )

    with open(args.output_file, "w") as f:
        f.write(str(len(tile_idx_lookup)))
