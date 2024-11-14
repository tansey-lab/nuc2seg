import argparse
import logging

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset
from nuc2seg.preprocessing import (
    tile_dataset_to_disk,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Divide xenium files into overlapping tiles.", allow_abbrev=False
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Directory to save tile files.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Nuc2SegDataset in h5 format.",
        type=str,
        required=True,
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

    dataset = Nuc2SegDataset.load_h5(args.dataset)

    tile_dataset_to_disk(
        dataset=dataset,
        output_dir=args.output_dir,
        tile_size=(args.tile_height, args.tile_width),
        overlap=args.overlap_percentage,
    )
