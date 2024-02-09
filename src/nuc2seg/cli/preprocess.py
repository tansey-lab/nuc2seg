import argparse
import logging
import numpy as np

from nuc2seg import log_config
from nuc2seg.xenium_utils import spatial_as_sparse_arrays

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
        "--output-dir",
        help="Output directory.",
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
        "--pixel-stride",
        help="Stride for the pixel grid.",
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
        "--tile-stride",
        help="Stride of the tiles.",
        type=int,
        default=48,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    prng = np.random.default_rng(args.seed)

    spatial_as_sparse_arrays(
        nuclei_file=args.nuclei_file,
        transcripts_file=args.transcripts_file,
        outdir=args.output_dir,
        pixel_stride=args.pixel_stride,
        min_qv=args.min_qv,
        foreground_nucleus_distance=args.foreground_nucleus_distance,
        background_nucleus_distance=args.background_nucleus_distance,
        background_pixel_transcripts=args.background_pixel_transcripts,
        background_transcript_distance=args.background_transcript_distance,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        tile_stride=args.tile_stride,
    )
