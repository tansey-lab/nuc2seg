import argparse
import logging
import numpy as np
import torch

from nuc2seg import log_config
from nuc2seg.segment import stitch_predictions
from nuc2seg.unet_model import SparseUNet
from nuc2seg.data import Nuc2SegDataset, TiledDataset

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a UNet model on preprocessed data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output",
        help="Model prediction output in h5 format.",
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
        "--model-weights",
        help="File to read model weights from.",
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
    parser.add_argument(
        "--num-dataloader-workers",
        help="Number of workers to use for the data loader.",
        type=int,
        default=0,
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

    ds = Nuc2SegDataset.load_h5(args.dataset)

    tiled_dataset = TiledDataset(
        ds,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
    )

    model = SparseUNet(600, ds.n_classes + 2, (args.tile_height, args.tile_width))

    model.load_state_dict(torch.load(args.model_weights))

    model_predictions = stitch_predictions(model=model, dataloader=tiled_dataset)

    model_predictions.save_h5(args.output)
