import argparse
import logging

import numpy as np
import torch
import tqdm

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, TiledDataset
from nuc2seg.segment import stitch_predictions

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Combine partial prediction files into single prediction file."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-file",
        help="Combined predictions in h5 format.",
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
        "--prediction-outputs",
        help="Partial prediction outputs in h5 format.",
        type=str,
        required=True,
        nargs="+",
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


def main():
    args = get_parser().parse_args()

    tile_indices = []
    values = []

    for fn in tqdm.tqdm(args.prediction_outputs, desc="Loading partial predictions"):
        obj = torch.load(fn)
        values.append(obj["predictions"])
        tile_indices.append(obj["tile_indices"])

    values = torch.concatenate(values)
    tile_indices = torch.concatenate(tile_indices)

    sort_idx = np.argsort(tile_indices)
    values = values[sort_idx]

    logger.info(f"Loading dataset from {args.dataset}")

    ds = Nuc2SegDataset.load_h5(args.dataset)

    tiled_dataset = TiledDataset(
        ds,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
    )

    model_predictions = stitch_predictions(results=values, tiler=tiled_dataset.tiler)
    model_predictions.save_h5(args.output_file)
