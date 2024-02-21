import argparse
import logging
import os.path
import tqdm

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, TiledDataset, ModelPredictions, generate_tiles
from nuc2seg.plotting import plot_model_predictions

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a UNet model on preprocessed data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--predictions",
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
        "--output-dir",
        help="Output directory for plots.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-plots",
        help="Number of plots to make",
        type=int,
        default=100,
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
    predictions = ModelPredictions.load_h5(args.predictions)

    tile_generator = generate_tiles(
        tiler=tiled_dataset.tiler,
        x_extent=ds.x_extent_pixels,
        y_extent=ds.y_extent_pixels,
        tile_size=(args.tile_width, args.tile_height),
        overlap_fraction=args.overlap_percentage,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for bbox in tqdm.tqdm(tile_generator, total=len(tiled_dataset), unit="plot"):
        if ds.labels[bbox[0] : bbox[2], bbox[1] : bbox[3]].count_nonzero() == 0:
            continue

        plot_model_predictions(
            dataset=ds,
            model_predictions=predictions,
            bbox=bbox,
            output_path=os.path.join(
                args.output_dir, "_".join([str(x) for x in bbox]) + ".pdf"
            ),
        )
